"""Small decoder-only GPT over cell tokens (MobilityGPT-style), with KV cache and left-padding support.
Token layout: 0 PAD, 1 BOS, 2 EOS, 3..3+N_SLOTS-1 start-slot tokens, then V cell tokens."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import N_SLOTS

PAD, BOS, EOS, SLOT0 = 0, 1, 2, 3
CELL0 = SLOT0 + N_SLOTS


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = drop

    def forward(self, x, mask, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, 2)
        sh = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if cache is not None:
            if cache.get('k') is not None:
                k = torch.cat([cache['k'], k], 2); v = torch.cat([cache['v'], v], 2)
            cache['k'], cache['v'] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_cells, d_model=384, n_layer=6, n_head=6, max_len=130, dropout=0.1):
        super().__init__()
        self.n_vocab = CELL0 + n_cells
        self.cfg = dict(n_cells=n_cells, d_model=d_model, n_layer=n_layer, n_head=n_head, max_len=max_len, dropout=dropout)
        self.tok = nn.Embedding(self.n_vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.n_vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, pos=None, mask=None, caches=None):
        B, T = idx.shape
        if pos is None:
            pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok(idx) + self.pos(pos.clamp(max=self.cfg['max_len'] - 1))
        for i, b in enumerate(self.blocks):
            x = b(x, mask, None if caches is None else caches[i])
        return self.head(self.lnf(x))

    def loss(self, idx):
        """Next-token loss on right-padded batches."""
        T = idx.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        logits = self(idx[:, :-1], mask=mask[:T - 1, :T - 1])
        tgt = idx[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), ignore_index=PAD)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class GPTAdapter:
    """Generation adapter: logits over [cells..., EOS] for the sampler in sampler.py."""

    def __init__(self, model, dev):
        self.m, self.dev = model.eval(), dev
        self.V = model.cfg['n_cells']

    def _to_task(self, logits):
        lg = logits[:, -1, :].float()
        return torch.cat([lg[:, CELL0:CELL0 + self.V], lg[:, EOS:EOS + 1]], 1).cpu()

    @torch.no_grad()
    def start(self, slots, prefixes=None):
        seqs = [[BOS, SLOT0 + int(s)] + ([CELL0 + int(t) for t in prefixes[i]] if prefixes else []) for i, s in enumerate(slots)]
        T = max(len(s) for s in seqs)
        B = len(seqs)
        idx = torch.full((B, T), PAD, dtype=torch.long)
        valid = torch.zeros(B, T, dtype=torch.bool)
        for i, s in enumerate(seqs):        # left-pad
            idx[i, T - len(s):] = torch.tensor(s); valid[i, T - len(s):] = True
        self.valid = valid.to(self.dev)
        self.pos = (valid.long().cumsum(1) - 1).clamp(min=0).to(self.dev)
        causal = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
        mask = (causal.unsqueeze(0) & self.valid.unsqueeze(1)).unsqueeze(1)   # (B,1,T,T)
        mask = mask | torch.eye(T, dtype=torch.bool, device=self.dev).view(1, 1, T, T)  # avoid all-False rows
        self.caches = [dict() for _ in self.m.blocks]
        lg = self.m(idx.to(self.dev), pos=self.pos, mask=mask, caches=self.caches)
        self.next_pos = self.pos[:, -1:] + 1
        return self._to_task(lg)

    def select(self, idx):
        idx = idx.to(self.dev)
        for c in self.caches:
            c['k'], c['v'] = c['k'][idx], c['v'][idx]
        self.valid, self.next_pos = self.valid[idx], self.next_pos[idx]

    @torch.no_grad()
    def step(self, task_tokens):
        tok = torch.where(task_tokens >= self.V, torch.full_like(task_tokens, EOS), task_tokens + CELL0)
        self.valid = torch.cat([self.valid, torch.ones(len(tok), 1, dtype=torch.bool, device=self.dev)], 1)
        mask = self.valid.unsqueeze(1).unsqueeze(1)   # (B,1,1,Ttot)
        lg = self.m(tok.view(-1, 1).to(self.dev), pos=self.next_pos, mask=mask, caches=self.caches)
        self.next_pos = self.next_pos + 1
        return self._to_task(lg)

    @torch.no_grad()
    def teacher_forced_top1(self, seqs, slots, bs=256):
        """For radioactivity reading mode: argmax next cell (task index; V = EOS) at every position of each sequence.
        Returns list of arrays aligned with positions 0..len-1 (prediction for cell t given cells < t)."""
        out = []
        for a in range(0, len(seqs), bs):
            chunk = seqs[a:a + bs]
            T = max(len(s) for s in chunk) + 2
            idx = torch.full((len(chunk), T), PAD, dtype=torch.long)
            for i, s in enumerate(chunk):
                idx[i, :len(s) + 2] = torch.tensor([BOS, SLOT0 + int(slots[a + i])] + [CELL0 + int(t) for t in s])
            mask = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
            lg = self.m(idx.to(self.dev), mask=mask).float()
            task = torch.cat([lg[:, :, CELL0:CELL0 + self.V], lg[:, :, EOS:EOS + 1]], 2)
            # forbid repeating the previous cell (as during sampling)
            for i, s in enumerate(chunk):
                for t in range(1, len(s)):
                    task[i, t + 1, int(s[t - 1])] = -float('inf')
            am = task.argmax(2).cpu().numpy()
            for i, s in enumerate(chunk):
                out.append(am[i, 1:1 + len(s)])   # logits at position p predict token p+1; cell t sits at p = t+2
        return out
