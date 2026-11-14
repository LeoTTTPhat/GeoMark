"""Watermarked sampling over a cell vocabulary (+ EOS), model-agnostic.

* none:       multinomial from softmax(logits)
* kgw:        GeoMark-KGW-2R: logits + delta*1[u_{s,c}<gamma] + delta_mid*1[v_{s,par(c)}<gamma]
* exact_kgw:  baseline: logits + delta*1[u_{s,c}<gamma] with s = previous exact fine cell
* gumbel:     GeoMark-Gumbel-H (distortion-free): P* = argmax_P v_{s,P}^{1/p(P)}, then c* = argmax_{c in P*} u_{s,c}^{1/p(c|P*)}
Masked steps (policy M1) are sampled without watermark. All sampling happens on CPU with a seeded generator.
The generator adapter supplies logits over [cells..., EOS] (EOS = index V).
"""
import numpy as np
import torch
from .prf import PRF, EOS_ID
from .scheme import SeedState, is_masked
from .hexgrid import parents


class VocabInfo:
    def __init__(self, cells, res):
        self.cells = np.asarray(cells, dtype=np.int64)          # H3 ints of the vocabulary (index = task id)
        self.V = len(self.cells)
        self.res = res
        mids = parents(self.cells, res - 1)
        self.mids, self.mid_idx = np.unique(mids, return_inverse=True)
        self.M = len(self.mids)
        self.index = {int(c): i for i, c in enumerate(self.cells)}
        self.cells_eos = np.append(self.cells, EOS_ID)
        self.mids_eos = np.append(self.mids, EOS_ID)
        self.mid_idx_eos = torch.as_tensor(np.append(self.mid_idx, self.M), dtype=torch.long)
        self._adj = None

    def adj_mask(self, prevs):
        """Stage 8: bool (len(prevs), V+1) mask of the H3 neighbours (grid distance 1) of each previous vocab index
        (all False for prev < 0; EOS never adjacent)."""
        if self._adj is None:
            from .hexgrid import neighbors
            self._adj = [torch.as_tensor([self.index[x] for x in neighbors(int(c), 1) if x in self.index], dtype=torch.long)
                         for c in self.cells]
        m = torch.zeros(len(prevs), self.V + 1, dtype=torch.bool)
        for j, p in enumerate(prevs):
            if p is not None and p >= 0 and len(self._adj[p]):
                m[j, self._adj[p]] = True
        return m


class WMSampler:
    def __init__(self, sch, key_id, vocab, gen_seed, cache_max=20000):
        self.sch, self.vocab = sch, vocab
        self.prf = PRF(key_id) if sch.method not in ('none',) else None
        self.gen = torch.Generator().manual_seed(int(gen_seed))
        self.cache, self.cache_max = {}, cache_max

    def vectors(self, seed):
        v = self.cache.get(seed)
        if v is None:
            # float64 throughout: float32 rounding could map u to exactly 1.0 (Gumbel argmax tie) or flip u < gamma
            # relative to the detector's float64 comparison
            u = torch.as_tensor(self.prf.vector('fine', seed, self.vocab.cells_eos), dtype=torch.float64)
            m = torch.as_tensor(self.prf.vector('mid', seed, self.vocab.mids_eos), dtype=torch.float64)
            if len(self.cache) >= self.cache_max:
                self.cache.clear()
            v = self.cache[seed] = (u, m)
        return v

    def sample(self, logits, seeds, masked, prevs=None):
        """logits: (B, V+1) float CPU; seeds: list of seed tuples; masked: list of bool; prevs: previous vocab index per
        row (-1 if none; used by the step-neutral sampler). Returns LongTensor (B,)."""
        sch = self.sch
        logits = logits.float()
        B = logits.shape[0]
        out = torch.empty(B, dtype=torch.long)
        wm_rows = [i for i in range(B) if sch.method != 'none' and not masked[i]]
        wm_set = set(wm_rows)
        plain = [i for i in range(B) if i not in wm_set]
        if plain:
            pr = torch.softmax(logits[plain], -1)
            out[plain] = torch.multinomial(pr, 1, generator=self.gen).squeeze(1)
        if not wm_rows:
            return out
        L = logits[wm_rows]
        U = torch.stack([self.vectors(seeds[i])[0] for i in wm_rows])
        Vm = torch.stack([self.vectors(seeds[i])[1] for i in wm_rows])
        mid_idx = self.vocab.mid_idx_eos
        V = self.vocab.V
        if sch.method in ('kgw', 'exact_kgw', 'exact_kgw_bin'):
            bias = sch.delta * (U < sch.gamma).float()
            if sch.delta_mid and (sch.method == 'kgw' or (sch.method == 'exact_kgw_bin' and sch.two_res)):
                bias = bias + sch.delta_mid * (Vm[:, mid_idx] < sch.gamma).float()
            bias[:, V] = 0.0  # EOS never biased
            if getattr(sch, 'adj_neutral', False) and prevs is not None:
                # Stage 8 step-neutral: groups {EOS}, {non-neighbours of prev}, {neighbours of prev}; keep every group's
                # unwatermarked mass and bias only the choice inside the neighbour group (first step: EOS-neutral)
                A = self.vocab.adj_mask([prevs[i] for i in wm_rows])
                has = A.any(1, keepdim=True)
                cell = torch.ones_like(A); cell[:, V] = False
                A = torch.where(has, A, cell)                      # no previous cell: one group of all cells
                Nn = cell & ~A
                p0 = torch.softmax(L.double(), -1)
                ninf = torch.full_like(L, -float('inf'))
                pa = (p0 * A).sum(1, keepdim=True); pn = (p0 * Nn).sum(1, keepdim=True); pe = p0[:, V:V + 1]
                ca = torch.nan_to_num(torch.softmax(torch.where(A, L + bias, ninf).double(), -1), nan=0.0)
                cn = torch.nan_to_num(torch.softmax(torch.where(Nn, L, ninf).double(), -1), nan=0.0)
                pr = (pa * ca + pn * cn)
                pr[:, V:V + 1] = pe
                pr = pr.float().clamp_min(0)
            elif getattr(sch, 'eos_neutral', False):
                # Stage 7: keep the unwatermarked stop probability; bias only the cell choice given continuation
                p0 = torch.softmax(L.double(), -1)
                pe = p0[:, V:V + 1]
                pc = torch.nan_to_num(torch.softmax((L + bias)[:, :V].double(), -1), nan=0.0)   # rows with only EOS
                pr = torch.cat([(1.0 - pe) * pc, pe], 1).float().clamp_min(0)
            else:
                pr = torch.softmax(L + bias, -1)
            out[wm_rows] = torch.multinomial(pr, 1, generator=self.gen).squeeze(1)
        elif sch.method == 'gumbel':
            pr = torch.softmax(L.double(), -1)
            pmid = torch.zeros(len(wm_rows), self.vocab.M + 1, dtype=torch.float64).index_add_(1, mid_idx, pr)
            with torch.no_grad():
                smid = torch.where(pmid > 0, torch.log(Vm.double()) / pmid.clamp_min(1e-300), torch.full_like(pmid, -np.inf))
                Pstar = smid.argmax(1)                                           # (b,)
                inP = mid_idx.unsqueeze(0) == Pstar.unsqueeze(1)                 # (b, V+1)
                pc = torch.where(inP, pr, torch.zeros_like(pr))
                pc = pc / pc.sum(1, keepdim=True).clamp_min(1e-300)
                sc = torch.where(pc > 0, torch.log(U.double()) / pc.clamp_min(1e-300), torch.full_like(pc, -np.inf))
                out[wm_rows] = sc.argmax(1)
        else:
            raise ValueError(sch.method)
        return out


def generate(adapter, sch, key_id, vocab, slots, gen_seed, max_len=126, prefixes=None, wm_on_prefix=False,
             forbid_repeat=True, top_k=0):
    """Autoregressive generation of collapsed cell sequences. Returns list of lists of vocab indices (no EOS).
    prefixes (optional): list of lists of vocab indices to force (re-generation attack; the continuation is sampled
    with scheme `sch`, typically method 'none')."""
    B = len(slots)
    smp = WMSampler(sch, key_id, vocab, gen_seed)
    states = [SeedState(sch, s) for s in slots]
    seen = [set() for _ in range(B)]
    seqs = [[] for _ in range(B)]
    if prefixes:
        for i, pf in enumerate(prefixes):
            for t in pf:
                states[i].update(vocab.cells[t]); seqs[i].append(int(t))
    logits = adapter.start(slots, prefixes)
    done = [False] * B
    rows = list(range(B))        # global index of each row currently in the adapter batch
    V = vocab.V
    steps = max(1, max_len - (max(len(pf) for pf in prefixes) if prefixes else 0))
    can_select = hasattr(adapter, 'select')
    for step in range(steps):
        lg = logits.float().cpu().clone()
        if forbid_repeat:
            for j, i in enumerate(rows):
                if seqs[i]:
                    lg[j, seqs[i][-1]] = -float('inf')
        if step == 0:
            lg[:, V] = -float('inf')     # at least one cell
        if top_k:
            kth = torch.topk(lg, top_k, dim=1).values[:, -1:]
            lg = torch.where(lg < kth, torch.full_like(lg, -float('inf')), lg)
        seeds, masked = [], []
        for i in rows:
            s = states[i].seed()
            masked.append(is_masked(sch, len(seqs[i]), s, states[i].prev, seen[i]))
            seeds.append(s)
        nxt = smp.sample(lg, seeds, masked, prevs=[seqs[i][-1] if seqs[i] else -1 for i in rows])
        for j, i in enumerate(rows):
            if done[i]:
                continue
            t = int(nxt[j])
            if t == V:
                done[i] = True
            else:
                seqs[i].append(t); states[i].update(vocab.cells[t])
        if all(done) or step == steps - 1:
            break
        alive = [j for j, i in enumerate(rows) if not done[i]]
        if can_select and len(rows) - len(alive) >= max(1, len(rows) // 4):   # drop finished rows (batch compaction)
            adapter.select(torch.tensor(alive, dtype=torch.long))
            rows = [rows[j] for j in alive]
            nxt = nxt[alive]
        dmask = torch.tensor([done[i] for i in rows])
        nxt = torch.where(dmask, torch.full_like(nxt, V), nxt)
        logits = adapter.step(nxt)
    return seqs
