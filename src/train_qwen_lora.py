"""E1b/E1c: LoRA fine-tuning of Qwen2.5-*-Instruct as a trajectory generator over new cell tokens.
Vocabulary surgery: one new token per fine cell (<cI>), per start slot (<sJ>) and an end token (<eot>); only the new
embedding rows are trainable (PEFT trainable_token_indices, tied with lm_head), plus LoRA on all linear layers.
Step-level checkpoints with resume. Usage: python src/train_qwen_lora.py --cfg src/configs/full.yaml --gen qwen15_porto_r9"""
import os, sys, math, argparse, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from geomark.common import load_cfg, set_seed, scaled, device, free_torch, dump_json, Timer, append_jsonl, N_SLOTS
from geomark.data import get_vocab, real_sequences

PROMPT = 'Synthesize one {ds} mobility trajectory as H3 cell tokens. Start slot:'


def new_tokens(V):
    return [f'<c{i}>' for i in range(V)] + [f'<s{j}>' for j in range(N_SLOTS)] + ['<eot>']


def build(base, V, dtype, lora_r=16, adapter_dir=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(base)
    n0 = len(tok)
    tok.add_tokens(new_tokens(V), special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    with torch.no_grad():  # deterministic init of the new rows: mean of the old embeddings
        E = model.get_input_embeddings().weight
        E[n0:] = E[:n0].float().mean(0).to(E.dtype)
    new_ids = list(range(n0, len(tok)))
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        lc = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type='CAUSAL_LM',
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                        trainable_token_indices={'embed_tokens': new_ids})
        model = get_peft_model(model, lc)
    ids = dict(cell0=n0, slot0=n0 + V, eot=n0 + V + N_SLOTS, n0=n0)
    return tok, model, ids


def encode(tok, ids, ds, seq, slot):
    pr = tok(PROMPT.format(ds=ds), add_special_tokens=False).input_ids + [ids['slot0'] + int(slot)]
    body = [ids['cell0'] + int(t) for t in seq] + [ids['eot']]
    return pr + body, len(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--gen', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    g = cfg['generators'][a.gen]
    out = os.path.join(cfg['model_dir'], a.gen)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'done.json')):
        print('done already'); return
    set_seed(g.get('train_seed', 0))
    vocab = get_vocab(cfg, g['dataset'], g['res'])
    index = {int(c): i for i, c in enumerate(vocab)}
    tr, trs = real_sequences(cfg, g['dataset'], g['res'], 'train', index, g['max_len'], limit=g.get('max_train'), seed=g.get('train_seed', 0))
    va, vas = real_sequences(cfg, g['dataset'], g['res'], 'val', index, g['max_len'], limit=g.get('n_val', 500))
    dev = device()
    ck = os.path.join(out, 'ckpt')
    tok, model, ids = build(g['base'], len(vocab), g.get('dtype', 'float32'), g.get('lora_r', 16))
    model.to(dev)
    model.print_trainable_parameters()
    enc = [encode(tok, ids, g['dataset'], s, sl) for s, sl in zip(tr, trs)]
    venc = [encode(tok, ids, g['dataset'], s, sl) for s, sl in zip(va, vas)]
    bs = scaled(g['batch'], 1)
    accum = max(1, g.get('eff_batch', g['batch']) // bs)
    steps = math.ceil(len(enc) * g['epochs'] / (bs * accum))
    if g.get('max_steps'):
        steps = min(steps, g['max_steps'])
    params = [q for q in model.parameters() if q.requires_grad]
    tokp = [q for n, q in model.named_parameters() if q.requires_grad and 'trainable_tokens' in n]
    restp = [q for n, q in model.named_parameters() if q.requires_grad and 'trainable_tokens' not in n]
    opt = torch.optim.AdamW([dict(params=restp, lr=g['lr']), dict(params=tokp, lr=g['lr'] * g.get('tok_lr_mult', 5.0))], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    step0 = 0
    if os.path.exists(os.path.join(ck, 'state.pt')):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(os.path.join(ck, 'state.pt'), map_location='cpu')
        set_peft_model_state_dict(model, load_file(os.path.join(ck, 'adapter_model.safetensors')))
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); step0 = st['step']
        print(f'resumed at step {step0}', flush=True)
    rng = np.random.default_rng(g.get('train_seed', 0))
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(int(math.ceil(g['epochs'])) + 1)])

    def batch(ix):
        b = [enc[i] for i in ix]
        T = max(len(x[0]) for x in b)
        inp = torch.full((len(b), T), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((len(b), T), -100, dtype=torch.long)
        att = torch.zeros((len(b), T), dtype=torch.long)
        for i, (x, npr) in enumerate(b):
            inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            lab[i, npr:len(x)] = torch.tensor(x[npr:])
        return inp.to(dev), att.to(dev), lab.to(dev)

    T = Timer()
    model.train()
    for step in range(step0, steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(accum):
            a0 = (step * accum + j) * bs
            inp, att, lab = batch(order[a0:a0 + bs])
            loss = model(input_ids=inp, attention_mask=att, labels=lab).loss / accum
            loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            print(f'step {step}/{steps} loss {tot:.3f} ({T():.0f}s)', flush=True)
            append_jsonl(os.path.join(out, 'train_log.jsonl'), dict(step=step, loss=tot, sec=T()))
        if (step + 1) % g.get('ckpt_every', 200) == 0 and step + 1 < steps:
            model.save_pretrained(ck)
            torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1), os.path.join(ck, 'state.pt'))
    model.eval()
    with torch.no_grad():
        vl, n = 0.0, 0
        for a0 in range(0, len(venc), bs):
            b = venc[a0:a0 + bs]
            T_ = max(len(x[0]) for x in b)
            inp = torch.full((len(b), T_), tok.pad_token_id or 0, dtype=torch.long); lab = torch.full_like(inp, -100); att = torch.zeros_like(inp)
            for i, (x, npr) in enumerate(b):
                inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1; lab[i, npr:len(x)] = torch.tensor(x[npr:])
            nt = int((lab != -100).sum())
            vl += float(model(input_ids=inp.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss) * nt; n += nt
    model.save_pretrained(out)
    tok.save_pretrained(out)
    import shutil
    shutil.rmtree(ck, ignore_errors=True)
    dump_json(dict(steps=steps, n_train=len(enc), val_loss=vl / max(n, 1), sec=T(), cfg=g, ids=ids), os.path.join(out, 'done.json'))
    print(f'val loss {vl / max(n, 1):.4f}; saved {out} ({T():.0f}s)')
    del model; free_torch()


class QwenAdapter:
    """Generation adapter (KV cache, left padding) returning logits over [cells..., EOS]."""

    def __init__(self, cfg, gen_name):
        from transformers import AutoTokenizer
        g = cfg['generators'][gen_name]
        self.g, self.ds = g, g['dataset']
        vocab = get_vocab(cfg, g['dataset'], g['res'])
        self.V = len(vocab)
        out = os.path.join(cfg['model_dir'], gen_name)
        self.tok, model, self.ids = build(g['base'], self.V, g.get('dtype', 'float32'), adapter_dir=out)
        self.dev = device()
        self.m = model.merge_and_unload().to(self.dev).eval() if g.get('merge', True) else model.to(self.dev).eval()
        self.prompt = self.tok(PROMPT.format(ds=self.ds), add_special_tokens=False).input_ids
        # Project only onto the task tokens ([cells..., eot]) instead of the full ~152k vocabulary: identical logits for the
        # task tokens (lm_head has no bias), ~15-25% less compute per step and no (B x 152k) logits tensor.
        base = self.m.get_base_model() if hasattr(self.m, 'get_base_model') else self.m
        self.body = base.model
        W = base.get_output_embeddings().weight
        c0 = self.ids['cell0']
        idx = torch.tensor(list(range(c0, c0 + self.V)) + [self.ids['eot']], device=W.device)
        self.Wsub = W.index_select(0, idx).detach().clone()

    def _task(self, hidden):
        return (hidden[:, -1, :] @ self.Wsub.T).float().cpu()

    def _fwd(self, **kw):
        return self.body(**kw)

    @torch.no_grad()
    def start(self, slots, prefixes=None):
        from transformers import DynamicCache
        seqs = [self.prompt + [self.ids['slot0'] + int(s)] + ([self.ids['cell0'] + int(t) for t in prefixes[i]] if prefixes else [])
                for i, s in enumerate(slots)]
        T = max(len(s) for s in seqs)
        inp = torch.zeros((len(seqs), T), dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, T - len(s):] = torch.tensor(s); att[i, T - len(s):] = 1
        self.att = att.to(self.dev)
        pos = (self.att.cumsum(1) - 1).clamp(min=0)
        self.cache = DynamicCache()
        o = self._fwd(input_ids=inp.to(self.dev), attention_mask=self.att, position_ids=pos, past_key_values=self.cache, use_cache=True)
        self.cache = o.past_key_values
        self.next_pos = pos[:, -1:] + 1
        out = self._task(o.last_hidden_state)
        if not getattr(self, '_checked', False):   # one-time check: sliced projection == full-model task logits
            ref = self.m(input_ids=inp[:2].to(self.dev), attention_mask=self.att[:2], position_ids=pos[:2]).logits[:, -1, :].float()
            c0 = self.ids['cell0']
            ref = torch.cat([ref[:, c0:c0 + self.V], ref[:, self.ids['eot']:self.ids['eot'] + 1]], 1).cpu()
            err = float((ref - out[:2]).abs().max()); scale = float(ref.abs().max())
            print(f'QwenAdapter logit check: max abs diff {err:.4f} (logit scale {scale:.2f})', flush=True)
            assert err <= 0.02 * max(scale, 1.0), 'sliced lm_head projection does not match the full model'
            self._checked = True
        return out

    def select(self, idx):
        idx = idx.to(self.dev)
        self.cache.batch_select_indices(idx)
        self.att, self.next_pos = self.att[idx], self.next_pos[idx]

    @torch.no_grad()
    def step(self, task_tokens):
        t = torch.where(task_tokens >= self.V, torch.full_like(task_tokens, self.ids['eot']), task_tokens + self.ids['cell0'])
        self.att = torch.cat([self.att, torch.ones((len(t), 1), dtype=torch.long, device=self.dev)], 1)
        o = self._fwd(input_ids=t.view(-1, 1).to(self.dev), attention_mask=self.att, position_ids=self.next_pos,
                      past_key_values=self.cache, use_cache=True)
        self.cache = o.past_key_values
        self.next_pos = self.next_pos + 1
        return self._task(o.last_hidden_state)


if __name__ == '__main__':
    main()
