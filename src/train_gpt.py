"""E1a: train the small GPT generator over cell tokens (also used for radioactivity students and the next-location
utility model). Per-epoch checkpoints with resume; done flag. Usage:
  python src/train_gpt.py --cfg src/configs/full.yaml --gen gpt_porto_r9_s0"""
import os, sys, json, math, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, set_seed, scaled, device, free_torch, dump_json, Timer, append_jsonl
from geomark.gpt import GPT, BOS, EOS, SLOT0, CELL0, PAD
from geomark.data import get_vocab, real_sequences


def encode(seq, slot):
    return [BOS, SLOT0 + int(slot)] + [CELL0 + int(t) for t in seq] + [EOS]


def batches(enc, bs, rng):
    order = rng.permutation(len(enc))
    for a in range(0, len(order), bs):
        b = [enc[i] for i in order[a:a + bs]]
        T = max(len(x) for x in b)
        x = torch.full((len(b), T), PAD, dtype=torch.long)
        for i, s in enumerate(b):
            x[i, :len(s)] = torch.tensor(s)
        yield x


@torch.no_grad()
def eval_loss(model, enc, dev, bs=256):
    model.eval()
    tot, n = 0.0, 0
    for x in batches(enc, bs, np.random.default_rng(0)):
        x = x.to(dev)
        ntok = int((x[:, 1:] != PAD).sum())
        tot += float(model.loss(x)) * ntok; n += ntok
    model.train()
    return tot / max(n, 1)


def train_model(out_dir, n_cells, train_seqs, train_slots, val_seqs, val_slots, mc, seed, log=print):
    """Train (or resume) a GPT; returns the model on device. mc: model/training config dict."""
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, 'done.json')
    dev = device()
    set_seed(seed)
    model = GPT(n_cells, d_model=mc['d_model'], n_layer=mc['n_layer'], n_head=mc['n_head'], max_len=mc['max_len'] + 4,
                dropout=mc.get('dropout', 0.1)).to(dev)
    if os.path.exists(done):
        model.load_state_dict(torch.load(os.path.join(out_dir, 'model.pt'), map_location=dev))
        return model.eval()
    enc = [encode(s, sl) for s, sl in zip(train_seqs, train_slots)]
    venc = [encode(s, sl) for s, sl in zip(val_seqs, val_slots)][:5000]
    opt = torch.optim.AdamW(model.parameters(), lr=mc['lr'], weight_decay=0.1, betas=(0.9, 0.95))
    bs = scaled(mc['batch'], 8)
    steps_per_epoch = math.ceil(len(enc) / bs)
    total = steps_per_epoch * mc['epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ep0 = 0
    ck = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); ep0 = st['epoch']
        log(f'resumed from epoch {ep0}')
    T = Timer()
    hist = []
    for ep in range(ep0, mc['epochs']):
        rng = np.random.default_rng(seed * 1000 + ep)
        model.train()
        for i, x in enumerate(batches(enc, bs, rng)):
            loss = model.loss(x.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if i % 200 == 0:
                log(f'  ep {ep} step {i}/{steps_per_epoch} loss {float(loss):.3f} ({T():.0f}s)')
        vl = eval_loss(model, venc, dev) if venc else float('nan')
        hist.append(dict(epoch=ep + 1, val_loss=vl, sec=T()))
        log(f'epoch {ep + 1}: val loss {vl:.4f} ({T():.0f}s)')
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1), ck)
        append_jsonl(os.path.join(out_dir, 'train_log.jsonl'), hist[-1])
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    dump_json(dict(n_params=model.n_params(), n_train=len(enc), history=hist, mc=mc, seed=seed, sec=T()), done)
    if os.path.exists(ck):
        os.remove(ck)
    return model.eval()


def load_gpt(cfg, gen_name):
    gc_ = cfg['generators'][gen_name]
    d = os.path.join(cfg['model_dir'], gen_name)
    vocab = get_vocab(cfg, gc_['dataset'], gc_['res'])
    dev = device()
    model = GPT(len(vocab), d_model=gc_['d_model'], n_layer=gc_['n_layer'], n_head=gc_['n_head'], max_len=gc_['max_len'] + 4).to(dev)
    model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location=dev))
    return model.eval(), vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--gen', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    for gen in a.gen.split(','):          # several generators in one guarded step (fewer memguard slot waits)
        g = cfg['generators'][gen]
        assert g['type'] == 'gpt'
        vocab = get_vocab(cfg, g['dataset'], g['res'])
        index = {int(c): i for i, c in enumerate(vocab)}
        tr, trs = real_sequences(cfg, g['dataset'], g['res'], 'train', index, g['max_len'], limit=g.get('max_train'), seed=g.get('train_seed', 0))
        va, vas = real_sequences(cfg, g['dataset'], g['res'], 'val', index, g['max_len'], limit=5000)
        print(f'{gen}: vocab {len(vocab)}, train {len(tr)}, val {len(va)}', flush=True)
        m = train_model(os.path.join(cfg['model_dir'], gen), len(vocab), tr, trs, va, vas, g, seed=g.get('train_seed', 0))
        print(gen, 'params', m.n_params(), flush=True)
        del m; free_torch()


if __name__ == '__main__':
    main()
