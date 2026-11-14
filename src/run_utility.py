"""E7 utility: distributional JSDs, diversity, memorisation for every release of the given sets (vs real test traces;
real-val vs real-test gives the noise floor), and a next-location downstream task (small GPT trained on synthetic,
tested on real; acc@1/acc@5). Output: <out_dir>/utility/<gen>.jsonl (key = rid), <out_dir>/utility/nextloc.jsonl.
Usage: python src/run_utility.py --cfg CFG --sets main,llm [--nextloc]"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, load_pickle, append_jsonl, done_keys, Timer, device, free_torch, seed_from
from geomark.utility import utility, nearest_jaccard
from geomark.data import load_ds, release_dir, get_vocab, real_sequences, to_task
from generate import release_jobs


def real_cells(cfg, ds, res, split, n, seed):
    D = load_ds(cfg, ds)
    idx = [i for i, t in enumerate(D['traces']) if t['split'] == split]
    rng = np.random.default_rng(seed)
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return [D['tok'][res][i] for i in idx]


@torch.no_grad()
def nextloc_acc(model, seqs, slots, dev, bs=256):
    from geomark.gpt import BOS, SLOT0, CELL0, PAD
    c1 = c5 = n = 0
    for a in range(0, len(seqs), bs):
        ch = seqs[a:a + bs]
        T = max(len(s) for s in ch) + 2
        x = torch.full((len(ch), T), PAD, dtype=torch.long)
        for i, s in enumerate(ch):
            x[i, :len(s) + 2] = torch.tensor([BOS, SLOT0 + slots[a + i]] + [CELL0 + t for t in s])
        lg = model(x.to(dev), mask=torch.ones(T, T, dtype=torch.bool, device=dev).tril()).float()
        for i, s in enumerate(ch):
            if len(s) < 2:
                continue
            pred = lg[i, 2:len(s) + 1]              # predicts cells 1..len-1
            tgt = torch.tensor([CELL0 + t for t in s[1:]], device=dev)
            top5 = pred.topk(5, -1).indices
            c1 += int((top5[:, 0] == tgt).sum()); c5 += int((top5 == tgt[:, None]).any(1).sum()); n += len(s) - 1
    return c1 / max(n, 1), c5 / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--sets', default='main')
    ap.add_argument('--nextloc', action='store_true')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    uc = cfg['utility']
    od = os.path.join(cfg['out_dir'], 'utility')
    os.makedirs(od, exist_ok=True)
    for gen, g in cfg['generators'].items():
        ds, res = g['dataset'], g['res']
        jobs = [j for sn in a.sets.split(',') for j in release_jobs(cfg, gen, [sn])]
        if not jobs:
            continue
        out = os.path.join(od, f'{gen}.jsonl')
        done = done_keys(out)
        real = real_cells(cfg, ds, res, 'test', uc['n'], 0)
        D = load_ds(cfg, ds)
        train_set = {tuple(int(x) for x in D['tok'][res][i]) for i, t in enumerate(D['traces']) if t['split'] == 'train'}
        train_cells = real_cells(cfg, ds, res, 'train', uc.get('n_mem_ref', 20000), 1)
        if 'real_val' not in done:
            u = utility(real_cells(cfg, ds, res, 'val', uc['n'], 2), real, ds, res, train_set)
            u['nn_jaccard'] = nearest_jaccard(real_cells(cfg, ds, res, 'val', uc['n'], 2), train_cells, n=uc.get('n_mem', 500))
            append_jsonl(out, dict(key='real_val', **u))
        for j in jobs:
            rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
            f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
            if rid in done or not os.path.exists(f):
                continue
            T = Timer()
            R = load_pickle(f)['traces'][:uc['n']]
            from geomark.hexgrid import tokenize
            cells = [tr['cells'] if 'cells' in tr and cfg['wm'][j['wm']]['method'] != 'wtrace' else tokenize(tr['lat'], tr['lon'], res) for tr in R]
            u = utility(cells, real, ds, res, train_set)
            u['nn_jaccard'] = nearest_jaccard(cells, train_cells, n=uc.get('n_mem', 500))
            if cfg['wm'][j['wm']]['method'] == 'posthoc':
                u['posthoc_frac_changed'] = float(np.mean([tr.get('frac_changed', 0) for tr in R]))
            append_jsonl(out, dict(key=rid, wm=j['wm'], wm_key=j['key'], seed=j['seed'], sec=T(), **u))
            print(f'utility {gen}/{rid}: bigram JSD {u["jsd_bigram"]:.4f} ({T():.0f}s)', flush=True)
    if a.nextloc:
        from train_gpt import train_model
        nc = uc['nextloc']
        out = os.path.join(od, 'nextloc.jsonl')
        done = done_keys(out)
        dev = device()
        for gen, rid in [x.split('/') for x in nc['releases']]:
            key = f'{gen}/{rid}'
            if key in done:
                continue
            cap = nc['n_train']
            if '@' in rid:                      # Stage 4: 'rid@N' = train on the first N traces (equal-n comparisons)
                rid, cap = rid.split('@')[0], min(int(rid.split('@')[1]), nc['n_train'])
            g = cfg['generators'][gen]
            vocab = get_vocab(cfg, g['dataset'], g['res'])
            index = {int(c): i for i, c in enumerate(vocab)}
            if rid == 'real':
                tr, trs = real_sequences(cfg, g['dataset'], g['res'], 'train', index, g['max_len'], limit=nc['n_train'], seed=0)
            else:
                f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
                if not os.path.exists(f):
                    continue
                R = load_pickle(f)['traces'][:cap]
                tr, trs = [], []
                for x in R:
                    s = to_task(x['cells'], index)
                    if s and len(s) >= 2:
                        tr.append(s); trs.append(x['slot'])
            te, tes = real_sequences(cfg, g['dataset'], g['res'], 'test', index, g['max_len'], limit=nc['n_test'], seed=1)
            mc = dict(nc['model'], max_len=g['max_len'])
            mdir = os.path.join(cfg['model_dir'], 'nextloc', gen, rid + (f'@{cap}' if key.endswith(f'@{cap}') else ''))
            m = train_model(mdir, len(vocab), tr, trs, te[:500], tes[:500], mc, seed=0)
            a1, a5 = nextloc_acc(m, te, tes, dev)
            append_jsonl(out, dict(key=key, gen=gen, rid=rid, cap=cap, acc1=a1, acc5=a5, n_train=len(tr), n_test=len(te)))
            print(f'nextloc {key}: acc@1 {a1:.4f} acc@5 {a5:.4f}', flush=True)
            del m; free_torch()


if __name__ == '__main__':
    main()
