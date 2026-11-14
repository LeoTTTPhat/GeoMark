"""Stage 4, Prop. 4 / P3 check: is the KEY-AVERAGED output distribution of a watermarked generator equal to the
unwatermarked one? Per-release utility (results/utility/) uses ONE key for all traces of a release, so cross-trace seed
reuse makes a single-key release differ from the unwatermarked generator even for distortion-free Gumbel-H. Here we pool
releases of 50 keys (100..149, 200 traces each; release set 'keymix') into two disjoint replicate pools of 5,000 traces
(100 traces per key each) and compute the same utility metrics as run_utility.py against the same real test traces,
next to single-key releases (first 5,000 traces of <wm>__k{0,1,2}__s0) and unwatermarked releases (none__k0__s{0,1,2}).
Output: <out_dir>/utility/keymix.jsonl. Usage: python src/run_keymix.py --cfg CFG"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, load_pickle, append_jsonl, done_keys, Timer
from geomark.utility import utility
from geomark.data import load_ds, release_dir
from run_utility import real_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    rs = cfg['releases']['keymix']
    uc = cfg['utility']
    out = os.path.join(cfg['out_dir'], 'utility', 'keymix.jsonl')
    done = done_keys(out)
    for gen in rs['gens']:
        g = cfg['generators'][gen]
        ds, res = g['dataset'], g['res']
        real = real_cells(cfg, ds, res, 'test', uc['n'], 0)           # identical to run_utility.py
        D = load_ds(cfg, ds)
        train_set = {tuple(int(x) for x in D['tok'][res][i]) for i, t in enumerate(D['traces']) if t['split'] == 'train'}
        rd = release_dir(cfg, gen)
        items = []
        for wm in rs['wm']:
            per_key = []
            for key in rs['keys']:
                f = os.path.join(rd, f'{wm}__k{key}__s0.pkl')
                if os.path.exists(f):
                    per_key.append(load_pickle(f)['traces'])
            if len(per_key) < len(rs['keys']):
                print(f'{gen} {wm}: only {len(per_key)} keymix releases; skipping', flush=True); continue
            half = rs['n'] // 2
            for rep in (0, 1):
                items.append((f'{gen}|{wm}|keymix{len(per_key)}|rep{rep}', wm, 'keymix',
                              [tr['cells'] for trs in per_key for tr in trs[rep * half:(rep + 1) * half]]))
            for key in (0, 1, 2):
                f = os.path.join(rd, f'{wm}__k{key}__s0.pkl')
                items.append((f'{gen}|{wm}|single_k{key}', wm, 'single', [tr['cells'] for tr in load_pickle(f)['traces'][:uc['n']]]))
        for s in (0, 1, 2):
            f = os.path.join(rd, f'none__k0__s{s}.pkl')
            items.append((f'{gen}|none|s{s}', 'none', 'none', [tr['cells'] for tr in load_pickle(f)['traces'][:uc['n']]]))
        for k, wm, kind, cells in items:
            if k in done:
                continue
            T = Timer()
            u = utility(cells, real, ds, res, train_set)
            append_jsonl(out, dict(key=k, gen=gen, wm=wm, kind=kind, n=len(cells), sec=T(), **u))
            print(f"keymix {k}: n={len(cells)} bigram JSD {u['jsd_bigram']:.4f} length JSD {u['jsd_length']:.4f} "
                  f"distinct-trigram {u['distinct_trigram_ratio']:.3f}", flush=True)


if __name__ == '__main__':
    main()
