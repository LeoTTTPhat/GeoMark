"""Stage 7: corrected key-mixture test of distortion-freeness (P3).

The Stage-4 'keymix' releases (keys 100..149, 200 traces each) all used sampling seed 0, and in generate.py both the
start slots and the sampler's random stream depend on the sampling seed only, not on the key. The 50 keys of a key mix
therefore share the same 100 start slots and the same random stream for unwatermarked (masked) steps, which inflates
duplicates and divergences of the mixture independently of the watermark. Here every key k in 100..149 gets its own
sampling seed 1000 + k (100 traces per key, 5,000 per mixture), for the unwatermarked generator (the matched reference),
GeoMark-KGW (plain and EOS-neutral) and Gumbel-H with masking M1 and M2. The utility metrics of src/rev1/utility.py are
computed on each 5,000-trace mixture against the same real test traces. Releases: gen/<gen>/<wm>__k<k>__s<1000+k>.pkl;
output: results/rev1/keymix2.jsonl. Loads one small model per worker (CPU). Usage: python src/rev1/keymix2.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_pickle, append_jsonl, read_jsonl, ppath
from rev1.utility import fixed_metrics
from geomark.common import scaled, Timer
from geomark.utility import utility, nearest_jaccard
from geomark.data import load_ds, release_dir
from run_utility import real_cells
import generate as G

WMS = ['none', 'gm_kgw', 'gm_kgw_eosn', 'gm_gumbel', 'gm_gumbel_m2']
KEYS = list(range(100, 150))


def main():
    cfg = cfg_rev1()
    cfg_path = ppath('src', 'configs', 'rev1.yaml')
    out = os.path.join(OUT, 'keymix2.jsonl')
    done = {r['key'] for r in read_jsonl(out)}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0'):
        g = cfg['generators'][gen]
        jobs = [dict(set='rev1_keymix2', wm=wm, key=k, seed=1000 + k, n=100, decode='uniform') for wm in WMS for k in KEYS]
        G.run_parallel(cfg, cfg_path, gen, jobs, scaled(4, 1), 'cpu', 6)
        ds, res = g['dataset'], g['res']
        real = real_cells(cfg, ds, res, 'test', 5000, 0)
        D = load_ds(cfg, ds)
        train_set = {tuple(int(x) for x in D['tok'][res][i]) for i, t in enumerate(D['traces']) if t['split'] == 'train'}
        train_cells = real_cells(cfg, ds, res, 'train', 20000, 1)
        for wm in WMS:
            k = f'{gen}|{wm}|keymix2'
            if k in done:
                continue
            T = Timer()
            cells = []
            for kk in KEYS:
                R = load_pickle(os.path.join(release_dir(cfg, gen), f'{wm}__k{kk}__s{1000 + kk}.pkl'))
                cells += [tr['cells'] for tr in R['traces']]
            u = utility(cells, real, ds, res, train_set)
            u.update(fixed_metrics(cells, real, ds, None, None))
            u['nn_jaccard'] = nearest_jaccard(cells, train_cells, n=500)
            append_jsonl(out, dict(key=k, gen=gen, wm=wm, n=len(cells), n_keys=len(KEYS), sec=T(), **u))
            print(f"{k}: n={len(cells)} bigram {u['jsd_bigram']:.4f} length(ovf) {u['jsd_length_ovf']:.4f} OD {u['jsd_od']:.4f} "
                  f"dup {u['duplicate_trace_frac']:.3f} mean tok {u['mean_tok_gen']:.1f}", flush=True)


if __name__ == '__main__':
    main()
