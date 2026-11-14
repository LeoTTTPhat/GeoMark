"""Stage 8: step realism of every release (reviewer: the KGW green bias pulls the unrestricted sampler towards
NON-ADJACENT cells, a distortion that bigram JSD hides).

For the first 3,000 traces of each release (seed-0 key-0..2 releases of the main, rev1 and rev2 sets, both small-GPT
generators and both Qwen generators) and for 3,000 real test traces: share of consecutive collapsed r9 cells that are
not H3 neighbours (grid distance > 1), mean / 99th-percentile grid distance per step, mean step length (km between
centroids) and the JSD of the grid-distance distribution (bins 1, 2, 3, 4-5, 6-9, >= 10) to the real test traces.
Output: results/rev2/steps.json (key = gen|rid). Model-free. Usage: python src/rev2/steps.py"""
import os, sys
import numpy as np
import h3.api.basic_int as h3i
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, real_cells
from geomark.utility import jsd
from geomark.common import LocalProj, Timer

GENS = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']
WMS = ['none', 'exact_kgw', 'exact_kgw_bin', 'gm_kgw', 'gm_kgw_eosn', 'gm_kgw_sn', 'a_g25_d4', 'a_g25_d4_eosn', 'gm_gumbel',
       'gm_gumbel_m2', 'a_k0', 'a_1r', 'k0_1r', 'exact_bin_2r', 'exact_bin_eosn', 'posthoc']
EDGES = [1, 2, 3, 4, 6, 10, 10 ** 9]


def step_stats(seqs, ds):
    gd, km = [], []
    pr = LocalProj.for_dataset(ds)
    for c in seqs:
        c = [int(x) for x in c]
        for a, b in zip(c[:-1], c[1:]):
            try:
                gd.append(h3i.grid_distance(a, b))
            except Exception:
                gd.append(99)
        if len(c) > 1:
            ll = np.array([h3i.cell_to_latlng(x) for x in c])
            x, y = pr.to_xy(ll[:, 0], ll[:, 1])
            km += list(np.hypot(np.diff(x), np.diff(y)) / 1000)
    gd = np.array(gd); km = np.array(km)
    hist = np.histogram(gd, EDGES)[0].astype(float)
    return dict(n_steps=int(len(gd)), frac_nonadj=float((gd > 1).mean()), mean_gd=float(gd.mean()), p99_gd=float(np.percentile(gd, 99)),
                mean_step_km=float(km.mean()), hist=hist.tolist())


def main():
    cfg = cfg_rev2()
    out = jload2('steps.json')
    T = Timer()
    for gen in GENS:
        g = cfg['generators'][gen]
        ds = g['dataset']
        rk = f'{gen}|real_test'
        if rk not in out:
            out[rk] = step_stats(real_cells(cfg, ds, 9, 'test', n=3000, seed=0)[0], ds)
        ref = np.array(out[rk]['hist'])
        for wm in WMS:
            for key in ((0,) if wm == 'none' else (0, 1, 2)):
                rid = f'{wm}__k{key}__s0'
                k = f'{gen}|{rid}'
                if k in out:
                    continue
                R = load_release(cfg, gen, rid)
                if R is None:
                    continue
                rec = step_stats([t['cells'] for t in R['traces'][:3000]], ds)
                rec['jsd_gd'] = float(jsd(np.array(rec['hist']), ref))
                out[k] = rec
                print(f"{k}: nonadj {rec['frac_nonadj']:.4f} mean gd {rec['mean_gd']:.2f} p99 {rec['p99_gd']:.0f} "
                      f"step {rec['mean_step_km']:.3f} km jsd {rec['jsd_gd']:.4f} ({T():.0f}s)", flush=True)
        jdump2(out, 'steps.json')


if __name__ == '__main__':
    main()
