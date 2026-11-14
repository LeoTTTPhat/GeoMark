"""E10 capacity on real traces: distinct (seed, cell) pairs per trace vs collapsed length for each resolution, history
length k and time-bin width; minimum attainable KGW p-value gamma^n; seed-space size and observations per seed.
Output: <out_dir>/capacity/capacity.json. Usage: python src/run_capacity.py --cfg CFG"""
import os, sys, argparse
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, dump_json, Timer
from geomark.scheme import Scheme
from geomark.detect import pair_sets
from geomark.data import load_ds


def q(x):
    x = np.asarray(x, dtype=float)
    return dict(p10=float(np.percentile(x, 10)), p50=float(np.median(x)), p90=float(np.percentile(x, 90)), mean=float(x.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    cc = cfg['capacity']
    out = os.path.join(cfg['out_dir'], 'capacity', 'capacity.json')
    if os.path.exists(out):
        print('exists'); return
    res_all = {}
    T = Timer()
    for ds in cfg['datasets']:
        D = load_ds(cfg, ds)
        rng = np.random.default_rng(0)
        idx = [i for i, t in enumerate(D['traces']) if t['split'] == 'test']
        idx = rng.choice(idx, size=min(cc['n'], len(idx)), replace=False)
        for r in cfg['data']['res']:
            for k in cc['k']:
                for b in cc['bins']:
                    sch = Scheme(method='kgw', res=r, k=k, bin_min=b)
                    npairs, nmid, lens, seeds = [], [], [], Counter()
                    for i in idx:
                        cells, slot = D['tok'][r][i], D['traces'][i]['slot']
                        f, m = pair_sets(cells, slot, sch)
                        npairs.append(len(f)); nmid.append(len(m)); lens.append(len(cells))
                        for s, _ in f:
                            seeds[s] += 1
                    npairs = np.array(npairs); lens = np.array(lens)
                    edges = [0, 10, 20, 40, 80, 1000]
                    by_len = {f'{lo}-{hi}': float(np.median(npairs[(lens >= lo) & (lens < hi)])) if ((lens >= lo) & (lens < hi)).any() else None
                              for lo, hi in zip(edges[:-1], edges[1:])}
                    res_all[f'{ds}_r{r}_k{k}_b{b}'] = dict(
                        ds=ds, res=r, k=k, bin_min=b, n_traces=len(idx), pairs_fine=q(npairs), pairs_mid=q(nmid), collapsed_len=q(lens),
                        pairs_by_len=by_len, frac_ge20=float((npairs >= 20).mean()), frac_ge25=float((npairs >= 25).mean()),
                        min_log10p_gamma025=q(npairs * np.log10(0.25)), distinct_seeds=len(seeds),
                        median_obs_per_seed=float(np.median(list(seeds.values()))) if seeds else 0.0)
            print(f'{ds} r{r} done ({T():.0f}s)', flush=True)
    dump_json(res_all, out)


if __name__ == '__main__':
    main()
