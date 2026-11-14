"""Stage 7: corrected utility metrics (reviewers: the length / gyration JSDs of geomark/utility.py drop every generated
trace beyond the real 99th percentile because np.histogram discards out-of-range values).

For every release that has a record in results/utility/<gen>.jsonl (first 5,000 traces, same real test reference as
run_utility.py), for the Stage-7 releases (rev1_* sets), for the key-mixed pools (keymix / rev1_gumkeymix) and for
real-vs-real, this computes
  * jsd_*_coded : the original (truncating) JSD, to document the bug,
  * jsd_*_ovf   : JSD with the same 40 bins on [0, P99(real)] PLUS one overflow bin [P99, inf) (nothing dropped),
  * w1_*_km     : Wasserstein-1 distance (km) between generated and real distributions (unbinned),
  * ks_*        : two-sample Kolmogorov-Smirnov statistic,
  * mean / median trip length in km and in cell tokens, share of generated trips above the real P99,
  * OD / cell-popularity / bigram JSD and duplicate share (unchanged definitions), nearest-training-trace Jaccard
    (memorisation; 500 traces vs 20,000 training traces, as run_utility.py) and exact-memorisation share.
Output: results/rev1/utility_fixed.jsonl (key = gen|rid). Model-free. Usage: python src/rev1/utility.py --cfg CFG"""
import os, sys, argparse, glob, re
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_pickle, append_jsonl, read_jsonl
from geomark.common import load_cfg, Timer
from geomark.utility import utility, nearest_jaccard, cell_length_km, jsd
from geomark.data import load_ds, release_dir
from geomark.hexgrid import tokenize
from run_utility import real_cells


def hist_ovf(x, edges):
    h = np.histogram(np.minimum(x, edges[-1]), np.append(edges[:-1], np.inf))[0]   # last bin [edges[-2], inf)
    return h.astype(float)


def fixed_metrics(gen_cells, real_cells_, ds, train_set, train_cells):
    gl = np.array([cell_length_km(c, ds) for c in gen_cells]); rl = np.array([cell_length_km(c, ds) for c in real_cells_])
    out = {}
    for j, name in ((0, 'length'), (1, 'gyration')):
        g, r = gl[:, j], rl[:, j]
        p99 = max(1e-3, np.percentile(r, 99))
        e = np.linspace(0, p99, 41)
        out[f'jsd_{name}_coded'] = jsd(np.histogram(g, e)[0].astype(float), np.histogram(r, e)[0].astype(float))
        e2 = np.append(e, np.inf)
        out[f'jsd_{name}_ovf'] = jsd(np.histogram(g, e2)[0].astype(float), np.histogram(r, e2)[0].astype(float))
        out[f'w1_{name}_km'] = float(stats.wasserstein_distance(g, r))
        out[f'ks_{name}'] = float(stats.ks_2samp(g, r).statistic)
        out[f'frac_above_p99_{name}'] = float((g > p99).mean())
        out[f'mean_{name}_km_gen'] = float(g.mean()); out[f'median_{name}_km_gen'] = float(np.median(g))
        out[f'mean_{name}_km_real'] = float(r.mean()); out[f'median_{name}_km_real'] = float(np.median(r))
    lt = np.array([len(c) for c in gen_cells]); lr = np.array([len(c) for c in real_cells_])
    out.update(mean_tok_gen=float(lt.mean()), median_tok_gen=float(np.median(lt)), mean_tok_real=float(lr.mean()),
               median_tok_real=float(np.median(lr)), w1_tok=float(stats.wasserstein_distance(lt, lr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--gens', default=None)
    ap.add_argument('--max_items', type=int, default=0, help='testing: stop after this many records')
    ap.add_argument('--out', default='utility_fixed.jsonl')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg) if a.cfg else cfg_rev1()
    uc = cfg['utility']
    out = os.path.join(OUT, a.out)
    n_new = 0
    done = {r['key'] for r in read_jsonl(out)}
    ud = os.path.join(cfg['out_dir'], 'utility')
    gens = sorted({os.path.basename(f)[:-6] for f in glob.glob(os.path.join(ud, '*.jsonl'))} - {'keymix', 'nextloc'})
    if a.gens:
        gens = [x for x in gens if x in a.gens.split(',')]
    for gen in gens:
        g = cfg['generators'][gen]
        ds, res = g['dataset'], g['res']
        rids = [r['key'] for r in read_jsonl(os.path.join(ud, f'{gen}.jsonl')) if r['key'] != 'real_val']
        rd = release_dir(cfg, gen)
        # Stage-7 releases of this generator
        for sn, rs in cfg['releases'].items():
            if not sn.startswith(('rev1_', 'rev2_')) or gen not in rs['gens'] or sn in ('rev1_manykey', 'rev1_gumkeymix', 'rev2_manykey', 'rev2_steal'):
                continue
            for wm in rs['wm']:
                for key in ([0] if cfg['wm'][wm]['method'] == 'none' else rs['keys']):
                    for s in rs['seeds']:
                        rids.append(f'{wm}__k{key}__s{s}')
        todo = [r for r in ['real_val'] + rids + ['keymix:gm_gumbel', 'keymix:gm_kgw', 'keymix:gm_gumbel_m2']
                if f'{gen}|{r}' not in done]
        if not todo:
            continue
        real = real_cells(cfg, ds, res, 'test', uc['n'], 0)
        D = load_ds(cfg, ds)
        train_set = {tuple(int(x) for x in D['tok'][res][i]) for i, t in enumerate(D['traces']) if t['split'] == 'train'}
        train_cells = real_cells(cfg, ds, res, 'train', uc.get('n_mem_ref', 20000), 1)
        for rid in todo:
            if a.max_items and n_new >= a.max_items:
                return
            n_new += 1
            T = Timer()
            meta = {}
            if rid == 'real_val':
                cells = real_cells(cfg, ds, res, 'val', uc['n'], 2)
            elif rid.startswith('keymix:'):
                wm = rid.split(':')[1]
                keys = range(100, 150)
                per = [os.path.join(rd, f'{wm}__k{k}__s0.pkl') for k in keys]
                if not all(os.path.exists(f) for f in per):
                    continue
                cells = [tr['cells'] for f in per for tr in load_pickle(f)['traces'][:100]]   # rep0 of run_keymix.py
                meta['n_keys'] = 50
            else:
                f = os.path.join(rd, rid + '.pkl')
                if not os.path.exists(f):
                    continue
                R = load_pickle(f)['traces'][:uc['n']]
                wm = rid.split('__')[0]
                meth = cfg['wm'].get(wm, {}).get('method', '')
                cells = [tr['cells'] if ('cells' in tr and meth != 'wtrace') else tokenize(tr['lat'], tr['lon'], res) for tr in R]
            u = utility(cells, real, ds, res, train_set)
            u.update(fixed_metrics(cells, real, ds, train_set, train_cells))
            u['nn_jaccard'] = nearest_jaccard(cells, train_cells, n=uc.get('n_mem', 500))
            rec = dict(key=f'{gen}|{rid}', gen=gen, rid=rid, n=len(cells), sec=T(), **meta, **u)
            append_jsonl(out, rec)
            print(f"{gen}|{rid}: n={len(cells)} length JSD coded {u['jsd_length_coded']:.3f} ovf {u['jsd_length_ovf']:.3f} "
                  f"W1 {u['w1_length_km']:.2f} km, mean {u['mean_length_km_gen']:.2f} vs {u['mean_length_km_real']:.2f} km "
                  f"({T():.0f}s)", flush=True)


if __name__ == '__main__':
    main()
