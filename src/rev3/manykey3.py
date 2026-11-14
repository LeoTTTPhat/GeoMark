"""Stage 10 (round-3 audit): the 50-key study with exact hashing + start bin under the SAME two-resolution embedding as
GeoMark-KGW (reviewers: the 50-key comparison 'GeoMark-KGW minus exact+bin' compared a two-resolution scheme with a
single-resolution one). Reads results/eval/gpt_porto_r9_s0 (pooled.jsonl / parquet) like src/rev2/stats2.py.
Output: results/rev3/manykey3.json. Model-free. Usage: python src/rev3/manykey3.py"""
import os, sys, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rev2'))
from r2common import cfg_rev2, wilson, ppath, dump_json
from rev1.stats import pq, pooled_lps

WMS = ['gm_kgw', 'exact_kgw_bin', 'exact_bin_2r', 'a_k0', 'gm_kgw_sn']
COMPS = [('gm_kgw', 'exact_bin_2r'), ('gm_kgw', 'exact_kgw_bin'), ('exact_bin_2r', 'exact_kgw_bin'), ('a_k0', 'exact_bin_2r'),
         ('gm_kgw_sn', 'exact_bin_2r'), ('gm_kgw', 'a_k0')]


def main():
    cfg = cfg_rev2()
    gen = 'gpt_porto_r9_s0'
    keys = cfg['releases']['rev2_manykey']['keys']
    out = {}
    for attack in ('none', 'mm'):
        per = {}
        for wm in WMS:
            tp, pm, det = {}, {}, []
            for k in keys:
                rid = f'{wm}__k{k}__s0'
                d = pq(gen, rid, attack)
                if d is not None:
                    tp[k] = float((d['lp'] <= -3).mean())
                lps = pooled_lps(gen, [rid], attack)
                if lps:
                    pm[k] = float(-np.median(lps)); det += [x <= -3 for x in lps]
            if not pm:
                continue
            per[wm] = dict(n_keys=len(pm), pooled_median_by_key={str(k): v for k, v in pm.items()},
                           pooled_mlp_mean_over_keys=float(np.mean(list(pm.values()))), pooled_n=len(det),
                           pooled_k=int(np.sum(det)), wilson_lo=wilson(int(np.sum(det)), len(det))[0],
                           tpr_mean=float(np.mean(list(tp.values()))) if tp else None, tpr_by_key={str(k): v for k, v in tp.items()})
        comps = {}
        for a_, b_ in COMPS:
            if a_ not in per or b_ not in per:
                continue
            ks = sorted(set(per[a_]['pooled_median_by_key']) & set(per[b_]['pooled_median_by_key']), key=int)
            e = np.array([per[a_]['pooled_median_by_key'][k] - per[b_]['pooled_median_by_key'][k] for k in ks])
            if len(e) < 2:
                continue
            w = stats.wilcoxon(e) if np.any(e != 0) else None
            h = stats.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / math.sqrt(len(e))
            rec = dict(n_keys=len(e), mean=float(e.mean()), ci=[float(e.mean() - h), float(e.mean() + h)],
                       wilcoxon_p=float(w.pvalue) if w else None, n_a_better=int((e > 0).sum()))
            kt = sorted(set(per[a_]['tpr_by_key']) & set(per[b_]['tpr_by_key']), key=int)
            if len(kt) >= 2:
                d = np.array([per[a_]['tpr_by_key'][k] - per[b_]['tpr_by_key'][k] for k in kt])
                h2 = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
                rec['tpr'] = dict(mean=float(d.mean()), ci=[float(d.mean() - h2), float(d.mean() + h2)])
            comps[f'{a_}-{b_}'] = rec
        out[attack] = dict(per=per, paired=comps)
        print(attack, {w: (round(p['pooled_mlp_mean_over_keys'], 2), p['pooled_k'], p['pooled_n'], p['n_keys']) for w, p in per.items()},
              {c: (round(r['mean'], 2), [round(x, 2) for x in r['ci']], r['wilcoxon_p'], r['n_a_better']) for c, r in comps.items()}, flush=True)
    dump_json(out, ppath('results', 'rev3', 'manykey3.json'))


if __name__ == '__main__':
    main()
