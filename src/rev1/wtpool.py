"""Stage 7: dataset-level W-Trace (reviewers: W-Trace was compared per trace against GeoMark's pooled test).

For every W-Trace release (3 generators x 3 keys, 1,000 marked traces per attack) and attack, disjoint 100-trace pools of
marked traces are tested against the 1,000 attacked unmarked originals of the same release (ncc_null) with a one-sided
Mann-Whitney U test (H1: the pool's NCCs are stochastically larger). Under H0 (unmarked pool) the pool and the null
sample are exchangeable, so the test is valid. A null control applies the same test to 100-trace pools drawn from the
null NCCs themselves (split-half). Also records the per-trace threshold actually used in the paper (the maximum of the
1,000 null NCCs, i.e. an empirical FPR of about 1e-3 on attacked unwatermarked synthetic traces, not real traces).
Output: results/rev1/wtpool.json. Usage: python src/rev1/wtpool.py"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import OUT, jdump, ppath


def main():
    out = {}
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(ppath('results', 'baselines', 'wtrace_*.parquet'))):
        name = os.path.basename(f)[len('wtrace_'):-len('.parquet')]
        gen, rid = name.split('_wtrace__')
        d = pd.read_parquet(f)
        for attack, g in d.groupby('attack'):
            x = g['ncc'].to_numpy(float); y = g['ncc_null'].to_numpy(float)
            lps = []
            for j in range(len(x) // 100):
                u = stats.mannwhitneyu(x[j * 100:(j + 1) * 100], y, alternative='greater')
                lps.append(float(np.log10(max(u.pvalue, 1e-300))))
            # null control: split the null sample, pools of 100 vs the remaining null values
            perm = rng.permutation(len(y)); a_, b_ = y[perm[:500]], y[perm[500:]]
            nl = [float(np.log10(stats.mannwhitneyu(a_[j * 100:(j + 1) * 100], b_, alternative='greater').pvalue)) for j in range(5)]
            thr = float(np.max(y))
            out.setdefault(gen, {}).setdefault(attack, []).append(dict(rid='wtrace__' + rid, lps=lps, null_lps=nl,
                                                                        tpr_trace_at_max_null=float((x > thr).mean())))
    summ = {}
    for gen, A in out.items():
        for attack, L in A.items():
            lps = [v for r in L for v in r['lps']]
            summ[f'{gen}|{attack}'] = dict(n_pools=len(lps), median_mlp=float(-np.median(lps)), frac_le_m3=float(np.mean(np.array(lps) <= -3)),
                                           null_frac_le_m2=float(np.mean(np.array([v for r in L for v in r['null_lps']]) <= -2)),
                                           tpr_trace=float(np.mean([r['tpr_trace_at_max_null'] for r in L])))
            print(gen, attack, summ[f'{gen}|{attack}'], flush=True)
    jdump(dict(per_release=out, summary=summ), 'wtpool.json')


if __name__ == '__main__':
    main()
