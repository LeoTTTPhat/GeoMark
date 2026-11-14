"""Stage 8: error distribution of approximation (6) (reviewers: '81% of 97 cells' was a one-sided criterion, measured >=
predicted, computed with the geometric coarse crossing rate before the switch to parent-defined labels).

For every one of the 97 noise cells of results/stats_tests.json lemma3.rows, the Stage-5 prediction used the geometric
coarse crossing rate q_{r-2}; we replace it by the parent-defined rate measured in results/rev1/lemma3.json (same
resolution and sigma): pred' = pred * ((1 - q_par) / (1 - q_geo))^(k+1), with k the scheme's history length (exact-hash
cells are left unchanged). Reported: mean / 90th percentile / max absolute error, share within 0.05 and 0.10, share with
measured >= predicted, by noise level and overall, for both versions.
Output: results/rev2/eq6.json. Model-free. Usage: python src/rev2/eq6.py"""
import os, sys, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import jdump2, ppath


def kof(wm):
    if wm.startswith('exact'):
        return None
    m = re.search(r'k(\d)', wm)
    return int(m.group(1)) if m else 1


def summ(err, meas, pred):
    a = np.abs(err)
    return dict(n=len(a), mean_abs=float(a.mean()), p90_abs=float(np.percentile(a, 90)), max_abs=float(a.max()),
                within_005=float((a <= 0.05).mean()), within_010=float((a <= 0.10).mean()), meas_ge_pred=float((meas >= pred).mean()))


def main():
    rows = json.load(open(ppath('results', 'stats_tests.json')))['lemma3']['rows']
    mc = json.load(open(ppath('results', 'rev1', 'lemma3.json')))['mc']
    q = {(r['res'], r['value_m']): r for r in mc if r['kind'] == 'gauss_sigma'}
    out = {}
    P0, P1, M, S = [], [], [], []
    for r in rows:
        res = int(re.search(r'_r(\d+)', r['gen']).group(1))
        sig = int(r['sigma'])
        k = kof(r['wm'])
        pred = r['pred_f']
        qq = q.get((res, sig))
        if k is not None and qq:
            pred2 = pred * ((1 - qq['q_coarse_parent']) / (1 - qq['q_coarse_geometric'])) ** (k + 1)
        else:
            pred2 = pred
        P0.append(pred); P1.append(pred2); M.append(r['meas_f']); S.append(sig)
    P0, P1, M, S = map(np.array, (P0, P1, M, S))
    for name, P in (('geometric', P0), ('parent', P1)):
        out[name] = dict(all=summ(M - P, M, P), **{f'sigma{s}': summ(M[S == s] - P[S == s], M[S == s], P[S == s]) for s in (10, 30, 100)})
    out['n_cells'] = int(len(M))
    jdump2(out, 'eq6.json')
    print(json.dumps(out, indent=1)[:1500])


if __name__ == '__main__':
    main()
