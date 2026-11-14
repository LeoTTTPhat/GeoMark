"""Stage 7: a valid combined radioactivity test across students (reviewers: the Fisher combination of the students' rank
p-values is invalid, because all students are ranked against the SAME null keys and read on nearly the same pairs).

Joint rank test: for every key (the teacher key and each of the shared null keys) the statistic is the SUM over the
three students of that key's log10 p (reading or sampling mode). Under H0 (students independent of the teacher key) the
teacher key is exchangeable with the i.i.d. null keys, so the rank of the teacher's sum among all keys is a valid
p-value whatever the dependence between students. Both modes are reported and a Bonferroni factor 2 over the two modes
gives the pre-specified combined decision. Also reports the per-student parametric p-values and the fixed wrong-key
(key 1) control. Inputs: results/radio/keynull.jsonl (500 null keys), keynull_power.jsonl (200 keys, more contexts),
results.jsonl. Output: results/rev1/radio_joint.json. Usage: python src/rev1/radio_joint.py"""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import OUT, jdump, read_jsonl, ppath


def joint(recs, mode):
    nulls = np.array([r[mode]['null_lps'] for r in recs])          # (students, keys)
    true = np.array([r[mode]['lp_true'] for r in recs])
    s_null = nulls.sum(0); s_true = true.sum()
    rank_p = (1 + (s_null <= s_true).sum()) / (len(s_null) + 1)
    return dict(n_students=len(recs), n_null_keys=int(nulls.shape[1]), sum_lp_true=float(s_true),
                null_sum_median=float(np.median(s_null)), rank_p=float(rank_p),
                per_student_rank_p=[r[mode]['rank_p'] for r in recs], per_student_lp=[round(float(x), 3) for x in true])


def main():
    out = {}
    for fn, tag in (('keynull.jsonl', 'n500'), ('keynull_power.jsonl', 'power_n200')):
        recs = read_jsonl(ppath('results', 'radio', fn))
        groups = defaultdict(list)
        for r in recs:
            groups[(r['wm'], r['rho'])].append(r)
        for (wm, rho), rs in sorted(groups.items()):
            rs = sorted(rs, key=lambda r: r['seed'])
            keysets = {tuple(r['null_keys']) for r in rs}
            if len(keysets) != 1 or len(rs) < 2:
                continue
            rec = {m: joint(rs, m) for m in ('read', 'sample')}
            rec['combined_bonferroni2'] = float(min(1.0, 2 * min(rec['read']['rank_p'], rec['sample']['rank_p'])))
            out[f'{tag}|{wm}|rho{rho}'] = rec
            print(f"{tag} {wm} rho={rho}: joint rank p read {rec['read']['rank_p']:.4f} sample {rec['sample']['rank_p']:.4f} "
                  f"-> combined {rec['combined_bonferroni2']:.4f}", flush=True)
    # fixed wrong key control (key 1) from the original run
    wk = {}
    for r in read_jsonl(ppath('results', 'radio', 'results.jsonl')):
        wk[r['key']] = dict(rho=r['rho'], wm=r['wm'], read_k0=r['read_k0']['lp'], read_k1=r['read_k1']['lp'],
                            sample_k0=r['sample_k0']['lp'], sample_k1=r['sample_k1']['lp'])
    out['wrong_key_control'] = wk
    jdump(out, 'radio_joint.json')


if __name__ == '__main__':
    main()
