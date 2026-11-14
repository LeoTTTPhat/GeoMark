"""Stage 7: record-linkage baseline (reviewer: an owner who keeps a copy of its release can attribute laundered traces by
matching them to the released traces, without any watermark).

Score of a suspect trace: the 5 released traces with the largest Jaccard similarity of r9 cell sets are short-listed
(inverted index; 10,000 released traces for the small GPT, 3,300 for Qwen), and the score is the largest normalized
longest-common-subsequence (LCS / max length) between the suspect's collapsed cell sequence and a short-listed one.
Only suspects and null traces with at least 8 distinct cells are scored (short traces match anything); the kept share is
reported. Both the Jaccard-only score and the LCS score are reported. Suspects: the first 1,000 traces of the
release under each attack (map-matching outputs from results/attacked/, synthetic attacks re-applied with fixed seeds).
Attack-matched nulls: 3,000 real test traces under the same attack scored against the same release; the per-trace
threshold is the maximum null score (empirical FPR <= 1/3,000). Also reported at r8. Output: results/rev1/linkage.json.
Model-free. Usage: python src/rev1/linkage.py"""
import os, sys, json
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump, load_pickle, ppath
from geomark.common import seed_from, Timer
from geomark.hexgrid import tokenize, parents
from geomark import attacks
from geomark.data import real_traces

ATTACKS = ['none', 'noise30', 'noise100', 'down3', 'crop50', 'mm', 'noise30+mm', 'mmrs']


class Index:
    def __init__(self, sets):
        self.sets = sets
        self.inv = {}
        for j, s in enumerate(sets):
            for x in s:
                self.inv.setdefault(x, []).append(j)

    def best(self, q):
        if not q:
            return 0.0
        cnt = Counter(j for x in q for j in self.inv.get(x, ()))
        return max((k / (len(q) + len(self.sets[j]) - k) for j, k in cnt.items()), default=0.0)

    def top(self, q, n=5):
        cnt = Counter(j for x in q for j in self.inv.get(x, ()))
        sc = sorted(((k / (len(q) + len(self.sets[j]) - k), j) for j, k in cnt.items()), reverse=True)[:n]
        return sc


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def seq_of(tr, res):
    if tr is None or len(tr['lat']) == 0:
        return []
    c = tokenize(tr['lat'], tr['lon'], 9)
    c = c if res == 9 else parents(c, res)
    out = []
    for x in c:
        if not out or out[-1] != int(x):
            out.append(int(x))
    return out


def score(idx, seqs, x, res):
    q = seq_of(x, res)
    if len(set(q)) < 8:
        return None, None
    top = idx.top(set(q))
    if not top:
        return 0.0, 0.0
    jac = top[0][0]
    best = max(lcs(q, seqs[j]) / max(len(q), len(seqs[j])) for _, j in top)
    return jac, best


def cellset(tr, res):
    if tr is None or len(tr['lat']) == 0:
        return set()
    c = tokenize(tr['lat'], tr['lon'], 9)
    return set(int(x) for x in (c if res == 9 else parents(c, res)))


def attacked(trs, attack, tag, cached):
    out = []
    for i, tr in enumerate(trs):
        if attacks.cache_prefix(attack):
            x = cached[i] if i < len(cached) else None
            out.append(x)
        else:
            out.append(attacks.apply(tr, attack, np.random.default_rng(seed_from(tag, attack, i))))
    return out


def main():
    cfg = cfg_rev1()
    out = {}
    T = Timer()
    for gen, rid in (('gpt_porto_r9_s0', 'gm_kgw__k0__s0'), ('gpt_geolife_r9_s0', 'gm_kgw__k0__s0'), ('qwen15_porto_r9', 'gm_kgw__k0__s0')):
        ds = cfg['generators'][gen]['dataset']
        R = load_release(cfg, gen, rid)['traces']
        real = real_traces(cfg, ds, 'test', limit=3000, seed=1, min_pts=2)
        for res in (9,):
            seqs = []
            for tr in R:
                c = tr['cells'] if res == 9 else parents(tr['cells'], res)
                o = []
                for x in c:
                    if not o or o[-1] != int(x):
                        o.append(int(x))
                seqs.append(o)
            idx = Index([set(q) for q in seqs])
            for attack in ATTACKS:
                pre = attacks.cache_prefix(attack)
                cs = cn = None
                if pre:
                    f1 = ppath('results', 'attacked', gen, f'{rid}__{pre}.pkl')
                    f2 = ppath('results', 'attacked', f'real_{ds}', f'test__{pre}.pkl')
                    if not (os.path.exists(f1) and os.path.exists(f2)):
                        continue
                    cs, cn = load_pickle(f1)['traces'], load_pickle(f2)['traces']
                sus = attacked(R[:1000], attack, f'{gen}/{rid}', cs)
                nul = attacked(real, attack, f'real_{ds}', cn)
                ss = [score(idx, seqs, x, res) for x in sus]
                sn = [score(idx, seqs, x, res) for x in nul]
                kept_s = [v for v in ss if v[0] is not None]; kept_n = [v for v in sn if v[0] is not None]
                rec = dict(n_sus=len(sus), n_null=len(nul), kept_sus=len(kept_s) / max(1, len(sus)), kept_null=len(kept_n) / max(1, len(nul)))
                for name, j in (('jaccard', 0), ('lcs', 1)):
                    a_ = np.array([v[j] for v in kept_s]); b_ = np.array([v[j] for v in kept_n])
                    if len(a_) and len(b_):
                        thr = float(b_.max()); thr99 = float(np.quantile(b_, 0.99))
                        rec[name] = dict(tpr_at_max_null=float((a_ > thr).mean()), tpr_at_1pct=float((a_ > thr99).mean()),
                                         thr=thr, thr99=thr99, sus_median=float(np.median(a_)), null_median=float(np.median(b_)))
                # dataset-level linkage: one-sided Mann-Whitney of 100-suspect pools against all null scores (LCS score)
                from scipy import stats as _st
                a_ = np.array([v[1] for v in kept_s]); b_ = np.array([v[1] for v in kept_n])
                if len(a_) >= 100 and len(b_) >= 200:
                    lps = [float(np.log10(max(_st.mannwhitneyu(a_[j:j + 100], b_, alternative='greater').pvalue, 1e-300)))
                           for j in range(0, len(a_) - 99, 100)]
                    rng_ = np.random.default_rng(0); perm = rng_.permutation(len(b_)); h = len(b_) // 2
                    nl = [float(np.log10(_st.mannwhitneyu(b_[perm[:h]][j:j + 100], b_[perm[h:]], alternative='greater').pvalue))
                          for j in range(0, h - 99, 100)]
                    rec['pooled'] = dict(n_pools=len(lps), median_mlp=float(-np.median(lps)), frac_le_m3=float(np.mean(np.array(lps) <= -3)),
                                         null_n=len(nl), null_frac_le_m3=float(np.mean(np.array(nl) <= -3)) if nl else None)
                out[f'{gen}|r{res}|{attack}'] = rec
                print(f"{gen} r{res} {attack}: kept {rec['kept_sus']:.2f}/{rec['kept_null']:.2f}; LCS TPR {rec.get('lcs', {}).get('tpr_at_max_null')} "
                      f"@1% {rec.get('lcs', {}).get('tpr_at_1pct')} (sus {rec.get('lcs', {}).get('sus_median')}, null max {rec.get('lcs', {}).get('thr')}) ({T():.0f}s)", flush=True)
    jdump(out, 'linkage.json')


if __name__ == '__main__':
    main()
