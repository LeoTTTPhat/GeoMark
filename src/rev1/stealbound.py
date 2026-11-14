"""Stage 7: instantiate Proposition 6 (stealing) in terms of the seed space, N, delta, gamma and local entropy
(reviewers: the proposition is a generic two-point bound with no dependence on the seed space or on N).

Model. Under seed s the unwatermarked next-cell probability of cell c is p = p(c|s); KGW multiplies green cells by
e^delta and renormalises with Z_s = 1 + (e^delta - 1) g_s (g_s = green mass, ~gamma). A visit of seed s is one Bernoulli
observation of "c follows": mean p e^delta / Z (green) or p / Z (red). The stealer thresholds at the prior frequency p
(as run_steal.Estimate does), so by Hoeffding the error is <= exp(-2 n_s m^2) with margin
m = p * min(1 - 1/Z, e^delta/Z - 1). Hence a pair is learnable with error <= eta once the seed has been visited
n_s >= ln(1/eta) / (2 m^2) times; after N traces the expected number of visits is N * v_s (v_s = visits per trace).
Computed from 80,000 real training traces (visit rates and p(c|s)), weighted by pair frequency, for k = 0, 1, 2 with
30-min bins, k = 1 with 120-min bins, and 4-key rotation (visits split over 4 keys), at N = 10^3..10^6. Compared with the
measured green precision / recall of the stealing experiment (results/summary.json steal).
Output: results/rev1/stealbound.json. Model-free. Usage: python src/rev1/stealbound.py"""
import os, sys, math, json
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, jdump, real_cells, ppath
from geomark.scheme import Scheme, seeds_for_cells


def main():
    cfg = cfg_rev1()
    cells, slots = real_cells(cfg, 'porto', 9, 'train', n=80000, seed=7)
    delta, gamma, eta = 2.0, 0.25, 0.1
    Z = 1 + (math.exp(delta) - 1) * gamma
    mfac = min(1 - 1 / Z, math.exp(delta) / Z - 1)
    out = dict(delta=delta, gamma=gamma, eta=eta, Z=Z, margin_factor=mfac, configs={})
    Ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    for name, k, binm, rot in (('k0_b30', 0, 30, 1), ('k1_b30', 1, 30, 1), ('k2_b30', 2, 30, 1), ('k1_b120', 1, 120, 1),
                               ('k1_b30_rot4', 1, 30, 4)):
        sch = Scheme(method='kgw', res=9, k=k, bin_min=binm)
        visits = Counter(); pc = Counter(); seen_s = Counter()
        for c, sl in zip(cells, slots):
            seeds, _ = seeds_for_cells(c, sl, sch)
            for s in set(seeds):
                seen_s[s] += 1
            for s, x in zip(seeds, c):
                visits[s] += 1; pc[(s, int(x))] += 1
        n_tr = len(cells)
        pairs = list(pc)
        w = np.array([pc[q] for q in pairs], float)                       # pair frequency (weight)
        p = np.array([pc[q] / visits[q[0]] for q in pairs])                # p(c | s)
        v = np.array([visits[q[0]] / n_tr for q in pairs])                 # visits of the seed per trace
        need = math.log(1 / eta) / (2 * (mfac * p) ** 2)                   # visits needed for error <= eta
        rec = dict(n_seeds=len(visits), n_pairs=len(pairs), median_visits_needed=float(np.median(need)),
                   learnable_frac={})
        for N in Ns:
            rec['learnable_frac'][str(N)] = float((w * (N * v / rot >= need)).sum() / w.sum())
        out['configs'][name] = rec
        print(name, rec['n_seeds'], {k_: round(x, 3) for k_, x in rec['learnable_frac'].items()}, flush=True)
    S = json.load(open(ppath('results', 'summary.json')))['steal_spoof5000']
    out['measured'] = {k_: dict(precision=v_['accuracy'].get('precision'), recall=v_['accuracy'].get('recall'),
                                n_seeds=v_['accuracy'].get('n_seeds')) for k_, v_ in S.items() if 'accuracy' in v_}
    jdump(out, 'stealbound.json')


if __name__ == '__main__':
    main()
