"""Stage 8: Proposition 7 (stealing) re-instantiated with MATCHING upper and lower bounds (reviewers: the Hoeffding
instantiation scales as 1/p^2 while the lower bound scales as 1/p, so its predictions were artifacts; it ignored the
medium-level coin shared by the ~7 fine siblings of a medium cell).

Model (KGW-2R; single resolution has delta_m = 0). Under seed s the unwatermarked next-cell probability is p = p(c|s).
With fine coin g_f and medium coin g_m (each green w.p. gamma) the observation "c follows s" has mean
mu = p * f, f = exp(delta g_f + delta_m g_m) / (Z_f Z_m), Z_f = 1 + (e^delta - 1) gamma, Z_m = 1 + (e^delta_m - 1) gamma
(green mass of a seed taken equal to its mean gamma; EOS handling ignored). The stealer thresholds the frequency at p.
  Upper bound (Chernoff, KL form): P[error] <= exp(-n KL(p || p f)) after n visits of s, so a pair of class f is learned
  with error <= eta once n >= ln(1/eta) / KL(p || p f)   [Theta(1/p)].
  Lower bound (Bretagnolle-Huber): any test between the two fine classes with the same medium coin errs w.p. at least
  (1/4) exp(-n KL(p f_R || p f_G)), so n >= ln(1/(4 eta)) / KL(p f_R || p f_G) is necessary   [Theta(1/p)].
  Medium coin: the pair (s, P) is observed whenever any child of P follows s: p_P = sum_c p(c|s), factor
  exp(delta_m g_m)/Z_m (fine coins average out), so the medium half of the evidence is learned from pooled sibling visits.
F_fine(N) / F_mid(N): share of fine / medium pair mass (weights = pair frequency in 80,000 real training traces) whose
expected number of visits N v_s reaches the Chernoff requirement (class-averaged); F_lower uses the necessary n.
Configurations: k = 0, 1, 2 (30-min bins), k = 1 with 120-min bins, exact previous cell + bin (single resolution), and
4-key rotation. Output: results/rev2/stealbound2.json. Model-free. Usage: python src/rev2/stealbound2.py"""
import os, sys, math
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, real_cells
from geomark.scheme import Scheme, seeds_for_cells
from geomark.hexgrid import parents


def kl(a, b):
    a = np.clip(a, 1e-15, 1 - 1e-15); b = np.clip(b, 1e-15, 1 - 1e-15)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def main():
    cfg = cfg_rev2()
    cells, slots = real_cells(cfg, 'porto', 9, 'train', n=80000, seed=7)
    gamma, eta = 0.25, 0.1
    Ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    out = dict(gamma=gamma, eta=eta, configs={})
    for name, kw, delta, dm, rot in (('k0_b30', dict(method='kgw', k=0, bin_min=30), 2.0, 1.0, 1),
                                     ('k1_b30', dict(method='kgw', k=1, bin_min=30), 2.0, 1.0, 1),
                                     ('k2_b30', dict(method='kgw', k=2, bin_min=30), 2.0, 1.0, 1),
                                     ('k1_b120', dict(method='kgw', k=1, bin_min=120), 2.0, 1.0, 1),
                                     ('exact_bin', dict(method='exact_kgw_bin', bin_min=30, two_res=False), 2.0, 0.0, 1),
                                     ('k1_b30_rot4', dict(method='kgw', k=1, bin_min=30), 2.0, 1.0, 4)):
        sch = Scheme(res=9, **kw)
        visits = Counter(); pc = Counter(); pm = Counter()
        for c, sl in zip(cells, slots):
            seeds, _ = seeds_for_cells(c, sl, sch)
            mids = parents(c, 8)
            for s, x, P in zip(seeds, c, mids):
                visits[s] += 1; pc[(s, int(x))] += 1; pm[(s, int(P))] += 1
        n_tr = len(cells)
        Zf = 1 + (math.exp(delta) - 1) * gamma
        Zm = 1 + (math.exp(dm) - 1) * gamma
        classes = [(gf, gm, (gamma if gf else 1 - gamma) * ((gamma if gm else 1 - gamma) if dm else (1.0 if gm == 0 else 0.0)))
                   for gf in (0, 1) for gm in (0, 1)]
        classes = [c_ for c_ in classes if c_[2] > 0]
        rec = dict(n_seeds=len(visits), n_pairs=len(pc), n_mid_pairs=len(pm), F_fine={}, F_fine_lower={}, F_mid={})
        for level, cnt in (('fine', pc), ('mid', pm)):
            pairs = list(cnt)
            w = np.array([cnt[q] for q in pairs], float)
            p = np.array([cnt[q] / visits[q[0]] for q in pairs])
            v = np.array([visits[q[0]] / n_tr for q in pairs]) / rot
            if level == 'fine':
                need_up = []; need_lo = []; prob = []
                for gf, gm, pr in classes:
                    f = math.exp(delta * gf + dm * gm) / (Zf * Zm)
                    need_up.append(math.log(1 / eta) / np.maximum(kl(p, p * f), 1e-300))
                    fR = math.exp(dm * gm) / (Zf * Zm); fG = fR * math.exp(delta)
                    need_lo.append(math.log(1 / (4 * eta)) / np.maximum(kl(p * fR, p * fG), 1e-300))
                    prob.append(pr)
                for N in Ns:
                    rec['F_fine'][str(N)] = float(sum(pr * (w * (N * v >= nu)).sum() for pr, nu in zip(prob, need_up)) / w.sum())
                    rec['F_fine_lower'][str(N)] = float(sum(pr * (w * (N * v >= nl)).sum() for pr, nl in zip(prob, need_lo)) / w.sum())
                rec['median_visits_needed_fine'] = float(np.median(need_up[0]))
                rec['median_visits_needed_fine_lower'] = float(np.median(need_lo[0]))
            elif dm:
                need = [math.log(1 / eta) / np.maximum(kl(p, p * math.exp(dm * gm) / Zm), 1e-300) for gm in (0, 1)]
                for N in Ns:
                    rec['F_mid'][str(N)] = float(sum(pr * (w * (N * v >= nu)).sum() for pr, nu in zip((1 - gamma, gamma), need)) / w.sum())
                rec['median_visits_needed_mid'] = float(np.median(need[0]))
        out['configs'][name] = rec
        print(name, rec['n_seeds'], {k: round(x, 3) for k, x in rec['F_fine'].items()}, {k: round(x, 3) for k, x in rec['F_mid'].items()}, flush=True)
    jdump2(out, 'stealbound2.json')


if __name__ == '__main__':
    main()
