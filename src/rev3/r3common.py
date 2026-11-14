"""Stage 10 (round-3 audit) helpers: pooled test with BOTH tails and the reference-calibrated (key-conditional) test.

* Parametric (exact over the key, Lemma 1): per level, upper tail P[Bin(n, gamma) >= T] and lower tail P[Bin(n, gamma) <= T];
  the two levels (fine, medium) are Fisher-combined per tail; the two-sided p-value is min(1, 2 min(p_up, p_low))
  (Bonferroni over the two tails; each tail is super-uniform under H0, so the two-sided test stays exact-conservative).
* Reference-calibrated (key-conditional, approximate): per level t = (z - mu_K)/(sd_K / sqrt(fpc) * sqrt(1 + 1/B)) with
  mu_K, sd_K the mean / sd of the pooled z-score over B random m-trace reference pools of the owner's own real training data
  (second half of the training split, as in src/rev2/keycond2.py) under the owner's key, a t_{B-1} tail per side, Fisher over
  the levels, and the same Bonferroni factor 2 for two sides.
Model-free."""
import os, sys, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (os.path.join(os.path.dirname(HERE), 'rev2'), os.path.dirname(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from r2common import cfg_rev2, load_release, load_pickle, save_pickle, ppath, dump_json, LN10, wilson, real_cells  # noqa: F401
from keycond2 import Enc, greens  # noqa: F401

OUT3 = ppath('results', 'rev3')
os.makedirs(OUT3, exist_ok=True)
L2 = math.log10(2.0)
BREF = {100: 200, 1000: 200, 5000: 100, 10000: 100}


def cfg_rev3():
    from geomark.common import load_cfg
    return load_cfg(ppath('src', 'configs', 'rev3.yaml'))


def fisher(a, b, na, nb):
    """Fisher combination of two log10 p-values; falls back to the level that has pairs."""
    if na > 0 and nb > 0:
        return float(stats.chi2.logsf(-2.0 * LN10 * (a + b), 4) / LN10)
    return float(a if na > 0 else b)


def two(up, low):
    return min(0.0, min(up, low) + L2)


class PairCodec:
    """Encodes (seed, cell) pairs of a scheme to integer ids (fine and medium levels) for vectorised scoring."""

    def __init__(self, sch):
        self.sch, self.ef, self.em = sch, Enc(), Enc()

    def enc(self, fm):
        f, m = fm
        return self.ef.ids(f), (self.em.ids(m) if self.sch.uses_mid else np.zeros(0, np.int64))

    def greens(self, key):
        g = self.sch.gamma
        Gf = greens(self.ef, key, 'fine', g)
        Gm = greens(self.em, key, 'mid', g) if self.sch.uses_mid and len(self.em.cell) else np.zeros(0, np.float32)
        return Gf, Gm


def union(items):
    """items: list of (fine ids, mid ids) -> unique ids per level."""
    if not items:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.unique(np.concatenate([x[0] for x in items])), np.unique(np.concatenate([x[1] for x in items]))


def ref_pools(ref, m, B, rng):
    """B random m-subsets of the encoded reference traces -> list of (fine ids, mid ids)."""
    out = []
    for _ in range(B):
        g = rng.choice(len(ref), size=m, replace=False)
        out.append(union([ref[j] for j in g]))
    return out


def counts(pool, Gf, Gm):
    F, M = pool
    return float(Gf[F].sum()) if len(F) else 0.0, len(F), (float(Gm[M].sum()) if len(M) and len(Gm) else 0.0), (len(M) if len(Gm) else 0)


def zval(T, n, gamma):
    return (T - n * gamma) / math.sqrt(n * gamma * (1 - gamma)) if n > 0 else 0.0


def ref_stats(pools, Gf, Gm, gamma, B, n_ref, m):
    zf, zm = [], []
    for p in pools:
        Tf, nf, Tm, nm = counts(p, Gf, Gm)
        zf.append(zval(Tf, nf, gamma)); zm.append(zval(Tm, nm, gamma))
    zf, zm = np.array(zf), np.array(zm)
    return dict(muf=float(zf.mean()), sdf=float(zf.std(ddof=1)), mum=float(zm.mean()), sdm=float(zm.std(ddof=1)),
                B=B, fpc=1 - m / n_ref)


def _tt(z, mu, sd, B, fpc):
    sd = max(sd, 1e-3)
    t = (z - mu) / (sd / math.sqrt(fpc) * math.sqrt(1 + 1 / B))
    return float(stats.t.logsf(t, B - 1) / LN10), float(stats.t.logcdf(t, B - 1) / LN10)


def score(pool, Gf, Gm, gamma, rs=None):
    """All p-values (log10) of one pooled dataset. rs: reference statistics of the key at this pool size (or None)."""
    Tf, nf, Tm, nm = counts(pool, Gf, Gm)
    uf = float(stats.binom.logsf(Tf - 1, nf, gamma) / LN10) if nf else 0.0
    lf = float(stats.binom.logcdf(Tf, nf, gamma) / LN10) if nf else 0.0
    um = float(stats.binom.logsf(Tm - 1, nm, gamma) / LN10) if nm else 0.0
    lm = float(stats.binom.logcdf(Tm, nm, gamma) / LN10) if nm else 0.0
    zf, zm = zval(Tf, nf, gamma), zval(Tm, nm, gamma)
    up, low = fisher(uf, um, nf, nm), fisher(lf, lm, nf, nm)
    r = dict(nf=nf, nm=nm, zf=zf, zm=zm, up=up, low=low, two=two(up, low))
    if rs is not None:
        cuf, clf = _tt(zf, rs['muf'], rs['sdf'], rs['B'], rs['fpc'])
        if nm:
            cum, clm = _tt(zm, rs['mum'], rs['sdm'], rs['B'], rs['fpc'])
        else:
            cum = clm = 0.0
        cu, cl = fisher(cuf, cum, nf, nm), fisher(clf, clm, nf, nm)
        r.update(cal_up=cu, cal_low=cl, cal_two=two(cu, cl))
    return r


def summ(recs, field):
    a = np.array([r[field] for r in recs if field in r])
    if len(a) == 0:
        return None
    k = int((a <= -3).sum())
    return dict(n=len(a), median_mlp=float(-np.median(a)), k_le_m3=k, frac_le_m3=k / len(a),
                wilson=list(wilson(k, len(a))))
