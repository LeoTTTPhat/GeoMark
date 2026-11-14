"""Statistics helpers: TPR at FPR, ROC-AUC, bootstrap CIs, KS calibration, Wilcoxon + Holm."""
import numpy as np
from scipy import stats


def tpr_at(lp, alpha):
    """Fraction with p <= alpha (lp = log10 p)."""
    lp = np.asarray(lp, dtype=float)
    return float((lp <= np.log10(alpha)).mean()) if len(lp) else float('nan')


def auc(pos_scores, neg_scores):
    """ROC-AUC where higher score = more watermark-like (use -lp)."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(x, fn=np.mean, B=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float('nan')] * 3
    if len(x) == 1:
        return [float(x[0])] * 3
    rng = np.random.default_rng(seed)
    bs = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(fn(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))]


def mean_ci_t(x):
    """Mean and 95% t-interval over replicates (keys/seeds)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float('nan')] * 3
    m = float(x.mean())
    if len(x) < 2:                     # Stage 4b: no interval from a single replicate (was [m, m, m])
        return [m, float('nan'), float('nan')]
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return [m, m - h, m + h]


def ks_uniform(pv):
    pv = np.asarray(pv, dtype=float)
    r = stats.kstest(pv, 'uniform')
    return dict(ks_stat=float(r.statistic), ks_p=float(r.pvalue), n=int(len(pv)))


def empirical_fpr(lp, alphas=(1e-1, 1e-2, 1e-3, 1e-4)):
    lp = np.asarray(lp, dtype=float)
    out = {}
    for a in alphas:
        k = int((lp <= np.log10(a)).sum())
        lo, hi = stats.beta.ppf(0.025, k, len(lp) - k + 1) if k else 0.0, stats.beta.ppf(0.975, k + 1, len(lp) - k)
        out[f'{a:g}'] = dict(fpr=k / max(len(lp), 1), ci=[float(lo), float(hi)], k=k, n=int(len(lp)))
    return out


def wilcoxon_holm(pairs):
    """pairs: dict name -> (x, y) paired arrays. Returns name -> dict(stat, p, p_holm)."""
    res = {}
    for k, (x, y) in pairs.items():
        x, y = np.asarray(x, float), np.asarray(y, float)
        try:
            r = stats.wilcoxon(x, y)
            res[k] = dict(stat=float(r.statistic), p=float(r.pvalue))
        except ValueError:
            res[k] = dict(stat=float('nan'), p=1.0)
    order = sorted(res, key=lambda k: res[k]['p'])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * res[k]['p']))
        res[k]['p_holm'] = run
    return res
