"""GeoMark detector: re-tokenise, recompute seeds, de-duplicate (seed, cell) pairs, exact tail p-values,
two-resolution Fisher combination, dataset-level pooling and time-bin-shift search (Bonferroni).

All p-values are returned as log10 values (they can be far below 1e-300 for pooled datasets).
Exactness (Lemma 1): the pair sets are functions of the trace only; distinct pairs get independent PRF uniforms
under H0 (trace independent of the key), so T is exactly Binomial(n, gamma) (KGW) or Gamma(n, 1) (Gumbel).
"""
import math
import numpy as np
from scipy import stats
from .scheme import seeds_for_cells
from .hexgrid import parents, tokenize, is_neighbor
from .common import slot_of
from .prf import PRF

LN10 = math.log(10.0)


def pair_sets(cells, slot, sch, dedup=True, mirror_mask=True):
    """Return (fine pairs, mid pairs) as lists of (seed, cell); de-duplicated unless dedup=False."""
    cells = np.asarray(cells, dtype=np.int64)
    if len(cells) == 0:
        return [], []
    seeds, masked = seeds_for_cells(cells, slot, sch)
    keep = [i for i in range(len(cells)) if not (mirror_mask and masked[i])]
    if getattr(sch, 'adj_neutral', False):   # Stage 8: only adjacent steps (and the first cell) carry the watermark
        keep = [i for i in keep if i == 0 or is_neighbor(cells[i - 1], cells[i])]
    fine = [(seeds[i], int(cells[i])) for i in keep]
    mid = []
    if sch.uses_mid:
        par = parents(cells, sch.mid_res)
        mid = [(seeds[i], int(par[i])) for i in keep]
    if dedup:
        fine = list(dict.fromkeys(fine))
        mid = list(dict.fromkeys(mid))
    return fine, mid


def uniforms(prf, pairs, domain):
    if not pairs:
        return np.zeros(0)
    s, c = zip(*pairs)
    return prf.pairs(domain, s, c)


def log10p_kgw(u, gamma, rng=None):
    n = len(u)
    if n == 0:
        return 0.0
    g = int((u < gamma).sum())
    if rng is None:   # conservative (super-uniform) exact binomial tail P[Bin >= g]
        return float(stats.binom.logsf(g - 1, n, gamma)) / LN10
    # randomised p-value: exactly Uniform(0,1) under H0
    pv = stats.binom.sf(g, n, gamma) + rng.random() * stats.binom.pmf(g, n, gamma)
    return math.log10(max(pv, 1e-320))


def log10p_gumbel(u):
    n = len(u)
    if n == 0:
        return 0.0
    T = float(-np.log1p(-u).sum())
    return float(stats.gamma.logsf(T, n)) / LN10


def log10p_scores(u, sch, rng=None):
    return log10p_gumbel(u) if sch.score == 'gumbel' else log10p_kgw(u, sch.gamma, rng)


def fisher(lps):
    """Fisher combination of independent log10 p-values (ignores levels with no evidence, i.e. lp == 0 & n == 0)."""
    lps = [lp for lp in lps if lp is not None]
    if not lps:
        return 0.0
    if len(lps) == 1:
        return lps[0]
    X = -2.0 * LN10 * sum(lps)
    return float(stats.chi2.logsf(X, 2 * len(lps))) / LN10


def score_pairs(fine, mid, sch, prf, rng=None):
    uf = uniforms(prf, fine, 'fine')
    lp_f = log10p_scores(uf, sch, rng)
    out = {'n_f': len(fine), 'lp_f': lp_f, 'stat_f': _stat(uf, sch)}
    if sch.uses_mid:
        um = uniforms(prf, mid, 'mid')
        lp_m = log10p_scores(um, sch, rng)
        out.update(n_m=len(mid), lp_m=lp_m, stat_m=_stat(um, sch))
        out['lp'] = fisher([lp_f if len(fine) else None, lp_m if len(mid) else None])
    else:
        out.update(n_m=0, lp_m=0.0, stat_m=0.0, lp=lp_f)
    return out


def _stat(u, sch):
    if len(u) == 0:
        return 0.0
    return float((u < sch.gamma).mean()) if sch.score == 'kgw' else float(-np.log1p(-u).mean())


def detect_cells(cells, slot, sch, prf, J=0, rng=None, dedup=True):
    """Detect on a collapsed fine-cell sequence with start slot. J > 0: also search slot shifts of the time bin
    (+-J bins) and report the Bonferroni-corrected minimum ('lp_shift')."""
    fine, mid = pair_sets(cells, slot, sch, dedup=dedup)
    out = score_pairs(fine, mid, sch, prf, rng)
    if J > 0:
        if sch.bin_min and sch.method != 'exact_kgw':   # exact_kgw_bin has a bin and is searched
            best = out['lp']
            step = sch.bin_min // 30
            for j in range(-J, J + 1):
                if j == 0:
                    continue
                f2, m2 = pair_sets(cells, (slot + j * step) % 48, sch, dedup=dedup)
                best = min(best, score_pairs(f2, m2, sch, prf, rng)['lp'])
            out['lp_shift'] = min(0.0, best + math.log10(2 * J + 1))
        else:
            out['lp_shift'] = out['lp']
    return out


def trace_cells(tr, sch):
    """Fine cells (collapsed) and start slot of a point trace dict {lat, lon, t}; ds needed for the local time."""
    cells = tokenize(tr['lat'], tr['lon'], sch.res)
    slot = slot_of(tr['t'][0], tr['ds']) if len(tr['t']) else 0
    return cells, slot


def detect_trace(tr, sch, prf, J=0, rng=None):
    cells, slot = trace_cells(tr, sch)
    out = detect_cells(cells, slot, sch, prf, J=J, rng=rng)
    out['len_pts'] = int(len(tr['lat']))
    out['len_tok'] = int(len(cells))
    return out


def pooled(pair_lists, sch, prf, rng=None):
    """Dataset-level test: union of distinct pairs over traces (still exact by Lemma 1)."""
    F, M = {}, {}
    for fine, mid in pair_lists:
        F.update(dict.fromkeys(fine)); M.update(dict.fromkeys(mid))
    return score_pairs(list(F), list(M), sch, prf, rng)


def survival(orig_pairs, new_pairs):
    if not orig_pairs:
        return float('nan')
    s = set(new_pairs)
    return sum(1 for x in orig_pairs if x in s) / len(orig_pairs)
