"""Utility metrics of a synthetic release vs real traces (E7): JSD of trip length, radius of gyration, OD matrix at the
coarse resolution, cell popularity, transition (bigram) distribution; diversity; memorisation."""
import numpy as np
from collections import Counter
from .common import LocalProj
from .hexgrid import parents


def jsd_counts(a, b):
    keys = list(set(a) | set(b))
    p = np.array([a.get(k, 0) for k in keys], dtype=float); q = np.array([b.get(k, 0) for k in keys], dtype=float)
    return jsd(p, q)


def jsd(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_hist(x, y, bins):
    return jsd(np.histogram(x, bins)[0].astype(float), np.histogram(y, bins)[0].astype(float))


def trip_stats(tr):
    pr = LocalProj.for_dataset(tr['ds'])
    x, y = pr.to_xy(tr['lat'], tr['lon'])
    length = float(np.hypot(np.diff(x), np.diff(y)).sum()) / 1000 if len(x) > 1 else 0.0
    rg = float(np.sqrt(((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean())) / 1000 if len(x) else 0.0
    return length, rg


def cell_length_km(cells, ds):
    import h3.api.basic_int as h3i
    if len(cells) < 2:
        return 0.0, 0.0
    ll = np.array([h3i.cell_to_latlng(int(c)) for c in cells])
    pr = LocalProj.for_dataset(ds)
    x, y = pr.to_xy(ll[:, 0], ll[:, 1])
    return float(np.hypot(np.diff(x), np.diff(y)).sum()) / 1000, float(np.sqrt(((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean())) / 1000


def utility(gen_cells, real_cells, ds, res, train_set=None, coarse_off=2):
    """gen_cells / real_cells: lists of collapsed fine-cell arrays. Lengths & gyration on cell centroids (decode-free)."""
    gl = np.array([cell_length_km(c, ds) for c in gen_cells]); rl = np.array([cell_length_km(c, ds) for c in real_cells])
    out = {}
    out['jsd_length'] = jsd_hist(gl[:, 0], rl[:, 0], np.linspace(0, max(1e-3, np.percentile(rl[:, 0], 99)), 41))
    out['jsd_gyration'] = jsd_hist(gl[:, 1], rl[:, 1], np.linspace(0, max(1e-3, np.percentile(rl[:, 1], 99)), 41))
    cr = res - coarse_off
    od_g = Counter((int(parents(c[:1], cr)[0]), int(parents(c[-1:], cr)[0])) for c in gen_cells if len(c))
    od_r = Counter((int(parents(c[:1], cr)[0]), int(parents(c[-1:], cr)[0])) for c in real_cells if len(c))
    out['jsd_od'] = jsd_counts(od_g, od_r)
    pop_g = Counter(int(x) for c in gen_cells for x in c); pop_r = Counter(int(x) for c in real_cells for x in c)
    out['jsd_cellpop'] = jsd_counts(pop_g, pop_r)
    bg_g = Counter((int(a), int(b)) for c in gen_cells for a, b in zip(c[:-1], c[1:]))
    bg_r = Counter((int(a), int(b)) for c in real_cells for a, b in zip(c[:-1], c[1:]))
    out['jsd_bigram'] = jsd_counts(bg_g, bg_r)
    out['mean_len_tok_gen'] = float(np.mean([len(c) for c in gen_cells]))
    out['mean_len_tok_real'] = float(np.mean([len(c) for c in real_cells]))
    out['mean_length_km_gen'] = float(gl[:, 0].mean()); out['mean_length_km_real'] = float(rl[:, 0].mean())
    # diversity: distinct trigrams / total trigrams, duplicate traces
    tri = [tuple(int(x) for x in c[i:i + 3]) for c in gen_cells for i in range(len(c) - 2)]
    out['distinct_trigram_ratio'] = len(set(tri)) / max(1, len(tri))
    seqs = [tuple(int(x) for x in c) for c in gen_cells]
    out['duplicate_trace_frac'] = 1 - len(set(seqs)) / max(1, len(seqs))
    if train_set is not None:
        out['memorised_exact_frac'] = float(np.mean([s in train_set for s in seqs if len(s) >= 5])) if seqs else 0.0
    return out


def nearest_jaccard(gen_cells, train_cells, n=500, seed=0):
    """Mean max-Jaccard (cell sets) of n generated traces to the training set (inverted index)."""
    rng = np.random.default_rng(seed)
    inv = {}
    tsets = [set(int(x) for x in c) for c in train_cells]
    for j, s in enumerate(tsets):
        for x in s:
            inv.setdefault(x, []).append(j)
    vals = []
    for i in rng.choice(len(gen_cells), size=min(n, len(gen_cells)), replace=False):
        g = set(int(x) for x in gen_cells[i])
        if not g:
            continue
        cnt = Counter(j for x in g for j in inv.get(x, ()))
        best = max((k / (len(g) + len(tsets[j]) - k) for j, k in cnt.items()), default=0.0)
        vals.append(best)
    return float(np.mean(vals)) if vals else 0.0
