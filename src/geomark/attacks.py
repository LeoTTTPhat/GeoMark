"""Launderer attacks on point traces {lat, lon, t, ds}. Names (composable with '+', applied left to right):
  none | noise<m> (isotropic Gaussian, per-axis sigma m metres) | down<k> (keep every k-th point)
  retime<min> (shift all timestamps by +-min minutes, random sign) | crop<pct> (keep a random contiguous (100-pct)%)
  splice<pct> (append a real trace segment so that pct% of points are real) | mm, mmrs (map-matching; precomputed
  cache, see mapmatch.py / src/mapmatch_run.py) | regen (precomputed by src/generate.py --regen)
Dataset-level mixing (mix<pct>) is handled in evaluate.py (pooled tests)."""
import re
import numpy as np
from .common import LocalProj

CACHED = ('mm', 'mmrs', 'regen')


def _copy(tr, idx=None):
    if idx is None:
        return dict(lat=np.array(tr['lat'], dtype=float), lon=np.array(tr['lon'], dtype=float), t=np.array(tr['t']), ds=tr['ds'])
    return dict(lat=np.asarray(tr['lat'])[idx], lon=np.asarray(tr['lon'])[idx], t=np.asarray(tr['t'])[idx], ds=tr['ds'])


def noise(tr, sigma, rng):
    pr = LocalProj.for_dataset(tr['ds'])
    x, y = pr.to_xy(tr['lat'], tr['lon'])
    x = x + rng.normal(0, sigma, len(x)); y = y + rng.normal(0, sigma, len(y))
    la, lo = pr.to_ll(x, y)
    o = _copy(tr); o['lat'], o['lon'] = la, lo
    return o


def down(tr, k, rng):
    n = len(tr['lat'])
    off = int(rng.integers(0, k)) if n > k else 0
    idx = np.arange(off, n, k)
    if len(idx) == 0:
        idx = np.array([0])
    return _copy(tr, idx)


def retime(tr, minutes, rng):
    o = _copy(tr)
    o['t'] = o['t'] + int(rng.choice([-1, 1]) * minutes * 60)
    return o


def crop(tr, pct, rng):
    n = len(tr['lat'])
    keep = max(1, int(round(n * (100 - pct) / 100)))
    a = int(rng.integers(0, n - keep + 1))
    return _copy(tr, np.arange(a, a + keep))


def splice(tr, pct, rng, pool):
    """Append a real segment so that pct% of the points are real (pool: list of real traces of the same dataset)."""
    n = len(tr['lat'])
    m = int(round(n * pct / (100 - pct)))
    if m == 0 or not pool:
        return _copy(tr)
    src = pool[int(rng.integers(0, len(pool)))]
    k = min(m, len(src['lat']))
    o = _copy(tr)
    dt = (o['t'][-1] - o['t'][-2]) if n > 1 else 15
    o['lat'] = np.concatenate([o['lat'], np.asarray(src['lat'][:k])])
    o['lon'] = np.concatenate([o['lon'], np.asarray(src['lon'][:k])])
    o['t'] = np.concatenate([o['t'], o['t'][-1] + dt * np.arange(1, k + 1)])
    return o


def apply(tr, name, rng, pool=None, cached=None):
    """Apply a (composite) attack. Precomputed (CACHED) parts are looked up in `cached`, keyed by the full attack
    prefix that ends with that part (e.g. 'noise30+mm'); the remaining parts are applied on top."""
    parts = name.split('+')
    o, start = tr, 0
    js = [i for i, x in enumerate(parts) if x in CACHED]
    if js:
        j = js[-1]
        key = '+'.join(parts[:j + 1])
        if cached is None or cached.get(key) is None:
            raise KeyError(f'no cached output for {key}')
        o, start = cached[key], j + 1
    if start == len(parts):
        return _copy(o)
    for part in parts[start:]:
        if part == 'none':
            o = _copy(o)
            continue
        m = re.fullmatch(r'([a-z]+)(\d+)', part)
        if not m:
            raise ValueError(part)
        kind, v = m.group(1), int(m.group(2))
        if kind == 'noise':
            o = noise(o, v, rng)
        elif kind == 'down':
            o = down(o, v, rng)
        elif kind == 'retime':
            o = retime(o, v, rng)
        elif kind == 'crop':
            o = crop(o, v, rng)
        elif kind == 'splice':
            o = splice(o, v, rng, pool)
        else:
            raise ValueError(part)
    return o


def cache_prefix(name):
    """The attack prefix that must be precomputed (ends with a CACHED part), or None."""
    parts = name.split('+')
    js = [i for i, x in enumerate(parts) if x in CACHED]
    return '+'.join(parts[:js[-1] + 1]) if js else None


