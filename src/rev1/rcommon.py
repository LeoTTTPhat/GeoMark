"""Shared helpers for the Stage-7 (revision 1) analyses in src/rev1/. No model is loaded by anything in this module."""
import os, sys, json, math, glob
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from geomark.common import load_cfg, load_pickle, save_pickle, dump_json, p as ppath, seed_from, append_jsonl, read_jsonl  # noqa: E402
from geomark.prf import PRF, master_key, _mix, _GOLD, _S11  # noqa: E402
from geomark.data import load_ds, release_dir  # noqa: E402

OUT = ppath('results', 'rev1')
os.makedirs(OUT, exist_ok=True)
LN10 = math.log(10.0)


def cfg_rev1():
    return load_cfg(ppath('src', 'configs', 'rev1.yaml'))


def load_release(cfg, gen, rid):
    f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
    return load_pickle(f) if os.path.exists(f) else None


def real_cells(cfg, ds, res, split, n=None, seed=0):
    """Collapsed fine cells + slots of real traces of a split (preprocessed tokens; same as run_utility.real_cells)."""
    D = load_ds(cfg, ds)
    idx = [i for i, t in enumerate(D['traces']) if t['split'] == split]
    if n is not None and len(idx) > n:
        idx = np.random.default_rng(seed).choice(idx, size=n, replace=False)
    return [D['tok'][res][i] for i in idx], [D['traces'][i]['slot'] for i in idx]


# ---------------- vectorised PRF scoring of many pair lists under many keys ----------------
class PairIndex:
    """Maps (seed, cell) pairs to integer arrays (seed id, cell) once, so that scoring under many keys is vectorised."""

    def __init__(self):
        self.seed_id = {}
        self.seeds = []

    def sid(self, s):
        v = self.seed_id.get(s)
        if v is None:
            v = self.seed_id[s] = len(self.seeds)
            self.seeds.append(s)
        return v

    def encode(self, pairs):
        if not pairs:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        s, c = zip(*pairs)
        return np.fromiter((self.sid(x) for x in s), np.int64, len(s)), np.asarray(c, dtype=np.int64)


def subkeys(key_id, domain, seeds):
    """(a, b) uint64 arrays of the per-(domain, seed) subkeys for all seeds (same as PRF.subkey)."""
    import hashlib
    k = master_key(key_id)
    a = np.empty(len(seeds), np.uint64); b = np.empty(len(seeds), np.uint64)
    for i, s in enumerate(seeds):
        d = hashlib.blake2b(f'{domain}|{s!r}'.encode(), key=k, digest_size=16).digest()
        a[i] = np.uint64(int.from_bytes(d[:8], 'little')); b[i] = np.uint64(int.from_bytes(d[8:], 'little'))
    return a, b


def uniforms_vec(key_id, domain, seeds, sid, cells, cache=None):
    """Uniforms for encoded pairs under one key (fast PRF only, key ids < CRYPTO_KEY0)."""
    ck = (key_id, domain, len(seeds))
    if cache is not None and ck in cache:
        a, b = cache[ck]
    else:
        a, b = subkeys(key_id, domain, seeds)
        if cache is not None:
            cache.clear(); cache[ck] = (a, b)
    return PRF._u(a[sid], b[sid], cells)


def log10_binom_sf(T, n, gamma):
    """log10 P[Bin(n, gamma) >= T] (vectorised; conservative KGW tail); 0 where n == 0."""
    T = np.asarray(T); n = np.asarray(n)
    out = stats.binom.logsf(T - 1, n, gamma) / LN10
    return np.where(n > 0, out, 0.0)


def log10_gamma_sf(T, n):
    T = np.asarray(T, dtype=float); n = np.asarray(n)
    out = stats.gamma.logsf(T, np.maximum(n, 1)) / LN10
    return np.where(n > 0, out, 0.0)


def fisher_log10(lf, lm, nf, nm):
    """Fisher combination of two log10 p-values (falls back to the available level)."""
    lf = np.asarray(lf, float); lm = np.asarray(lm, float)
    X = -2.0 * LN10 * (lf + lm)
    both = stats.chi2.logsf(X, 4) / LN10
    return np.where((nf > 0) & (nm > 0), both, np.where(nf > 0, lf, lm))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def jdump(obj, name):
    dump_json(obj, os.path.join(OUT, name))
    print('wrote', os.path.join(OUT, name), flush=True)
