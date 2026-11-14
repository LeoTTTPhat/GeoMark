"""Access to preprocessed traces (written by src/preprocess.py), vocabularies, training sequences, releases."""
import os, glob
import numpy as np
from .common import load_pickle, save_pickle, BBOX
from .hexgrid import tokenize


def ds_path(cfg, ds):
    return os.path.join(cfg['data_dir'], f'{ds}.pkl')


_CACHE = {}


def load_ds(cfg, ds):
    """dict: traces = list of {uid, split, lat, lon, t, slot, ds}; tok = {res: list of int64 arrays (collapsed)};
    ppc = {res: points-per-collapsed-cell sample}; dt = sampling interval (s)."""
    k = (cfg['data_dir'], ds)
    if k not in _CACHE:
        _CACHE[k] = load_pickle(ds_path(cfg, ds))
    return _CACHE[k]


def split_idx(D, split):
    return [i for i, tr in enumerate(D['traces']) if tr['split'] == split]


def vocab_path(cfg, ds, res):
    return os.path.join(cfg['data_dir'], f'vocab_{ds}_r{res}.npy')


def get_vocab(cfg, ds, res):
    """Vocabulary = all fine cells occurring in the train split (sorted H3 ints)."""
    fp = vocab_path(cfg, ds, res)
    if os.path.exists(fp):
        return np.load(fp)
    D = load_ds(cfg, ds)
    cells = set()
    for i in split_idx(D, 'train'):
        cells.update(D['tok'][res][i].tolist())
    v = np.array(sorted(cells), dtype=np.int64)
    np.save(fp, v)
    return v


def to_task(cells, index):
    """H3 ints -> vocab indices; returns None if any cell is out of vocabulary."""
    out = []
    for c in cells:
        j = index.get(int(c))
        if j is None:
            return None
        out.append(j)
    return out


def real_sequences(cfg, ds, res, split, index, max_len, limit=None, seed=0):
    """(task sequences, slots) of real traces in a split, truncated to max_len cells, OOV traces dropped."""
    D = load_ds(cfg, ds)
    idx = split_idx(D, split)
    if limit and len(idx) > limit:
        idx = list(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    seqs, slots = [], []
    for i in idx:
        s = to_task(D['tok'][res][i][:max_len], index)
        if s is not None and len(s) >= 2:
            seqs.append(s); slots.append(D['traces'][i]['slot'])
    return seqs, slots


def real_traces(cfg, ds, split, limit=None, seed=0, min_pts=2):
    D = load_ds(cfg, ds)
    idx = [i for i in split_idx(D, split) if len(D['traces'][i]['lat']) >= min_pts]
    if limit and len(idx) > limit:
        idx = sorted(np.random.default_rng(seed).choice(idx, size=limit, replace=False).tolist())
    return [D['traces'][i] for i in idx]


def slot_distribution(cfg, ds):
    D = load_ds(cfg, ds)
    s = np.array([tr['slot'] for tr in D['traces'] if tr['split'] == 'train'])
    p = np.bincount(s, minlength=48).astype(float) + 1e-3
    return p / p.sum()


def t0_for_slot(slot, ds, rng, day0):
    """Absolute start time: random day (within the dataset year) + slot + uniform offset inside the 30-min slot."""
    from .common import TZ_OFFSET_H
    day = int(day0) + int(rng.integers(0, 365)) * 86400
    return day + int(slot) * 1800 + int(rng.integers(0, 1800)) - 3600 * TZ_OFFSET_H[ds]


# ---------------- releases ----------------
def release_dir(cfg, gen):
    return os.path.join(cfg['out_dir'], 'gen', gen)


def release_files(cfg, gen=None):
    pat = os.path.join(cfg['out_dir'], 'gen', gen or '*', '*.pkl')
    return sorted(f for f in glob.glob(pat) if '.part' not in f)


def load_release(path):
    return load_pickle(path)
