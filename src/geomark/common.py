"""Shared helpers: paths, config loading, MEMGUARD_SCALE, deterministic seeding, JSONL checkpointing, local projection."""
import os, json, gc, hashlib, random, time, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project folder
SCALE = float(os.environ.get('MEMGUARD_SCALE', '1'))
ATTEMPT = int(os.environ.get('MEMGUARD_ATTEMPT', '0'))

# Dataset constants (bbox = lon_min, lat_min, lon_max, lat_max; same as src/fetch_osm.py)
BBOX = {'porto': (-8.70, 41.10, -8.55, 41.20), 'geolife': (116.25, 39.85, 116.45, 40.03)}
TZ_OFFSET_H = {'porto': 0, 'geolife': 8}       # local time = UTC + offset (Porto WET; DST ignored)
OSM_GRAPH = {'porto': 'data/osm/porto_drive.graphml', 'geolife': 'data/osm/beijing_drive.graphml'}
N_SLOTS = 48                                   # generator start-time slots (30 min)


def p(*a):
    """Absolute path inside the project."""
    return os.path.join(ROOT, *a)


def load_cfg(path):
    import yaml
    if not os.path.isabs(path):
        path = path if os.path.exists(path) else p(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # inherit from a base config if requested (shallow per top-level key, deep for dicts)
    if 'inherit' in cfg:
        base = load_cfg(os.path.join(os.path.dirname(path), cfg.pop('inherit')))
        cfg = deep_update(base, cfg)
    for k in ('out_dir', 'data_dir', 'model_dir'):
        cfg[k] = p(cfg[k])
        os.makedirs(cfg[k], exist_ok=True)
    return cfg


def deep_update(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def scaled(n, minimum=1):
    """Shrink a batch size / worker count by MEMGUARD_SCALE."""
    return max(minimum, int(round(n * SCALE)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def stable_hash(*parts, mod=None):
    h = int.from_bytes(hashlib.blake2b('|'.join(map(str, parts)).encode(), digest_size=8).digest(), 'little')
    return h % mod if mod else h


def seed_from(*parts):
    return stable_hash(*parts) % (2 ** 31 - 1)


# ---------------- JSONL / pickle checkpoint helpers ----------------
def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn last line after a kill
    return out


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(rec, default=_json_default) + '\n')
        f.flush()


def done_keys(path, key='key'):
    return {r[key] for r in read_jsonl(path) if key in r}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written checkpoint


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def free_torch():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def device():
    import torch
    d = os.environ.get('GEOMARK_DEVICE')
    if d:
        return torch.device(d)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return time.time() - self.t


# ---------------- local metric projection ----------------
class LocalProj:
    """Equirectangular projection around (lat0, lon0); error < 0.1% inside a 20 km city box."""
    R = 6371008.8

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = np.cos(np.radians(lat0)) * np.pi / 180 * self.R
        self.ky = np.pi / 180 * self.R

    @classmethod
    def for_dataset(cls, ds):
        b = BBOX[ds]
        return cls((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_ll(self, x, y):
        return np.asarray(y) / self.ky + self.lat0, np.asarray(x) / self.kx + self.lon0


def slot_of(t_unix, ds):
    """30-min start slot (0..47) of a unix timestamp in local time."""
    m = ((int(t_unix) + 3600 * TZ_OFFSET_H[ds]) % 86400) // 60
    return int(m // 30)
