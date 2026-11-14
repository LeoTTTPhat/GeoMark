"""H3 tokenisation: points -> fine cells, run-collapse, parents (par() defined on the fine cell, as in the paper),
and release decoding cells -> points (D1 centroid, D2 uniform-in-cell, D3 road-snapped)."""
import numpy as np
import h3.api.basic_int as h3i
from .common import LocalProj

_PAR = {}
_GEOM = {}


def cells_of_points(lat, lon, res):
    return np.fromiter((h3i.latlng_to_cell(float(a), float(b), res) for a, b in zip(lat, lon)), dtype=np.int64, count=len(lat))


def collapse(cells):
    """Run-collapse consecutive repeats; returns (collapsed cells, index of the first point of each run)."""
    cells = np.asarray(cells, dtype=np.int64)
    if len(cells) == 0:
        return cells, np.zeros(0, dtype=np.int64)
    keep = np.ones(len(cells), dtype=bool)
    keep[1:] = cells[1:] != cells[:-1]
    idx = np.nonzero(keep)[0]
    return cells[idx], idx


def tokenize(lat, lon, res):
    return collapse(cells_of_points(lat, lon, res))[0]


def parent(c, res):
    key = (int(c), res)
    v = _PAR.get(key)
    if v is None:
        v = h3i.cell_to_parent(int(c), res)
        if len(_PAR) > 5_000_000:
            _PAR.clear()
        _PAR[key] = v
    return v


def parents(cells, res):
    return np.fromiter((parent(c, res) for c in cells), dtype=np.int64, count=len(cells))


def resolution(c):
    return h3i.get_resolution(int(c))


def neighbors(c, ring=1):
    return [x for x in h3i.grid_disk(int(c), ring) if x != int(c)]


def is_neighbor(a, b):
    return bool(h3i.are_neighbor_cells(int(a), int(b)))


def centroid(c):
    return h3i.cell_to_latlng(int(c))


def _geom(c, proj):
    key = (int(c), id(proj))
    g = _GEOM.get(key)
    if g is None:
        lat, lon = h3i.cell_to_latlng(int(c))
        bx, by = proj.to_xy(*zip(*h3i.cell_to_boundary(int(c))))
        cx, cy = proj.to_xy(lat, lon)
        g = (float(cx), float(cy), float(np.max(np.hypot(np.asarray(bx) - cx, np.asarray(by) - cy))))
        if len(_GEOM) > 2_000_000:
            _GEOM.clear()
        _GEOM[key] = g
    return g


def sample_in_cell(c, n, rng, proj):
    """n points uniform inside hexagon c (rejection from the circumscribed disk). Returns lat, lon arrays."""
    res = h3i.get_resolution(int(c))
    cx, cy, R = _geom(c, proj)
    outx, outy = [], []
    need = n
    for _ in range(50):
        m = int(need * 1.4) + 4
        r = R * np.sqrt(rng.random(m)); a = rng.random(m) * 2 * np.pi
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        la, lo = proj.to_ll(x, y)
        ok = np.fromiter((h3i.latlng_to_cell(float(u), float(v), res) == int(c) for u, v in zip(la, lo)), dtype=bool, count=m)
        outx.extend(x[ok][:need]); outy.extend(y[ok][:need])
        need = n - len(outx)
        if need <= 0:
            break
    if need > 0:  # pathological; fall back to centroid
        outx.extend([cx] * need); outy.extend([cy] * need)
    return proj.to_ll(np.array(outx[:n]), np.array(outy[:n]))


class Decoder:
    """Cells -> GPS-like points with timestamps.
    mode: 'centroid' (D1), 'uniform' (D2, default), 'road' (D3: nearest OSM node inside the cell, else uniform)."""

    def __init__(self, ds, ppc_hist, dt, mode='uniform', road_nodes=None):
        self.ds, self.proj = ds, LocalProj.for_dataset(ds)
        vals, cnts = np.unique(np.clip(np.asarray(ppc_hist, dtype=int), 1, 20), return_counts=True)
        self.ppc_vals, self.ppc_p = vals, cnts / cnts.sum()   # empirical points-per-collapsed-cell distribution
        self.dt, self.mode = dt, mode
        self.road = None
        if mode == 'road' and road_nodes is not None:
            from scipy.spatial import cKDTree
            x, y = self.proj.to_xy(road_nodes[:, 0], road_nodes[:, 1])
            self.road = (cKDTree(np.c_[x, y]), np.c_[x, y])

    def decode(self, cells, t0, rng):
        m = rng.choice(self.ppc_vals, size=len(cells), p=self.ppc_p)
        lats, lons = [], []
        for c, k in zip(cells, m):
            if self.mode == 'centroid':
                la, lo = centroid(c); la, lo = np.full(k, la), np.full(k, lo)
            elif self.mode == 'road' and self.road is not None:
                la, lo = self._road(c, k, rng)
            else:
                la, lo = sample_in_cell(c, k, rng, self.proj)
            lats.append(la); lons.append(lo)
        lat, lon = np.concatenate(lats), np.concatenate(lons)
        t = int(t0) + self.dt * np.arange(len(lat), dtype=np.int64)
        return lat, lon, t

    def _road(self, c, k, rng):
        res = h3i.get_resolution(int(c))
        cx, cy, R = _geom(c, self.proj)
        idx = self.road[0].query_ball_point([cx, cy], R)
        xy = self.road[1][idx] if idx else np.zeros((0, 2))
        la, lo = self.proj.to_ll(xy[:, 0], xy[:, 1]) if len(xy) else (np.zeros(0), np.zeros(0))
        ok = [i for i in range(len(la)) if h3i.latlng_to_cell(float(la[i]), float(lo[i]), res) == int(c)]
        if not ok:
            return sample_in_cell(c, k, rng, self.proj)
        pick = rng.choice(ok, size=k)
        return la[pick], lo[pick]
