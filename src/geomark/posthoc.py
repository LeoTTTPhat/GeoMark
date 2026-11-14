"""Post-hoc cell recolouring with GeoMark's coarse-context hashing.
Used for (i) the post-hoc baseline (owner recolours an unwatermarked release with the true key: identical hashing,
no generator access), (ii) spoofing (adversary pushes real traces towards *estimated* green cells) and (iii) scrubbing
(adversary pushes watermarked traces towards estimated red cells). A cell may be replaced only by itself or a
neighbouring fine cell with the same coarse (r-2) parent (so seeds of later positions are preserved), adjacent to the
(possibly replaced) previous cell when the original was, and different from the next cell."""
import numpy as np
from .scheme import SeedState
from .hexgrid import parent, neighbors, is_neighbor, tokenize, collapse, cells_of_points, sample_in_cell
from .common import LocalProj, slot_of


def true_key_scorer(sch, prf):
    def f(seed, cands):
        u = prf.vector('fine', seed, cands)
        g = (u < sch.gamma).astype(float)
        if sch.uses_mid:
            mids = [parent(c, sch.mid_res) for c in cands]
            g = g + 0.5 * (prf.vector('mid', seed, mids) < sch.gamma)
        return g
    return f


def recolor(cells, slot, sch, scorer, target='green', rate=1.0, rng=None):
    """Returns (new_cells, changed bool array). scorer(seed, cand_cells) -> score (higher = greener; NaN = unknown)."""
    rng = rng or np.random.default_rng(0)
    cells = [int(c) for c in cells]
    n = len(cells)
    new = list(cells)
    changed = np.zeros(n, dtype=bool)
    st = SeedState(sch, slot)
    cr = sch.coarse_res
    for t in range(n):
        s = st.seed()
        c = cells[t]
        if rng.random() < rate:
            pc = parent(c, cr)
            prev = new[t - 1] if t else None
            nxt = cells[t + 1] if t + 1 < n else None
            need_adj = t > 0 and is_neighbor(c, cells[t - 1])
            cands = [c] + [x for x in neighbors(c, 1) if parent(x, cr) == pc and x != prev and x != nxt
                           and (not need_adj or is_neighbor(x, prev))]
            sc = np.asarray(scorer(s, cands), dtype=float)
            if target == 'red':
                sc = -sc
            sc = np.where(np.isnan(sc), -np.inf if target == 'green' else -np.inf, sc)
            if np.isfinite(sc).any():
                best = np.flatnonzero(sc == np.nanmax(sc))
                j = 0 if 0 in best else int(rng.choice(best))
                if j != 0:
                    new[t] = cands[j]; changed[t] = True
        st.update(new[t])
    return np.array(new, dtype=np.int64), changed


def rewrite_points(tr, res, new_cells, changed, rng):
    """Move the points of changed runs uniformly into their new cells (timestamps kept)."""
    cells = cells_of_points(tr['lat'], tr['lon'], res)
    col, first = collapse(cells)
    lat, lon = np.array(tr['lat'], dtype=float), np.array(tr['lon'], dtype=float)
    if len(col) != len(new_cells):
        return None
    proj = LocalProj.for_dataset(tr['ds'])
    ends = np.append(first[1:], len(lat))
    for i in np.flatnonzero(changed):
        a, b = first[i], ends[i]
        la, lo = sample_in_cell(int(new_cells[i]), b - a, rng, proj)
        lat[a:b], lon[a:b] = la, lo
    return dict(lat=lat, lon=lon, t=np.array(tr['t']), ds=tr['ds'], cells=np.asarray(new_cells),
                slot=tr.get('slot', slot_of(tr['t'][0], tr['ds'])))


def recolor_trace(tr, sch, scorer, target, rate, rng):
    cells = tokenize(tr['lat'], tr['lon'], sch.res)
    slot = slot_of(tr['t'][0], tr['ds'])
    new, ch = recolor(cells, slot, sch, scorer, target, rate, rng)
    out = rewrite_points(tr, sch.res, new, ch, rng)
    if out is not None:
        out['frac_changed'] = float(ch.mean()) if len(ch) else 0.0
    return out
