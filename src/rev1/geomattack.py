"""Stage 7: cheap geometric attacks the reviewers raised, and a deterministic robustness certificate.

(1) Boundary hopping (key-agnostic, coarse-preserving). Every point is moved just across the nearest edge of its fine
    (r) cell whose neighbour has the same coarse (r-2) parent: 'hopf' additionally requires the same medium (r-1)
    parent (destroys fine evidence only), 'hopfm' prefers a neighbour with a DIFFERENT medium parent (destroys fine
    and medium evidence). The new point lies 3 m inside the neighbour. Mean displacement is reported.
(2) Global translation of a whole release by 100 / 300 / 700 m in a random direction (shape, length and gyration
    preserved). Detector-side registration search: candidate inverse offsets on a 40 m grid in [-840, 840]^2
    (M = 43^2 = 1,849 offsets), Bonferroni p = min(1, M * min p). The search is coarse-to-fine (120 m grid, then the
    40 m grid around the 3 best coarse offsets); every evaluated offset belongs to the 40 m grid, so the reported p is
    at least the full-grid Bonferroni p and the test stays valid.
(3) Certificate (KGW, k = 1, no masking). For a perturbation that moves every point by at most d metres, keeps every
    point and the start bin, a fine pair (s_t, c_t) is reproduced whenever all points of positions t-1 and t lie more
    than d from the boundary of their fine cells (then c_{t-1}, c_t stay consecutive and s_t = (par_{r-2}(c_{t-1}), b)
    is unchanged). With u = #points within d of their cell boundary, the attacked trace has at most n_runs + 2u
    positions, so at most n_runs + 2u - s new pairs (s = #safe positions). Hence, deterministically,
        p(X') <= P[Bin(n_safe + N_new, gamma) >= T_safe]
    (fine and medium level, Fisher-combined). Computed per trace and for 100-trace pools at d = 0.5..20 m.
Output: results/rev1/geomattack.json. Model-free. Usage: python src/rev1/geomattack.py"""
import os, sys, argparse, json, math
import numpy as np
import h3.api.basic_int as h3i
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump, log10_binom_sf, fisher_log10
from geomark.common import LocalProj, seed_from, slot_of, Timer
from geomark.hexgrid import cells_of_points, collapse, parent, tokenize
from geomark.detect import pair_sets, score_pairs, pooled, detect_cells
from geomark.prf import PRF
from geomark.data import real_traces
from generate import wm_scheme

_EDGES = {}


def cell_edges(c, proj):
    """List of (x0, y0, x1, y1, neighbour) for the 6 edges of cell c in local metres."""
    k = (int(c), id(proj))
    v = _EDGES.get(k)
    if v is None:
        v = []
        for e in h3i.origin_to_directed_edges(int(c)):
            (a0, b0), (a1, b1) = h3i.directed_edge_to_boundary(e)
            x, y = proj.to_xy(np.array([a0, a1]), np.array([b0, b1]))
            v.append((float(x[0]), float(y[0]), float(x[1]), float(y[1]), int(h3i.get_directed_edge_destination(e))))
        _EDGES[k] = v
    return v


def seg_dist(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    qx, qy = x0 + t * dx, y0 + t * dy
    return np.hypot(px - qx, py - qy), qx, qy


def boundary_dist(lat, lon, res, proj):
    """Distance (m) of every point to the boundary of its own fine cell."""
    cells = cells_of_points(lat, lon, res)
    x, y = proj.to_xy(lat, lon)
    d = np.empty(len(x))
    for i, c in enumerate(cells):
        d[i] = min(seg_dist(x[i], y[i], *e[:4])[0] for e in cell_edges(c, proj))
    return cells, d


def hop(tr, sch, mode, proj, eps=3.0):
    """mode: hopf | hopfm, optionally followed by a distance limit in metres (e.g. hopfm30: only points within 30 m of
    an admissible edge are moved)."""
    import re
    m_ = re.fullmatch(r'(hopfm|hopf)(\d*)', mode)
    mode, max_d = m_.group(1), (float(m_.group(2)) if m_.group(2) else float('inf'))
    lat, lon = np.asarray(tr['lat'], float), np.asarray(tr['lon'], float)
    cells = cells_of_points(lat, lon, sch.res)
    x, y = proj.to_xy(lat, lon)
    nx, ny = x.copy(), y.copy()
    disp = np.zeros(len(x))
    cr, mr = sch.res - 2, sch.res - 1
    for i, c in enumerate(cells):
        best = None
        pc, pm = parent(c, cr), parent(c, mr)
        for (x0, y0, x1, y1, nb) in cell_edges(c, proj):
            if parent(nb, cr) != pc:
                continue
            same_mid = parent(nb, mr) == pm
            if mode == 'hopf' and not same_mid:
                continue
            dist, qx, qy = seg_dist(x[i], y[i], x0, y0, x1, y1)
            pen = 0.0 if (mode == 'hopf' or not same_mid) else 1e6      # hopfm: prefer a different medium parent
            if best is None or dist + pen < best[0]:
                best = (dist + pen, dist, qx, qy, nb, (x0 + x1) / 2, (y0 + y1) / 2)
        if best is None or best[1] > max_d:
            continue
        _, dist, qx, qy, nb, mx, my = best
        ncx, ncy = proj.to_xy(*h3i.cell_to_latlng(nb))
        for (bx, by) in ((qx, qy), (mx, my)):
            v = np.array([ncx - bx, ncy - by]); v /= max(np.linalg.norm(v), 1e-9)
            tx, ty = bx + eps * v[0], by + eps * v[1]
            la, lo = proj.to_ll(np.array([tx]), np.array([ty]))
            if h3i.latlng_to_cell(float(la[0]), float(lo[0]), sch.res) == nb:
                nx[i], ny[i] = tx, ty
                disp[i] = math.hypot(tx - x[i], ty - y[i])
                break
    la, lo = proj.to_ll(nx, ny)
    return dict(lat=la, lon=lo, t=np.asarray(tr['t']), ds=tr['ds']), disp


def translate(tr, dx, dy, proj):
    x, y = proj.to_xy(tr['lat'], tr['lon'])
    la, lo = proj.to_ll(x + dx, y + dy)
    return dict(lat=la, lon=lo, t=np.asarray(tr['t']), ds=tr['ds'])


def pool_lp(trs, sch, prf):
    pl = [pair_sets(*_cs(t, sch), sch) for t in trs]
    return pooled(pl, sch, prf)['lp']


def _cs(t, sch):
    cells = tokenize(t['lat'], t['lon'], sch.res)
    return cells, (slot_of(t['t'][0], t['ds']) if len(t['t']) else 0)


def registration_search(xy_traces, sch, prf, proj, R=840, step=40, coarse=120, top=3):
    """xy_traces: list of (x, y, t, ds). Returns (Bonferroni log10 p, best offset, M)."""
    grid = np.arange(-R, R + 1, step)
    M = len(grid) ** 2
    cache = {}

    def ev(ox, oy):
        k = (int(ox), int(oy))
        if k not in cache:
            trs = []
            for x, y, t, ds in xy_traces:
                la, lo = proj.to_ll(x + ox, y + oy)
                trs.append(dict(lat=la, lon=lo, t=t, ds=ds))
            cache[k] = pool_lp(trs, sch, prf)
        return cache[k]
    cg = np.arange(-R, R + 1, coarse)
    res = sorted(((ev(ox, oy), ox, oy) for ox in cg for oy in cg))[:top]
    for _, cx, cy in res:
        for ox in np.arange(cx - coarse, cx + coarse + 1, step):
            for oy in np.arange(cy - coarse, cy + coarse + 1, step):
                if abs(ox) <= R and abs(oy) <= R:
                    ev(ox, oy)
    lp, ox, oy = min((v, k[0], k[1]) for k, v in cache.items())
    return min(0.0, lp + math.log10(M)), (float(ox), float(oy)), M, len(cache)


def certificate(trs, sch, prf, proj, ds_list):
    """Certified log10 p (per trace and pooled over the list) for each d in ds_list."""
    info = []
    for tr in trs:
        cells_pt, bd = boundary_dist(np.asarray(tr['lat'], float), np.asarray(tr['lon'], float), sch.res, proj)
        col, first = collapse(cells_pt)
        ends = np.append(first[1:], len(cells_pt))
        posmin = np.array([bd[a:b].min() for a, b in zip(first, ends)])        # min boundary distance per position
        slot = slot_of(tr['t'][0], tr['ds'])
        fine, mid = pair_sets(col, slot, sch, dedup=False)                       # one pair per position (M0)
        info.append((col, posmin, bd, fine, mid))
    out = {}
    for d in ds_list:
        per_trace = []
        poolF, poolM, Nnew = {}, {}, 0
        Fall, Mall = {}, {}
        for col, posmin, bd, fine, mid in info:
            n = len(col)
            safe = np.zeros(n, bool)
            if n:
                ok = posmin > d
                safe[0] = ok[0]
                safe[1:] = ok[1:] & ok[:-1]
            u = int((bd <= d).sum())
            s = int(safe.sum())
            new = max(0, n + 2 * u - s)
            sf = {fine[t]: None for t in range(n) if safe[t]}
            sm = {mid[t]: None for t in range(n) if safe[t]} if mid else {}
            per_trace.append(_cert_lp(list(sf), list(sm), new, sch, prf))
            poolF.update(sf); poolM.update(sm); Nnew += new
        out[str(d)] = dict(per_trace=per_trace, pooled=_cert_lp(list(poolF), list(poolM), Nnew, sch, prf),
                           frac_pts_near=float(np.mean(np.concatenate([i[2] for i in info]) <= d)))
    return out


def _cert_lp(sf, sm, new, sch, prf):
    from geomark.detect import uniforms
    uf = uniforms(prf, sf, 'fine')
    nf = len(sf) + new
    lf = float(log10_binom_sf(int((uf < sch.gamma).sum()), nf, sch.gamma)) if nf else 0.0
    if not sch.uses_mid:
        return lf
    um = uniforms(prf, sm, 'mid')
    nm = len(sm) + new
    lm = float(log10_binom_sf(int((um < sch.gamma).sum()), nm, sch.gamma)) if nm else 0.0
    return float(fisher_log10(lf, lm, nf, nm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--parts', default='hop,translate,cert')
    a = ap.parse_args()
    cfg = cfg_rev1()
    fn = os.path.join(OUT, 'geomattack.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    T = Timer()
    parts = a.parts.split(',')
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0']
    nmax = 200 if a.quick else 1000
    if 'hop' in parts:
        for gen in gens:
            ds = cfg['generators'][gen]['dataset']; proj = LocalProj.for_dataset(ds)
            for wm in ('gm_kgw', 'exact_kgw', 'a_g25_d4', 'gm_gumbel', 'gm_kgw_eosn'):
                for key in (0, 1, 2):
                    rid = f'{wm}__k{key}__s0'
                    R = load_release(cfg, gen, rid)
                    if R is None:
                        continue
                    sch = wm_scheme(cfg, wm, 9); prf = PRF(key)
                    for mode in ('hopf', 'hopfm', 'hopfm30', 'hopfm60'):
                        k = f'hop|{gen}|{rid}|{mode}'
                        if k in out:
                            continue
                        trs, disps = [], []
                        for tr in R['traces'][:nmax]:
                            x, dsp = hop(tr, sch, mode, proj); trs.append(x); disps.append(dsp)
                        lps = [detect_cells(*_cs(t, sch), sch, prf)['lp'] for t in trs]
                        pools = [pool_lp(trs[j:j + 100], sch, prf) for j in range(0, len(trs) - 99, 100)]
                        d_all = np.concatenate(disps)
                        out[k] = dict(tpr_m3=float(np.mean(np.array(lps) <= -3)), pooled_lps=pools,
                                      pooled_median=float(np.median(pools)), pooled_frac_le_m3=float(np.mean(np.array(pools) <= -3)),
                                      mean_disp_m=float(d_all.mean()), median_disp_m=float(np.median(d_all)),
                                      frac_moved=float((d_all > 0).mean()), n=len(trs))
                        print(f"{k}: TPR {out[k]['tpr_m3']:.3f} pooled median {out[k]['pooled_median']:.1f} "
                              f"disp {out[k]['mean_disp_m']:.1f} m ({T():.0f}s)", flush=True)
                        jdump(out, 'geomattack.json')
                        if a.quick:
                            return
    if 'translate' in parts:
        for gen in gens + ['real']:
            for wm in ('gm_kgw', 'exact_kgw'):
                for key in (0, 1, 2):
                    for t_m in (100, 300, 700):
                        k = f'translate|{gen}|{wm}|k{key}|t{t_m}'
                        if k in out:
                            continue
                        if gen == 'real':
                            for ds in ('porto', 'geolife'):
                                kk = f'translate|real_{ds}|{wm}|k{key}|t{t_m}'
                                if kk in out or t_m != 300:
                                    continue
                                trs0 = real_traces(cfg, ds, 'test', limit=300, seed=1, min_pts=2)
                                out[kk] = _translate_block(trs0, ds, wm, key, t_m, cfg, kk)
                                jdump(out, 'geomattack.json')
                            continue
                        R = load_release(cfg, gen, f'{wm}__k{key}__s0')
                        if R is None:
                            continue
                        ds = cfg['generators'][gen]['dataset']
                        out[k] = _translate_block(R['traces'][:300], ds, wm, key, t_m, cfg, k)
                        print(f"{k}: plain {out[k]['plain_median']:.1f} registered {out[k]['reg_median']:.1f} ({T():.0f}s)", flush=True)
                        jdump(out, 'geomattack.json')
    if 'cert' in parts:
        dl = [0.5, 1, 2, 5, 10, 20]
        for gen in gens:
            ds = cfg['generators'][gen]['dataset']; proj = LocalProj.for_dataset(ds)
            for wm in ('gm_kgw', 'a_g25_d4', 'gm_kgw_eosn'):
                for key in (0, 1, 2):
                    k = f'cert|{gen}|{wm}|k{key}'
                    if k in out:
                        continue
                    R = load_release(cfg, gen, f'{wm}__k{key}__s0')
                    if R is None:
                        continue
                    sch = wm_scheme(cfg, wm, 9); prf = PRF(key)
                    trs = R['traces'][:nmax]
                    rec = {}
                    for j in range(0, len(trs) - 99, 100):
                        c = certificate(trs[j:j + 100], sch, prf, proj, dl)
                        for d, v in c.items():
                            r = rec.setdefault(d, dict(pooled=[], per_trace_tpr=[], frac_pts_near=[]))
                            r['pooled'].append(v['pooled'])
                            r['per_trace_tpr'].append(float(np.mean(np.array(v['per_trace']) <= -3)))
                            r['frac_pts_near'].append(v['frac_pts_near'])
                    for d, r in rec.items():
                        r['pooled_median'] = float(np.median(r['pooled'])); r['pooled_frac_le_m3'] = float(np.mean(np.array(r['pooled']) <= -3))
                        r['per_trace_tpr_mean'] = float(np.mean(r['per_trace_tpr'])); r['frac_pts_near_mean'] = float(np.mean(r['frac_pts_near']))
                    out[k] = rec
                    print(f"{k}: certified pooled median " + ', '.join(f"d={d}: {r['pooled_median']:.1f}" for d, r in rec.items()) + f" ({T():.0f}s)", flush=True)
                    jdump(out, 'geomattack.json')


def _translate_block(trs0, ds, wm, key, t_m, cfg, tag):
    proj = LocalProj.for_dataset(ds)
    sch = wm_scheme(cfg, wm, 9); prf = PRF(key)
    rng = np.random.default_rng(seed_from(tag))
    plain, reg, offs, nev = [], [], [], []
    per_trace = []
    for j in range(0, len(trs0) - 99, 100):
        th = rng.uniform(0, 2 * np.pi)
        dx, dy = t_m * math.cos(th), t_m * math.sin(th)
        xy = []
        moved = []
        for tr in trs0[j:j + 100]:
            x, y = proj.to_xy(tr['lat'], tr['lon'])
            xy.append((x + dx, y + dy, np.asarray(tr['t']), ds))
            la, lo = proj.to_ll(x + dx, y + dy)
            moved.append(dict(lat=la, lon=lo, t=np.asarray(tr['t']), ds=ds))
        plain.append(pool_lp(moved, sch, prf))
        per_trace += [detect_cells(*_cs(t, sch), sch, prf)['lp'] for t in moved]
        lp, best, M, n_ev = registration_search(xy, sch, prf, proj)
        reg.append(lp); offs.append((best, (-dx, -dy))); nev.append(n_ev)
    return dict(plain=plain, reg=reg, plain_median=float(np.median(plain)), reg_median=float(np.median(reg)),
                reg_frac_le_m3=float(np.mean(np.array(reg) <= -3)), per_trace_tpr=float(np.mean(np.array(per_trace) <= -3)),
                offsets=offs, M=M, n_evaluated=nev)


if __name__ == '__main__':
    main()
