"""Stage 8: boundary hopping at release size, a reverse-hop / edge-proximity defense, an adaptive hopper, and dilution
(reviewers: breaking attacks were reported only for 100-trace pools; hopping leaves points metres from an edge, which is
detectable and invertible; no utility was reported for hopped data; dilution stopped at 50%).

Release R_A = first 10,000 traces of the key-0 stealing corpus (GeoMark-KGW k=1, EOS-neutral, step-neutral).
Attacks: hopfm (src/rev1/geomattack.hop: every point moved 3 m across the nearest admissible edge into a neighbour with the
same coarse and, where possible, a different medium parent) and hoprand (ADAPTIVE: same neighbour choice, but the point is
placed uniformly at random inside the neighbour cell, so it is not near an edge).
Defense: reverse hop = every point within 10 m of its fine-cell boundary is moved 3 m across the nearest edge (any
neighbour); the detector reports min(1, 2 min(p_plain, p_reversed)) (Bonferroni over the two pair sets, exact by Lemma 1).
Edge proximity = share of points within 5 m of their fine-cell boundary (a tamper indicator, not a provenance test).
Pools: 100 traces (first 2,000), 1,000 and 10,000 traces. Utility of the attacked releases on the first 5,000 traces.
Dilution: d% of the dataset from R_A, the rest from an unwatermarked small-GPT release (none__k0__s20), d = 1, 2, 5, 10, 20;
datasets of 1,000 (10 per d) and 10,000 traces (3 per d).
Output: results/rev2/breakatk.json. Model-free. Usage: python src/rev2/breakatk.py"""
import os, sys, math
from multiprocessing import Pool
import numpy as np
import h3.api.basic_int as h3i
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, OUT2, save_pickle, load_pickle
from geomark.common import seed_from, Timer, scaled, LocalProj
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, pooled
from geomark.hexgrid import cells_of_points, tokenize, sample_in_cell
from geomark.utility import utility
from geomark.data import load_ds
from run_utility import real_cells
from rev1.utility import fixed_metrics
from rev1.geomattack import hop, cell_edges, seg_dist, boundary_dist
from generate import wm_scheme

D = os.path.join(OUT2, 'breakatk')
CORPORA = [('k1', 'gm_kgw'), ('eosn', 'gm_kgw_eosn'), ('sn', 'gm_kgw_sn')]
_SCH = None


def _init(sch):
    global _SCH
    _SCH = sch


def _hop(args):
    tr, mode, seed = args
    proj = LocalProj.for_dataset(tr['ds'])
    x, disp = hop(tr, _SCH, 'hopfm', proj)
    if mode == 'hoprand':
        rng = np.random.default_rng(seed)
        old = cells_of_points(np.asarray(tr['lat'], float), np.asarray(tr['lon'], float), _SCH.res)
        new = cells_of_points(np.asarray(x['lat'], float), np.asarray(x['lon'], float), _SCH.res)
        la, lo = np.array(x['lat'], float), np.array(x['lon'], float)
        ox, oy = proj.to_xy(np.asarray(tr['lat'], float), np.asarray(tr['lon'], float))
        for i in np.flatnonzero(new != old):
            a, b = sample_in_cell(int(new[i]), 1, rng, proj)
            la[i], lo[i] = a[0], b[0]
        nx, ny = proj.to_xy(la, lo)
        disp = np.where(new != old, np.hypot(nx - ox, ny - oy), 0.0)
        x = dict(lat=la, lon=lo, t=x['t'], ds=x['ds'])
    near = _near(x, proj)
    return x, float(disp.mean()) if len(disp) else 0.0, near


def _near(tr, proj, d=5.0):
    _, bd = boundary_dist(np.asarray(tr['lat'], float), np.asarray(tr['lon'], float), _SCH.res, proj)
    return int((bd <= d).sum()), int(len(bd))


def _rev(tr, eps=10.0, inside=3.0):
    proj = LocalProj.for_dataset(tr['ds'])
    lat, lon = np.asarray(tr['lat'], float), np.asarray(tr['lon'], float)
    cells = cells_of_points(lat, lon, _SCH.res)
    x, y = proj.to_xy(lat, lon)
    nx, ny = x.copy(), y.copy()
    for i, c in enumerate(cells):
        best = None
        for (x0, y0, x1, y1, nb) in cell_edges(c, proj):
            dist, qx, qy = seg_dist(x[i], y[i], x0, y0, x1, y1)
            if best is None or dist < best[0]:
                best = (dist, qx, qy, nb)
        if best is None or best[0] > eps:
            continue
        dist, qx, qy, nb = best
        ncx, ncy = proj.to_xy(*h3i.cell_to_latlng(nb))
        v = np.array([ncx - qx, ncy - qy]); v /= max(np.linalg.norm(v), 1e-9)
        nx[i], ny[i] = qx + inside * v[0], qy + inside * v[1]
    la, lo = proj.to_ll(nx, ny)
    return dict(lat=la, lon=lo, t=np.asarray(tr['t']), ds=tr['ds'])


def _near_only(tr):
    return _near(tr, LocalProj.for_dataset(tr['ds']))


def pools_lp(pl, sch, prf, m, n_max=None):
    src = pl[:n_max] if n_max else pl
    return [pooled(src[i:i + m], sch, prf)['lp'] for i in range(0, len(src) - m + 1, m)]


def summ(lps):
    a = np.array(lps)
    return dict(n_pools=len(a), median_mlp=float(-np.median(a)) if len(a) else None,
                frac_le_m3=float((a <= -3).mean()) if len(a) else None)


def main():
    cfg = cfg_rev2()
    os.makedirs(D, exist_ok=True)
    out = jload2('breakatk.json')
    gen, ds, res = 'gpt_porto_r9_s0', 'porto', 9
    T = Timer()
    real_u = real_cells(cfg, ds, res, 'test', 5000, 0)
    Dd = load_ds(cfg, ds)
    train_set = {tuple(int(x) for x in Dd['tok'][res][i]) for i, t in enumerate(Dd['traces']) if t['split'] == 'train'}
    filler = load_release(cfg, gen, 'none__k0__s20')
    for name, wm in CORPORA:
        R = load_release(cfg, gen, f'{wm}__k0__s10')
        if R is None:
            print('missing', name, flush=True); continue
        sch = wm_scheme(cfg, wm, res)
        prf = PRF(0)
        RA = R['traces'][:10000]
        pool = Pool(scaled(16, 1), initializer=_init, initargs=(sch,))
        rec = out.get(name, {})
        for var in ('clean', 'hopfm', 'hoprand'):
            if var in rec:
                continue
            cf = os.path.join(D, f'{name}_{var}.pkl')
            if os.path.exists(cf):
                X, disp, near = load_pickle(cf)
            elif var == 'clean':
                X = RA
                nr = pool.map(_near_only, RA[:5000], chunksize=20)
                disp, near = 0.0, nr
                save_pickle((X, disp, near), cf)
            else:
                rs = pool.map(_hop, [(t, var, seed_from('breakatk', name, var, i)) for i, t in enumerate(RA)], chunksize=20)
                X = [r[0] for r in rs]; disp = float(np.mean([r[1] for r in rs])); near = [r[2] for r in rs[:5000]]
                save_pickle((X, disp, near), cf)
            Xr = pool.map(_rev, X, chunksize=20)
            pl = [pair_sets(*trace_cells(t, sch), sch) for t in X]
            plr = [pair_sets(*trace_cells(t, sch), sch) for t in Xr]
            r = dict(mean_disp_m=disp, edge5m_share=float(sum(a for a, _ in near) / max(1, sum(b for _, b in near))))
            for m, nmax in ((100, 2000), (1000, None), (10000, None)):
                a = pools_lp(pl, sch, prf, m, nmax); b = pools_lp(plr, sch, prf, m, nmax)
                comb = [min(0.0, min(x, y) + math.log10(2)) for x, y in zip(a, b)]
                r[f'm{m}'] = dict(plain=summ(a), reverse=summ(b), combined=summ(comb))
            cells = [tokenize(t['lat'], t['lon'], res) for t in X[:5000]]
            u = utility(cells, real_u, ds, res, train_set)
            u.update(fixed_metrics(cells, real_u, ds, None, None))
            r['utility'] = {k: v for k, v in u.items() if isinstance(v, (int, float))}
            rec[var] = r
            fm = lambda x: 'na' if x is None else f'{x:.1f}'
            print(f"{name} {var}: disp {disp:.0f} m, edge5 {r['edge5m_share']:.3f}; " + '; '.join(
                f"{m}: {fm(r[m]['plain']['median_mlp'])}/{fm(r[m]['combined']['median_mlp'])}" for m in ('m100', 'm1000', 'm10000'))
                  + f"; bigram {r['utility'].get('jsd_bigram', float('nan')):.3f} ({T():.0f}s)", flush=True)
            out[name] = rec; jdump2(out, 'breakatk.json')
        pool.close()
        if 'dilution' not in rec and filler is not None:
            F = filler['traces']
            plw = [pair_sets(t['cells'], t['slot'], sch) for t in RA]
            plf = [pair_sets(t['cells'], t['slot'], sch) for t in F]
            dil = {}
            for size, reps in ((1000, 10), (10000, 3)):
                for d in (0.01, 0.02, 0.05, 0.1, 0.2):
                    nw = int(round(d * size))
                    lps = []
                    for j in range(reps):
                        rng = np.random.default_rng(seed_from('dilution', name, size, d, j))
                        wi = rng.choice(len(plw), nw, replace=False); fi = rng.choice(len(plf), size - nw, replace=False)
                        lps.append(pooled([plw[i] for i in wi] + [plf[i] for i in fi], sch, prf)['lp'])
                    dil[f'{size}|{d}'] = dict(summ(lps), lps=[round(float(x), 2) for x in lps])
            rec['dilution'] = dil
            print(f"{name} dilution: " + ', '.join(f"{k}: {v['median_mlp']:.1f}" for k, v in dil.items()) + f' ({T():.0f}s)', flush=True)
            out[name] = rec; jdump2(out, 'breakatk.json')


if __name__ == '__main__':
    main()
