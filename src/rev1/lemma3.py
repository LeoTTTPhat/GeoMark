"""Stage 7: Lemma 3 re-validation (reviewers: the Monte Carlo used geometric r7 cells, but seeds use the parent-defined
label par_{r-2}(cell_r(x)), whose region is a ragged union of fine cells; and Lemma 3(c) ignores the dilution by new,
null-behaving pairs that noise creates).

(1) Monte Carlo (Porto box, 40,000 uniform points per setting, same sampling as src/geometry_check.py): crossing
    probability of the fine cell (r9, r8), of the geometric coarse cell (latlng_to_cell(., r-2)) and of the
    parent-defined coarse / medium labels par_{r-2}(cell_r(x)), par_{r-1}(cell_r(x)), under Gaussian noise
    sigma = 10/30/100 m and fixed displacements d = 10/30/100 m. Also the perimeter/area ratio of the parent-defined
    r7 regions measured from the r9 tiling (boundary r9 edges), which replaces 4/(sqrt(3) a) in Lemma 3(b).
(2) Dilution check on the evaluation outputs: for GeoMark-KGW (3 keys, Porto and GeoLife small GPT) under noise and
    map-matching, the surviving fraction rho = surv_f, the attacked-pair signal fraction rho_att = surv_f n_f0 / n_f,
    and the measured green excess of attacked pairs vs the prediction rho_att * eps_0 (eps_0 = clean excess).
Output: results/rev1/lemma3.json. Model-free. Usage: python src/rev1/lemma3.py"""
import os, sys, math
import numpy as np
import h3
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import OUT, jdump, ppath


def mc():
    rng = np.random.default_rng(0)
    LAT0, LON0 = 41.15, -8.62
    MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
    N = 40000
    rows = []
    for r in (9, 8):
        a = h3.average_hexagon_edge_length(r, 'm')
        span = 4000.0 if r >= 8 else 20000.0
        lat = LAT0 + rng.uniform(-span, span, N) / MLAT; lon = LON0 + rng.uniform(-span, span, N) / MLON
        f0 = [h3.latlng_to_cell(x, y, r) for x, y in zip(lat, lon)]
        pc0 = np.array([h3.cell_to_parent(c, r - 2) for c in f0]); pm0 = np.array([h3.cell_to_parent(c, r - 1) for c in f0])
        g0 = np.array([h3.latlng_to_cell(x, y, r - 2) for x, y in zip(lat, lon)])
        f0 = np.array(f0)
        for kind, vals in (('gauss_sigma', (10, 30, 100)), ('fixed_d', (10, 30, 100))):
            for v in vals:
                if kind == 'gauss_sigma':
                    dx, dy = rng.normal(0, v, N), rng.normal(0, v, N)
                else:
                    th = rng.uniform(0, 2 * math.pi, N); dx, dy = v * np.cos(th), v * np.sin(th)
                la1, lo1 = lat + dy / MLAT, lon + dx / MLON
                f1 = [h3.latlng_to_cell(x, y, r) for x, y in zip(la1, lo1)]
                pc1 = np.array([h3.cell_to_parent(c, r - 2) for c in f1]); pm1 = np.array([h3.cell_to_parent(c, r - 1) for c in f1])
                g1 = np.array([h3.latlng_to_cell(x, y, r - 2) for x, y in zip(la1, lo1)])
                f1 = np.array(f1)
                rows.append(dict(res=r, kind=kind, value_m=v, q_fine=float((f1 != f0).mean()),
                                 q_coarse_geometric=float((g1 != g0).mean()), q_coarse_parent=float((pc1 != pc0).mean()),
                                 q_mid_parent=float((pm1 != pm0).mean())))
                print(rows[-1], flush=True)
    return rows


def parent_perimeter_ratio(r=9, n_coarse=60):
    """Perimeter/area of parent-defined r-2 regions (union of their r-level descendants) vs the regular hexagon."""
    c = h3.latlng_to_cell(41.15, -8.62, r - 2)
    coarse = list(h3.grid_disk(c, 3))[:n_coarse]
    a_r = h3.average_hexagon_edge_length(r, 'm')
    per, area = [], []
    for P in coarse:
        kids = [k for k in h3.grid_disk(h3.cell_to_center_child(P, r), 12) if h3.cell_to_parent(k, r - 2) == P]
        ks = set(kids)
        e = 0
        for k in kids:
            for nb in h3.grid_ring(k, 1):
                if nb not in ks:
                    e += 1
        per.append(e * h3.average_hexagon_edge_length(r, 'm')); area.append(len(kids) * 1.5 * math.sqrt(3) * a_r ** 2)
    ratio = float(np.mean(np.array(per) / np.array(area)))
    a_c = h3.average_hexagon_edge_length(r - 2, 'm')
    return dict(res=r, coarse_res=r - 2, mean_children=float(np.mean([a / (1.5 * math.sqrt(3) * a_r ** 2) for a in area])),
                perimeter_over_area_parent=ratio, perimeter_over_area_hexagon=4 / (math.sqrt(3) * a_c),
                ratio_parent_to_hexagon=ratio / (4 / (math.sqrt(3) * a_c)))


def dilution():
    out = {}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0'):
        for wm in ('gm_kgw', 'a_g25_d4'):
            clean = []
            for key in (0, 1, 2):
                f = ppath('results', 'eval', gen, f'{wm}__k{key}__s0__none.parquet')
                if os.path.exists(f):
                    d = pd.read_parquet(f)
                    clean.append(d)
            if not clean:
                continue
            c = pd.concat(clean)
            eps0 = float((c['stat_f'] * c['n_f']).sum() / c['n_f'].sum() - 0.25)
            for attack in ('noise10', 'noise30', 'noise100', 'mm', 'noise30+mm', 'down3'):
                ds_ = []
                for key in (0, 1, 2):
                    f = ppath('results', 'eval', gen, f'{wm}__k{key}__s0__{attack}.parquet')
                    if os.path.exists(f):
                        ds_.append(pd.read_parquet(f))
                if not ds_:
                    continue
                d = pd.concat(ds_)
                d = d[(d['n_f'] > 0) & (d['n_f0'] > 0)]
                surv = (d['surv_f'] * d['n_f0'])
                rho = float(surv.sum() / d['n_f0'].sum())
                rho_att = float(surv.sum() / d['n_f'].sum())
                eps_att = float((d['stat_f'] * d['n_f']).sum() / d['n_f'].sum() - 0.25)
                out[f'{gen}|{wm}|{attack}'] = dict(rho=rho, rho_att=rho_att, n_ratio=float(d['n_f'].sum() / d['n_f0'].sum()),
                                                   eps0=eps0, eps_att=eps_att, pred_rho=rho * eps0, pred_rho_att=rho_att * eps0)
                print(gen, wm, attack, out[f'{gen}|{wm}|{attack}'], flush=True)
    return out


def main():
    res = dict(mc=mc(), parent_perimeter=[parent_perimeter_ratio(9), parent_perimeter_ratio(8)], dilution=dilution())
    print(res['parent_perimeter'])
    jdump(res, 'lemma3.json')


if __name__ == '__main__':
    main()
