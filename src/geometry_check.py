"""Monte-Carlo check of Lemma 3 (fine-cell crossing probability) on real H3 cells around Porto.
Point uniform inside an H3 cell (rejection sampling), displaced by (a) isotropic Gaussian noise sigma per axis,
(b) fixed distance d in a uniformly random direction. Compares with closed forms using the hexagon perimeter/area ratio
  Gaussian:   P ~= (P/A) * sigma / sqrt(2*pi) = 0.921 sigma / a
  fixed d:    P ~= (P/A) * d / pi            = 0.735 d / a       (worst case over directions: 2.309 d / a)
Writes results/geometry_mc.json. Cheap (CPU, < 2 min)."""
import os, json, math
import numpy as np, h3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
LAT0, LON0 = 41.15, -8.62
MLAT = 111320.0; MLON = 111320.0 * math.cos(math.radians(LAT0))
N = 40000
out = []
for r in (6, 7, 8, 9):
    a = h3.average_hexagon_edge_length(r, 'm')
    # sample uniform points in a box around Porto, keep their cell assignment (uniform-in-cell on average)
    span = 4000.0 if r >= 8 else 20000.0
    lat = LAT0 + rng.uniform(-span, span, N) / MLAT; lon = LON0 + rng.uniform(-span, span, N) / MLON
    c0 = np.array([h3.latlng_to_cell(x, y, r) for x, y in zip(lat, lon)])
    for kind, vals in (('gauss_sigma', (10, 30, 100)), ('fixed_d', (10, 30, 100))):
        for v in vals:
            if kind == 'gauss_sigma':
                dx, dy = rng.normal(0, v, N), rng.normal(0, v, N); pred = 4 / math.sqrt(3) / a * v / math.sqrt(2 * math.pi)
            else:
                th = rng.uniform(0, 2 * math.pi, N); dx, dy = v * np.cos(th), v * np.sin(th); pred = 4 / math.sqrt(3) / a * v / math.pi
            c1 = np.array([h3.latlng_to_cell(x + ddy / MLAT, y + ddx / MLON, r) for x, y, ddx, ddy in zip(lat, lon, dx, dy)])
            emp = float((c1 != c0).mean())
            out.append(dict(res=r, edge_m=a, kind=kind, value_m=v, empirical=emp, closed_form=min(pred, 1.0),
                            worst_case_bound=min(4 / math.sqrt(3) * v / a, 1.0) if kind == 'fixed_d' else None))
            print(f'r{r} a={a:7.1f}m {kind:11s} {v:4d}m  empirical {emp:.4f}  closed-form {min(pred,1):.4f}')
json.dump(out, open(os.path.join(ROOT, 'results', 'geometry_mc.json'), 'w'), indent=1)
