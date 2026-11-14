"""Stage-1 data check for GeoMark: prints basic statistics of GeoLife, Porto taxi and the OSM drive graphs,
plus a capacity preview (distinct H3 cells per trace after collapsing consecutive repeats) on a seeded sample.
Writes results/data_stats.json. Run under memguard:
  shared/memguard.py --name p1-geomark-trajectory-watermark-scope-datacheck -- .venv/bin/python src/data_check.py
Honours MEMGUARD_SCALE (shrinks the capacity-preview sample)."""
import os, sys, glob, json, time, random, gc
import numpy as np, pandas as pd, h3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
SCALE = float(os.environ.get('MEMGUARD_SCALE', '1'))
NSAMP = max(200, int(3000 * SCALE))
BBOX = {'beijing': (116.25, 39.85, 116.45, 40.03), 'porto': (-8.70, 41.10, -8.55, 41.20)}  # same as src/fetch_osm.py
stats = {}

def collapse(cells):
    out = []
    for c in cells:
        if not out or out[-1] != c: out.append(c)
    return out

def capacity(trajs, name):
    """trajs: list of (lat, lon) arrays. Distinct consecutive cells per trace at r7/r8/r9, unique (coarse-ctx, cell) pairs."""
    res = {}
    for r in (7, 8, 9):
        L, U, P = [], [], []
        for lat, lon in trajs:
            cells = [h3.latlng_to_cell(a, b, r) for a, b in zip(lat, lon)]
            seq = collapse(cells)
            L.append(len(seq)); U.append(len(set(seq)))
            if r >= 8:  # unique (parent_{r-2}(prev), cell) pairs, k=1, no time bin
                par = [h3.cell_to_parent(c, r - 2) for c in seq]
                P.append(len(set(zip(par[:-1], seq[1:]))))
        q = lambda x: {k: float(v) for k, v in zip(['p10', 'p50', 'p90', 'mean'], [*np.percentile(x, [10, 50, 90]), np.mean(x)])}
        res[f'r{r}'] = {'collapsed_len': q(L), 'unique_cells': q(U)}
        if P: res[f'r{r}']['unique_seed_cell_pairs_k1'] = q(P)
    print(f'  [{name}] capacity preview on {len(trajs)} traces:')
    for r, v in res.items():
        print(f'    {r}: collapsed len median {v["collapsed_len"]["p50"]:.0f} (p10 {v["collapsed_len"]["p10"]:.0f}, p90 {v["collapsed_len"]["p90"]:.0f}); '
              f'unique cells median {v["unique_cells"]["p50"]:.0f}' + (f'; unique (seed,cell) k=1 median {v["unique_seed_cell_pairs_k1"]["p50"]:.0f}' if 'unique_seed_cell_pairs_k1' in v else ''))
    return res

def in_bbox(lat, lon, b):
    return (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])

# ---------------- GeoLife ----------------
t = time.time()
files = sorted(glob.glob(os.path.join(RAW, 'geolife', 'Geolife Trajectories 1.3', 'Data', '*', 'Trajectory', '*.plt')))
users = {f.split(os.sep)[-3] for f in files}
npts, dts, inb, durations = 0, [], 0, []
rng = random.Random(0); samp_idx = set(rng.sample(range(len(files)), min(NSAMP, len(files))))
samp = []
for i, f in enumerate(files):
    df = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 4], names=['lat', 'lon', 'days'])
    npts += len(df)
    m = in_bbox(df.lat.values, df.lon.values, BBOX['beijing']); inb += int(m.sum())
    if len(df) > 1:
        d = np.diff(df.days.values) * 86400
        dts.append(np.median(d)); durations.append((df.days.values[-1] - df.days.values[0]) * 24)
    if i in samp_idx and m.mean() > 0.9 and len(df) >= 20:
        samp.append((df.lat.values[m], df.lon.values[m]))
g = {'users': len(users), 'trajectories': len(files), 'points': npts, 'points_in_bbox_frac': inb / max(npts, 1),
     'median_sampling_interval_s': float(np.median(dts)), 'median_traj_duration_h': float(np.median(durations)),
     'bbox': BBOX['beijing']}
print(f'GeoLife: {g["users"]} users, {g["trajectories"]} trajectories, {npts:,} points; {100*g["points_in_bbox_frac"]:.1f}% in central-Beijing bbox; '
      f'median sampling interval {g["median_sampling_interval_s"]:.1f}s; median duration {g["median_traj_duration_h"]:.2f}h ({time.time()-t:.0f}s)')
g['capacity'] = capacity(samp, 'GeoLife (bbox-contained sample)')
stats['geolife'] = g; del samp; gc.collect()

# ---------------- Porto ----------------
t = time.time()
pcsv = glob.glob(os.path.join(RAW, 'porto', '**', 'train.csv'), recursive=True) or \
       glob.glob(os.path.join(RAW, 'porto', 'train_partial.csv'))  # fallback: salvaged from truncated zip
if pcsv:
    df = pd.read_csv(pcsv[0], usecols=['TRIP_ID', 'CALL_TYPE', 'TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE'])
    n = len(df); lens = df.POLYLINE.str.count(r'\],\[').values + 1
    lens[df.POLYLINE.str.len().values <= 2] = 0
    p = {'trips': n, 'taxis': int(df.TAXI_ID.nunique()), 'points': int(lens.sum()), 'missing_data_trips': int(df.MISSING_DATA.sum()),
         'empty_trips': int((lens == 0).sum()), 'median_points_per_trip': float(np.median(lens[lens > 0])),
         'sampling_interval_s': 15, 'call_type_counts': df.CALL_TYPE.value_counts().to_dict(),
         'time_range_utc': [str(pd.to_datetime(df.TIMESTAMP.min(), unit='s')), str(pd.to_datetime(df.TIMESTAMP.max(), unit='s'))],
         'bbox': BBOX['porto'], 'source_file': os.path.basename(pcsv[0])}
    print(f'Porto: {n:,} trips, {p["taxis"]} taxis, {p["points"]:,} points (15 s sampling); median {p["median_points_per_trip"]:.0f} pts/trip; '
          f'{p["empty_trips"]} empty, {p["missing_data_trips"]} MISSING_DATA; {p["time_range_utc"][0]} .. {p["time_range_utc"][1]} ({time.time()-t:.0f}s)')
    rng = np.random.default_rng(0); cand = rng.choice(np.where(lens >= 20)[0], size=min(NSAMP * 2, int((lens >= 20).sum())), replace=False)
    samp = []
    for j in cand:
        a = np.array(json.loads(df.POLYLINE.values[j]))
        lat, lon = a[:, 1], a[:, 0]
        m = in_bbox(lat, lon, BBOX['porto'])
        if m.all(): samp.append((lat, lon))
        if len(samp) >= NSAMP: break
    p['bbox_contained_frac_in_sample'] = len(samp) / max(1, len(cand))
    p['capacity'] = capacity(samp, 'Porto (bbox-contained sample)')
    stats['porto'] = p; del df, samp; gc.collect()
else:
    print('Porto: train.csv not found under data/raw/porto (download/unzip pending)'); stats['porto'] = None

# ---------------- OSM + H3 vocab ----------------
import networkx as nx
for name in ('porto', 'beijing'):
    fn = os.path.join(ROOT, 'data', 'osm', f'{name}_drive.graphml')
    b = BBOX[name]
    poly = h3.LatLngPoly([(b[1], b[0]), (b[1], b[2]), (b[3], b[2]), (b[3], b[0])])
    vocab = {f'r{r}': len(h3.polygon_to_cells(poly, r)) for r in (7, 8, 9)}
    o = {'h3_cells_in_bbox': vocab}
    if os.path.exists(fn):
        G = nx.read_graphml(fn); o.update(nodes=G.number_of_nodes(), edges=G.number_of_edges()); del G
    print(f'OSM {name}: {o.get("nodes")} nodes, {o.get("edges")} edges; H3 cells in bbox {vocab}')
    stats[f'osm_{name}'] = o
stats['h3_avg_edge_m'] = {f'r{r}': h3.average_hexagon_edge_length(r, 'm') for r in (6, 7, 8, 9, 10)}
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
json.dump(stats, open(os.path.join(ROOT, 'results', 'data_stats.json'), 'w'), indent=1, default=str)
print('wrote results/data_stats.json')
