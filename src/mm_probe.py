"""Stage-1 feasibility probe: LeuvenMapMatching speed on the Beijing OSM graph with GeoLife traces (bbox-contained,
resampled to >= 15 s). Writes results/mm_probe.json. Usage: python src/mm_probe.py [n_traces]"""
import os, sys, glob, time, json, random
import numpy as np, pandas as pd, networkx as nx
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
B = (116.25, 39.85, 116.45, 40.03)
G = nx.read_graphml(os.path.join(ROOT, 'data', 'osm', 'beijing_drive.graphml'))
t0 = time.time()
mp = InMemMap('bj', use_latlon=True, use_rtree=True, index_edges=True)
for n, d in G.nodes(data=True): mp.add_node(int(n), (float(d['y']), float(d['x'])))
for u, v in G.edges(): mp.add_edge(int(u), int(v))
build = time.time() - t0
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'raw', 'geolife', 'Geolife Trajectories 1.3', 'Data', '*', 'Trajectory', '*.plt')))
random.Random(1).shuffle(files)
res = []
for f in files:
    if len(res) >= N: break
    df = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 4], names=['lat', 'lon', 'days'])
    m = (df.lon.between(B[0], B[2]) & df.lat.between(B[1], B[3]))
    if m.mean() < 0.95 or len(df) < 40: continue
    df = df[m]; t = (df.days.values - df.days.values[0]) * 86400
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= 15: keep.append(i)
    df = df.iloc[keep[:200]]
    path = list(zip(df.lat.values, df.lon.values))
    mt = DistanceMatcher(mp, max_dist=200, obs_noise=30, obs_noise_ne=75, max_lattice_width=5, non_emitting_states=True)
    s = time.time()
    try:
        _, last = mt.match(path); ok = last + 1
    except Exception as e:
        ok = -1
    res.append(dict(n_pts=len(path), matched=ok, sec=time.time() - s))
    print(res[-1], flush=True)
sec = [r['sec'] for r in res]
out = dict(graph_build_sec=build, n=len(res), median_sec=float(np.median(sec)), mean_sec=float(np.mean(sec)),
           median_pts=float(np.median([r['n_pts'] for r in res])), frac_fully_matched=float(np.mean([r['matched'] == r['n_pts'] for r in res])), traces=res)
json.dump(out, open(os.path.join(ROOT, 'results', 'mm_probe.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'traces'})
