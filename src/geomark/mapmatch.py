"""HMM map-matching to the OSM drive graph (LeuvenMapMatching DistanceMatcher) with restart-after-break.
Two adversary outputs:
  'mm'   : every matched observation snapped to its projection on the matched road segment (timestamps kept);
           unmatched observations are dropped.
  'mmrs' : the matched node path re-sampled at the trace's median step length (cleaning pipeline that replaces the
           trace by its road route; timestamps linearly interpolated).
"""
import os, signal
import numpy as np
from .common import p, OSM_GRAPH, LocalProj

_MAP = {}


def get_map(ds):
    if ds not in _MAP:
        import networkx as nx
        from leuvenmapmatching.map.inmem import InMemMap
        G = nx.read_graphml(p(OSM_GRAPH[ds]))
        mp = InMemMap(ds, use_latlon=True, use_rtree=True, index_edges=True)
        for n, d in G.nodes(data=True):
            mp.add_node(int(n), (float(d['y']), float(d['x'])))
        for u, v in G.edges():
            mp.add_edge(int(u), int(v)); mp.add_edge(int(v), int(u))
        _MAP[ds] = mp
    return _MAP[ds]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def match_trace(tr, mc):
    """Returns dict(mm=trace, mmrs=trace, frac_matched, n_breaks)."""
    from leuvenmapmatching.matcher.distance import DistanceMatcher
    ds = tr['ds']
    mp = get_map(ds)
    lat, lon, t = np.asarray(tr['lat']), np.asarray(tr['lon']), np.asarray(tr['t'])
    n = len(lat)
    snapped = {}
    nodes = []
    start, breaks = 0, 0
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(mc.get('timeout_s', 30)))
    try:
        while start < n - 1:
            path = list(zip(lat[start:], lon[start:]))
            m = DistanceMatcher(mp, max_dist=mc['max_dist'], max_dist_init=mc['max_dist'], obs_noise=mc['obs_noise'],
                                obs_noise_ne=mc['obs_noise_ne'], max_lattice_width=mc['lattice_width'],
                                non_emitting_states=mc.get('non_emitting', True), dist_noise=mc.get('dist_noise', 50))
            try:
                _, last = m.match(path)
            except Exception:
                last = -1
            if last < 0 or not m.lattice_best:
                start += 1; breaks += 1
                continue
            for st in m.lattice_best:
                if st.is_emitting():
                    e = st.edge_m
                    pt = e.pi if e.pi is not None else e.p1
                    snapped[start + st.obs] = (float(pt[0]), float(pt[1]))
            try:
                nodes.extend(m.path_pred_onlynodes)
            except Exception:
                pass
            if start + last >= n - 1:
                break
            start = start + last + 2   # skip the point that broke the match
            breaks += 1
    except _Timeout:
        breaks += 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    idx = sorted(snapped)
    mm = dict(lat=np.array([snapped[i][0] for i in idx]), lon=np.array([snapped[i][1] for i in idx]),
              t=t[idx] if idx else np.zeros(0, dtype=np.int64), ds=ds)
    mmrs = resample_route(nodes, mp, lat, lon, t, ds) if nodes else dict(mm)
    return dict(mm=mm, mmrs=mmrs, frac_matched=len(idx) / max(n, 1), n_breaks=breaks)


def resample_route(nodes, mp, lat, lon, t, ds):
    pr = LocalProj.for_dataset(ds)
    nn = [nodes[0]] + [x for a, x in zip(nodes[:-1], nodes[1:]) if x != a]
    ll = np.array([mp.node_coordinates(x) for x in nn])
    x, y = pr.to_xy(ll[:, 0], ll[:, 1])
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0], np.cumsum(seg)])
    ox, oy = pr.to_xy(lat, lon)
    step = float(np.median(np.hypot(np.diff(ox), np.diff(oy)))) if len(lat) > 1 else 50.0
    step = max(step, 10.0)
    if cum[-1] <= 0:
        la, lo = pr.to_ll(x[:1], y[:1])
        return dict(lat=la, lon=lo, t=np.asarray(t[:1]), ds=ds)
    s = np.arange(0, cum[-1] + 1e-9, step)
    rx, ry = np.interp(s, cum, x), np.interp(s, cum, y)
    la, lo = pr.to_ll(rx, ry)
    tt = np.round(np.interp(s, [0, cum[-1]], [t[0], t[-1]])).astype(np.int64)
    return dict(lat=la, lon=lo, t=tt, ds=ds)
