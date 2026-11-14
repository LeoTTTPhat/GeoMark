"""Download OSM drive networks (Overpass via OSMnx, no login) for the map-matching attack.
Skips regions already saved. Usage: python src/fetch_osm.py [porto] [beijing]"""
import os, sys, time
import osmnx as ox
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'osm')
os.makedirs(OUT, exist_ok=True)
ox.settings.use_cache = True
ox.settings.cache_folder = os.path.join(OUT, 'cache')
ox.settings.requests_timeout = 600
ox.settings.log_console = False
# (west, south, east, north) -- osmnx>=2 bbox order (left, bottom, right, top)
REGIONS = {
    'porto':   (-8.70, 41.10, -8.55, 41.20),   # Porto municipality + core taxi area
    'beijing': (116.25, 39.85, 116.45, 40.03), # central Beijing (GeoLife densest area, Haidian + core)
}
for name in (sys.argv[1:] or REGIONS):
    fn = os.path.join(OUT, f'{name}_drive.graphml')
    if os.path.exists(fn):
        print(name, 'exists, skip'); continue
    t = time.time()
    G = ox.graph_from_bbox(REGIONS[name], network_type='drive', simplify=True)
    ox.save_graphml(G, fn)
    print(name, 'nodes', G.number_of_nodes(), 'edges', G.number_of_edges(), 'sec %.0f' % (time.time() - t), flush=True)
    del G
