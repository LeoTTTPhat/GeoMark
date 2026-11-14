"""E0 preprocessing: bbox filter, glitch/gap split, resampling, token cap, H3 tokenisation at all resolutions,
splits by taxi (Porto) / user (GeoLife). Writes <data_dir>/{porto,geolife}.pkl and vocab files. Idempotent (skips
datasets already written). Usage: python src/preprocess.py --cfg src/configs/full.yaml [--ds porto]"""
import os, sys, glob, json, argparse, zlib
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, BBOX, p, save_pickle, scaled, slot_of, LocalProj, dump_json, Timer
from geomark.hexgrid import cells_of_points, collapse
from geomark.data import ds_path, get_vocab


def split_of(uid, ds):
    h = zlib.crc32(f'{ds}-{uid}'.encode()) % 10
    return 'train' if h < 8 else ('val' if h == 8 else 'test')


def in_bbox(lat, lon, b):
    return (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])


def segments(lat, lon, t, ds, vmax, gap_s):
    """Split at out-of-bbox points, speed glitches (> vmax m/s) and time gaps (> gap_s)."""
    pr = LocalProj.for_dataset(ds)
    ok = in_bbox(lat, lon, BBOX[ds])
    x, y = pr.to_xy(lat, lon)
    brk = np.zeros(len(lat), dtype=bool)
    if len(lat) > 1:
        d = np.hypot(np.diff(x), np.diff(y)); dt = np.maximum(np.diff(t), 1)
        brk[1:] = (d / dt > vmax) | (np.diff(t) > gap_s)
    out, cur = [], []
    for i in range(len(lat)):
        if not ok[i] or brk[i]:
            if len(cur) > 1: out.append(np.array(cur))
            cur = []
        if ok[i]:
            cur.append(i)
    if len(cur) > 1: out.append(np.array(cur))
    return out


def chunk_tokens(lat, lon, t, res_list, max_tok, min_tok, min_pts, split_res=9):
    """Cut a segment into pieces with <= max_tok collapsed split_res cells; tokenise each piece at all resolutions."""
    cells9 = cells_of_points(lat, lon, split_res)
    col, first = collapse(cells9)
    pieces = []
    for a in range(0, len(col), max_tok):
        s = first[a]; e = first[a + max_tok] if a + max_tok < len(col) else len(lat)
        if e - s < min_pts or min(max_tok, len(col) - a) < min_tok:
            continue
        la, lo, tt = lat[s:e], lon[s:e], t[s:e]
        tok, ppc = {}, {}
        for r in res_list:
            c, f = collapse(cells9[s:e] if r == split_res else cells_of_points(la, lo, r))
            tok[r] = c
            ppc[r] = np.diff(np.append(f, len(la)))
        pieces.append((la, lo, tt, tok, ppc))
    return pieces


def resample(t, min_dt):
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= min_dt:
            keep.append(i)
    return np.array(keep)


# ---------------- Porto ----------------
def porto(cfg, dc):
    T = Timer()
    f = p('data/raw/porto/train.csv')
    res_list = cfg['data']['res']
    traces, tok, ppc = [], {r: [] for r in res_list}, {r: [] for r in res_list}
    frac = dc.get('frac', 0.1)
    nrows_read = 0
    for chunk in pd.read_csv(f, usecols=['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE'],
                             chunksize=scaled(50000, 5000), nrows=dc.get('max_rows')):
        nrows_read += len(chunk)
        keep = chunk.TRIP_ID.astype(str).map(lambda s: zlib.crc32(s.encode()) % 10000 < frac * 10000)
        chunk = chunk[keep & (~chunk.MISSING_DATA.astype(bool)) & (chunk.POLYLINE.str.len() > 2 + 20 * dc['min_pts'])]
        for tid, taxi, ts, pl in zip(chunk.TRIP_ID.values, chunk.TAXI_ID.values, chunk.TIMESTAMP.values, chunk.POLYLINE.values):
            a = np.asarray(json.loads(pl), dtype=float)
            if len(a) < dc['min_pts']:
                continue
            lat, lon = a[:, 1], a[:, 0]
            t = int(ts) + dc['dt'] * np.arange(len(a), dtype=np.int64)
            for seg in segments(lat, lon, t, 'porto', vmax=40, gap_s=300):
                for la, lo, tt, tk, pc in chunk_tokens(lat[seg], lon[seg], t[seg], res_list, cfg['data']['max_tok'], cfg['data']['min_tok'], dc['min_pts']):
                    traces.append(dict(uid=int(taxi), tid=str(tid), split=split_of(taxi, 'porto'), lat=la, lon=lo, t=tt,
                                       slot=slot_of(tt[0], 'porto'), ds='porto'))
                    for r in res_list:
                        tok[r].append(tk[r]); ppc[r].append(pc[r])
        print(f'  porto: read {nrows_read} rows, {len(traces)} traces ({T():.0f}s)', flush=True)
        if len(traces) >= dc['max_traces']:
            break
    return traces, tok, ppc


# ---------------- GeoLife ----------------
def _geolife_file(args):
    fn, uid, dc, res_list, max_tok, min_tok = args
    try:
        df = pd.read_csv(fn, skiprows=6, header=None, usecols=[0, 1, 4], names=['lat', 'lon', 'days'])
    except Exception:
        return []
    lat, lon = df.lat.values.astype(float), df.lon.values.astype(float)
    t = np.round((df.days.values - 25569.0) * 86400).astype(np.int64)  # Excel serial date -> unix
    if len(t) < dc['min_pts'] or not in_bbox(lat, lon, BBOX['geolife']).any():
        return []
    o = np.argsort(t, kind='stable'); lat, lon, t = lat[o], lon[o], t[o]
    out = []
    for seg in segments(lat, lon, t, 'geolife', vmax=50, gap_s=60 * dc['gap_min']):
        k = resample(t[seg], dc['dt'])
        seg = seg[k]
        if len(seg) < dc['min_pts']:
            continue
        for la, lo, tt, tk, pc in chunk_tokens(lat[seg], lon[seg], t[seg], res_list, max_tok, min_tok, dc['min_pts']):
            out.append((dict(uid=int(uid), tid=os.path.basename(fn), split=split_of(uid, 'geolife'), lat=la, lon=lo, t=tt,
                             slot=slot_of(tt[0], 'geolife'), ds='geolife'), tk, pc))
    return out


def geolife(cfg, dc):
    files = sorted(glob.glob(p('data/raw/geolife/Geolife Trajectories 1.3/Data/*/Trajectory/*.plt')))
    users = sorted({f.split(os.sep)[-3] for f in files})
    if dc.get('max_users'):
        users = set(users[:dc['max_users']]); files = [f for f in files if f.split(os.sep)[-3] in users]
    res_list = cfg['data']['res']
    args = [(f, int(f.split(os.sep)[-3]), dc, res_list, cfg['data']['max_tok'], cfg['data']['min_tok']) for f in files]
    traces, tok, ppc = [], {r: [] for r in res_list}, {r: [] for r in res_list}
    with Pool(scaled(8, 1)) as pool:
        for i, res in enumerate(pool.imap(_geolife_file, args, chunksize=16)):
            for tr, tk, pc in res:
                traces.append(tr)
                for r in res_list:
                    tok[r].append(tk[r]); ppc[r].append(pc[r])
            if i % 2000 == 0:
                print(f'  geolife: {i}/{len(files)} files, {len(traces)} traces', flush=True)
            if dc.get('max_traces') and len(traces) >= dc['max_traces']:
                break
    return traces, tok, ppc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ds', default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    stats = {}
    for ds in ([a.ds] if a.ds else cfg['datasets']):
        out = ds_path(cfg, ds)
        if os.path.exists(out):
            print(f'{ds}: exists, skipping ({out})'); continue
        T = Timer()
        dc = cfg['data'][ds]
        traces, tok, ppc = (porto if ds == 'porto' else geolife)(cfg, dc)
        if dc.get('max_traces') and len(traces) > dc['max_traces']:
            traces = traces[:dc['max_traces']]; tok = {r: v[:dc['max_traces']] for r, v in tok.items()}
        ppc_s = {r: np.concatenate([x for tr, x in zip(traces, ppc[r]) if tr['split'] == 'train'][:20000]) for r in ppc}
        D = dict(traces=traces, tok=tok, ppc=ppc_s, dt=dc['dt'], ds=ds,
                 day0=int(min(tr['t'][0] for tr in traces[:5000]) // 86400 * 86400))
        save_pickle(D, out)
        for r in cfg['data']['res']:
            get_vocab(cfg, ds, r)
        sp = {s: sum(tr['split'] == s for tr in traces) for s in ('train', 'val', 'test')}
        L = {r: float(np.median([len(x) for x in tok[r]])) for r in tok}
        stats[ds] = dict(traces=len(traces), splits=sp, median_tok=L, median_pts=float(np.median([len(tr['lat']) for tr in traces])),
                         vocab={r: int(len(np.load(os.path.join(cfg['data_dir'], f'vocab_{ds}_r{r}.npy')))) for r in cfg['data']['res']},
                         sec=T())
        print(ds, stats[ds], flush=True)
        dump_json(stats[ds], os.path.join(cfg['out_dir'], 'prep', f'{ds}_stats.json'))


if __name__ == '__main__':
    main()
