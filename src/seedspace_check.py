"""Stage-1 planning statistic for Prop. 5/6: effective seed-space size on real Porto trips (r9 fine, r7 coarse):
number of distinct coarse histories h (k = 0..3 previous distinct coarse cells, run-collapsed) x start-time bins,
and how concentrated the observations are (fraction of scored positions covered by the top-100 seeds).
Writes results/seedspace.json. Usage: python src/seedspace_check.py [n_trips]"""
import os, sys, json, glob, collections
import numpy as np, pandas as pd, h3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(int(sys.argv[1]) if len(sys.argv) > 1 else 20000 * float(os.environ.get('MEMGUARD_SCALE', '1')))
B = (-8.70, 41.10, -8.55, 41.20)
f = (glob.glob(os.path.join(ROOT, 'data', 'raw', 'porto', '**', 'train.csv'), recursive=True) or [os.path.join(ROOT, 'data', 'raw', 'porto', 'train_partial.csv')])[0]
df = pd.read_csv(f, usecols=['TIMESTAMP', 'POLYLINE'], nrows=4 * N)
df = df[df.POLYLINE.str.len() > 40].sample(frac=1, random_state=0)
cnt = {(k, w): collections.Counter() for k in (0, 1, 2, 3) for w in (0, 30, 120)}
n = 0; pos = 0
for ts, pl in zip(df.TIMESTAMP.values, df.POLYLINE.values):
    a = np.array(json.loads(pl))
    if not ((a[:, 0] >= B[0]) & (a[:, 0] <= B[2]) & (a[:, 1] >= B[1]) & (a[:, 1] <= B[3])).all(): continue
    cells = [h3.latlng_to_cell(y, x, 9) for x, y in a]
    seq = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
    hour_min = ((ts % 86400) // 60)
    coarse = []
    for t in range(1, len(seq)):
        p = h3.cell_to_parent(seq[t - 1], 7)
        if not coarse or coarse[-1] != p: coarse.append(p)
        for (k, w), C in cnt.items():
            b = int(hour_min // w) if w else 0
            C[(tuple(coarse[-k:]) if k else (), b)] += 1
        pos += 1
    n += 1
    if n >= N: break
out = {'trips': n, 'scored_positions': pos, 'source': os.path.basename(f), 'by_k_bin': []}
for (k, w), C in sorted(cnt.items()):
    tot = sum(C.values()); top = sum(v for _, v in C.most_common(100))
    out['by_k_bin'].append(dict(k=k, bin_min=w, distinct_seeds=len(C), median_obs_per_seed=float(np.median(list(C.values()))),
                                top100_coverage=top / tot))
    print(f'k={k} bin={w:3d}min: distinct seeds {len(C):6d}, median obs/seed {np.median(list(C.values())):6.0f}, top-100 seeds cover {100*top/tot:5.1f}% of positions')
json.dump(out, open(os.path.join(ROOT, 'results', 'seedspace.json'), 'w'), indent=1)
