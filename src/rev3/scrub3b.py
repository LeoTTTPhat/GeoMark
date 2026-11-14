"""Stage 10 (round-3 audit), follow-up to scrub3.py: an ORACLE-RATE scrubber, i.e. the strongest member of the
rate-based scrubbing family. The licensee recolours towards its estimated-red cells (same estimate and recolouring as
scrub3.py) at a single rate r, and r is chosen with knowledge of the TRUE key so as to minimise the two-sided pooled
evidence of the whole 10^4-trace release (golden-section search over r in [0, 1], 10 evaluations, common random
numbers). Only one scalar is tuned with the key, so the result bounds what any scrubber in this family can achieve; it
separates residual evidence that comes from over-shooting (anti-green data, visible in the lower tail) from evidence the
scrub cannot remove. Same configurations, keys, tests and output format as scrub3.py (variant 'orc').
Output: results/rev3/scrub3/<config>_k<key>_orc.json (+ .pkl); merged into results/rev3/scrub3.json by scrub3.aggregate().
Model-free. Usage: python src/rev3/scrub3b.py"""
import os, sys, json, math
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r3common import (cfg_rev3, load_release, load_pickle, save_pickle, ppath, dump_json, BREF, PairCodec, union,
                      ref_pools, ref_stats, score)
from r2common import real_cells as real_cells_slots
from geomark.common import seed_from, Timer, scaled
from geomark.detect import pair_sets, trace_cells
from geomark.utility import utility
from geomark.data import real_traces
from geomark.hexgrid import tokenize
from run_steal import Estimate
from run_utility import real_cells
from rev1.utility import fixed_metrics
from generate import wm_scheme
import scrub3 as S3

GR = (math.sqrt(5) - 1) / 2


def run_config(args):
    name, wm, keys = args
    cfg = cfg_rev3()
    sch = wm_scheme(cfg, wm, 9)
    T = Timer()
    todo = [k for k in keys if not os.path.exists(os.path.join(S3.D3, f'{name}_k{k}_orc.json'))]
    if not todo:
        return name
    codec = PairCodec(sch)
    rc, rs = real_cells_slots(cfg, 'porto', 9, 'train')
    ref = [codec.enc(pair_sets(c, s, sch)) for c, s in zip(rc, rs)]
    ref = ref[len(ref) // 2:]
    n_ref = len(ref)
    rpools = {m: ref_pools(ref, m, BREF[m], np.random.default_rng(seed_from('rev3-ref', name, m))) for m in S3.SIZES}
    del ref
    prior = [pair_sets(*trace_cells(t, sch), sch) for t in real_traces(cfg, 'porto', 'train', limit=80000, seed=7)]
    real_u = real_cells(cfg, 'porto', 9, 'test', 5000, 0)
    for key in todo:
        rid = f'{wm}__k0__s10' if (key == 0 and name != 'exb2r') else f'{wm}__k{key}__s10'
        R = load_release(cfg, S3.GEN, rid)
        RA = R['traces'][:10000]
        del R
        est = Estimate(sch, [pair_sets(t['cells'], t['slot'], sch) for t in RA], prior)
        sc = est.scorer()
        seed = seed_from('rev3-scrub-orc', name, key)
        cache, best = {}, {}

        def evaluate(r):
            r = round(r, 4)
            if r not in cache:
                X = S3.scrub(RA, sch, sc, r, seed) if r > 0 else RA
                E = [codec.enc(pair_sets(*trace_cells(t, sch), sch)) if (t is not None and len(t['lat'])) else
                     (np.zeros(0, np.int64), np.zeros(0, np.int64)) for t in X]
                Gf, Gm = codec.greens(key)
                cache[r] = score(union(E), Gf, Gm, sch.gamma, None)['two']
                if not best or cache[r] > best['v']:
                    best.update(v=cache[r], r=r, X=X, E=E)
            return cache[r]
        # golden-section search for the r that MAXIMISES the two-sided log10 p (i.e. minimises evidence)
        a, b = 0.0, 1.0
        c, d = b - GR * (b - a), a + GR * (b - a)
        fc, fd = evaluate(c), evaluate(d)
        for _ in range(8):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - GR * (b - a); fc = evaluate(c)
            else:
                a, c, fc = c, d, fd
                d = a + GR * (b - a); fd = evaluate(d)
        f1 = evaluate(1.0)
        r_best, X, E = best['r'], best['X'], best['E']
        Gf, Gm = codec.greens(key)
        rstats = {m: ref_stats(rpools[m], Gf, Gm, sch.gamma, BREF[m], n_ref, m) for m in S3.SIZES}
        rec = dict(name=name, wm=wm, key=key, rid=rid, orc_rate=r_best, search=sorted(cache.items()),
                   variant={})
        v = rec['variant']
        v['frac_changed'] = float(np.mean([t.get('frac_changed', 0.0) for t in X])) if r_best > 0 else 0.0
        for m in S3.SIZES:
            pools = [union(E[i:i + m]) for i in range(0, len(E) - m + 1, m)]
            v[f'm{m}'] = [score(p, Gf, Gm, sch.gamma, rstats[m]) for p in pools]
        cells = [tokenize(t['lat'], t['lon'], 9) for t in X[:5000] if t is not None]
        u = utility(cells, real_u, 'porto', 9, None)
        u.update(fixed_metrics(cells, real_u, 'porto', None, None))
        v['utility'] = {k: x for k, x in u.items() if isinstance(x, (int, float))}
        dump_json(rec, os.path.join(S3.D3, f'{name}_k{key}_orc.json'))
        print(f'[{name} k{key}] oracle rate {r_best:.3f}: whole two-sided {-best["v"]:.1f} (rate 1: {-f1:.1f}); '
              f'm100 two median {-np.median([p["two"] for p in v["m100"]]):.1f}; cal_two whole {-v["m10000"][0]["cal_two"]:.1f} ({T():.0f}s)', flush=True)
    return name


def main():
    jobs = [(n, w, [0] if n == 'eosn' else S3.KEYS) for n, w in S3.CONFIGS]
    with Pool(scaled(len(jobs), 1)) as pool:
        for n in pool.imap_unordered(run_config, jobs):
            print('done', n, flush=True)
    S3.aggregate()


if __name__ == '__main__':
    main()
