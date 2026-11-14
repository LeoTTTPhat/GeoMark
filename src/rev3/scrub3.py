"""Stage 10 (round-3 audit): scrubbing by a licensee, re-run over FIVE keys per seeding configuration and scored with
BOTH tails and with the reference-calibrated key-conditional test (reviewers: the 'coarse context resists scrubbing'
claim rested on a one-sided test, because the stealer's scrub over-corrects into red cells and leaves the data anti-green,
i.e. still key-dependent; every scrub result used one key; release-size results were single parametric tests).

Per configuration (GeoMark-KGW k=1, bin only k=0, exact+bin 1R, exact+bin 2R, step-neutral, k=2; EOS-neutral key 0 only)
and key K = 0..4, release R_A = 10^4 traces (key 0: first 10^4 traces of the rev2 stealing corpus; keys 1-4 and exact+bin 2R:
src/configs/rev3.yaml). The licensee estimates the green lists from R_A against 80,000 real training traces
(run_steal.Estimate, as in src/rev2/steal2.py) and releases:
  full : every cell recoloured towards estimated-red cells (rate 1; = the rev2 scrub; key 0 reuses results/rev2/steal2);
  cal  : CALIBRATED scrub: recolouring at the smallest rate r at which the estimated-green share (fine level, de-duplicated
         pairs, estimable pairs only) of the scrubbed release equals the share the same estimate gives to 1,000 held-out
         real test traces, i.e. the licensee stops when its own estimate scores the release like real data (bisection on
         2,000 traces, 8 steps, common random numbers; then applied to all 10^4 traces);
  sel10: selection, the 10% of R_A with the lowest estimated green share (no trace modified).
Tests on disjoint 100-, 1,000- and 10,000-trace pools: parametric upper / lower / two-sided, and reference-calibrated
upper / two-sided (src/rev3/r3common.py). Utility: bigram JSD etc. on the first 5,000 traces.
Output: results/rev3/scrub3.json (+ per-config/key checkpoints and scrubbed releases in results/rev3/scrub3/).
Model-free. Usage: python src/rev3/scrub3.py"""
import os, sys, math, json
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r3common import (cfg_rev3, load_release, load_pickle, save_pickle, ppath, dump_json, OUT3, BREF, PairCodec, union,
                      ref_pools, ref_stats, score, summ, wilson)
from r2common import real_cells as real_cells_slots
from geomark.common import seed_from, Timer, scaled
from geomark.detect import pair_sets, trace_cells
from geomark.posthoc import recolor_trace
from geomark.utility import utility
from geomark.data import real_traces
from geomark.hexgrid import tokenize
from run_steal import Estimate
from run_utility import real_cells
from rev1.utility import fixed_metrics
from generate import wm_scheme

SMOKE = os.environ.get('R3_SMOKE') == '1'                       # quick test: k1 / key 0 / 2,000 traces, separate folder
D3 = os.path.join(OUT3, 'scrub3_smoke' if SMOKE else 'scrub3')
CONFIGS = [('k1', 'gm_kgw'), ('k0', 'a_k0'), ('exb', 'exact_kgw_bin'), ('exb2r', 'exact_bin_2r'), ('sn', 'gm_kgw_sn'),
           ('k2', 'gm_kgw_k2'), ('eosn', 'gm_kgw_eosn')]
KEYS = [0, 1, 2, 3, 4]
GEN = 'gpt_porto_r9_s0'
SIZES = (100, 1000, 10000)


def efrac(est, pls):
    """Estimated-green share of the distinct estimable fine pairs of a set of traces."""
    F = {}
    for f, _ in pls:
        F.update(dict.fromkeys(f))
    g = [est.green_fine(s, c) for s, c in F]
    g = np.array([x for x in g if not np.isnan(x)])
    return float(g.mean()) if len(g) else float('nan')


def scrub(trs, sch, sc, rate, seed_key):
    rng = np.random.default_rng(seed_key)
    out = []
    for t in trs:
        x = recolor_trace(t, sch, sc, 'red', rate, rng)
        out.append(x if x is not None else dict(t, frac_changed=0.0))
    return out


def run_config(args):
    name, wm, keys = args
    cfg = cfg_rev3()
    sch = wm_scheme(cfg, wm, 9)
    T = Timer()
    todo = [k for k in keys if not os.path.exists(os.path.join(D3, f'{name}_k{k}.json'))]
    if not todo:
        return name
    codec = PairCodec(sch)
    rc, rs = real_cells_slots(cfg, 'porto', 9, 'train')
    ref = [codec.enc(pair_sets(c, s, sch)) for c, s in zip(rc, rs)]
    ref = ref[len(ref) // 2:]                                     # as src/rev2/keycond2.py (second half of the split)
    n_ref = len(ref)
    rpools = {m: ref_pools(ref, m, BREF[m], np.random.default_rng(seed_from('rev3-ref', name, m))) for m in SIZES}
    del ref
    prior = [pair_sets(*trace_cells(t, sch), sch) for t in real_traces(cfg, 'porto', 'train', limit=80000, seed=7)]
    rtest = [pair_sets(*trace_cells(t, sch), sch) for t in real_traces(cfg, 'porto', 'test', limit=1000, seed=8)]
    real_u = real_cells(cfg, 'porto', 9, 'test', 5000, 0)
    print(f'[{name}] reference {n_ref} traces, prior {len(prior)} ({T():.0f}s)', flush=True)
    for key in todo:
        rid = f'{wm}__k0__s10' if (key == 0 and name != 'exb2r') else f'{wm}__k{key}__s10'
        R = load_release(cfg, GEN, rid)
        if R is None:
            print(f'[{name}] missing release {rid}', flush=True)
            continue
        RA = R['traces'][:2000 if SMOKE else 10000]
        del R
        wm_pairs = [pair_sets(t['cells'], t['slot'], sch) for t in RA]
        est = Estimate(sch, wm_pairs, prior)
        sc = est.scorer()
        target = efrac(est, rtest)
        rec = dict(name=name, wm=wm, key=key, rid=rid, n_release=len(RA), target_est_green_real=target,
                   est_green_release=efrac(est, wm_pairs), variants={})
        # --- variants ---
        V = {'clean': RA}
        f0 = ppath('results', 'rev2', 'steal2', f'{name}_scrub_r1.0.pkl')
        ff = os.path.join(D3, f'{name}_k{key}_full.pkl')
        if key == 0 and os.path.exists(f0) and not SMOKE:
            V['full'] = load_pickle(f0); rec['full_source'] = 'results/rev2/steal2'
        elif os.path.exists(ff):
            V['full'] = load_pickle(ff)
        else:
            V['full'] = scrub(RA, sch, sc, 1.0, seed_from('rev3-scrub-full', name, key)); save_pickle(V['full'], ff)
        fc = os.path.join(D3, f'{name}_k{key}_cal.pkl')
        if os.path.exists(fc):
            V['cal'], rec['cal_rate'], rec['cal_trace'] = load_pickle(fc)
        else:
            sub = RA[:2000]
            cseed = seed_from('rev3-scrub-cal', name, key)
            fr = lambda r: efrac(est, [pair_sets(*trace_cells(t, sch), sch) for t in scrub(sub, sch, sc, r, cseed)])
            trace = []
            f1 = fr(1.0); trace.append((1.0, f1))
            if f1 >= target:
                r_ = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    fm = fr(mid); trace.append((mid, fm))
                    if fm > target:
                        lo = mid
                    else:
                        hi = mid
                r_ = 0.5 * (lo + hi)
            X = scrub(RA, sch, sc, r_, cseed)
            V['cal'], rec['cal_rate'], rec['cal_trace'] = X, r_, trace
            save_pickle((X, r_, trace), fc)
        frac = []
        for f, _ in wm_pairs:
            g = [est.green_fine(s, c) for s, c in f]
            g = [x for x in g if not np.isnan(x)]
            frac.append(np.mean(g) if g else 0.5)
        order = np.argsort(np.array(frac), kind='stable')
        V['sel10'] = [RA[i] for i in sorted(order[:int(0.1 * len(order))].tolist())]
        print(f'[{name} k{key}] target {target:.3f}, release {rec["est_green_release"]:.3f}, cal rate {rec["cal_rate"]:.3f} ({T():.0f}s)', flush=True)
        # --- encode, then score under the key ---
        enc = {}
        for vn, X in V.items():
            pls = [pair_sets(*trace_cells(t, sch), sch) if (t is not None and len(t['lat'])) else ([], []) for t in X]
            if vn != 'clean':
                rec['variants'].setdefault(vn, {})['est_green'] = efrac(est, pls)
            enc[vn] = [codec.enc(p) for p in pls]
        Gf, Gm = codec.greens(key)
        rstats = {m: ref_stats(rpools[m], Gf, Gm, sch.gamma, BREF[m], n_ref, m) for m in SIZES}
        for vn, E in enc.items():
            r = rec['variants'].setdefault(vn, {})
            X = V[vn]
            r['frac_changed'] = float(np.mean([t.get('frac_changed', 0.0) for t in X])) if vn in ('full', 'cal') else 0.0
            for m in SIZES:
                if len(E) < m:
                    continue
                pools = [union(E[i:i + m]) for i in range(0, len(E) - m + 1, m)]
                r[f'm{m}'] = [score(p, Gf, Gm, sch.gamma, rstats[m]) for p in pools]
            cells = [tokenize(t['lat'], t['lon'], 9) for t in X[:5000] if t is not None]
            u = utility(cells, real_u, 'porto', 9, None)
            u.update(fixed_metrics(cells, real_u, 'porto', None, None))
            r['utility'] = {k: v for k, v in u.items() if isinstance(v, (int, float))}
        rec['ref_stats'] = {str(m): v for m, v in rstats.items()}
        dump_json(rec, os.path.join(D3, f'{name}_k{key}.json'))
        s = lambda vn, m, f: (summ(rec['variants'][vn].get(f'm{m}', []), f) or {}).get('median_mlp', float('nan'))
        print(f'[{name} k{key}] full up/two/cal_two m100 {s("full",100,"up"):.1f}/{s("full",100,"two"):.1f}/{s("full",100,"cal_two"):.1f} '
              f'm10000 {s("full",10000,"up"):.1f}/{s("full",10000,"two"):.1f}/{s("full",10000,"cal_two"):.1f}; cal m100 {s("cal",100,"two"):.1f} '
              f'm10000 {s("cal",10000,"two"):.1f}/{s("cal",10000,"cal_two"):.1f} ({T():.0f}s)', flush=True)
    return name


def aggregate():
    out = dict(configs={})
    for name, wm in CONFIGS:
        recs = []
        for k in KEYS:
            f = os.path.join(D3, f'{name}_k{k}.json')
            if os.path.exists(f):
                r_ = json.load(open(f))
                fo = os.path.join(D3, f'{name}_k{k}_orc.json')          # oracle-rate scrub (scrub3b.py)
                if os.path.exists(fo):
                    o = json.load(open(fo))
                    r_['variants']['orc'] = o['variant']; r_['orc_rate'] = o['orc_rate']
                recs.append(r_)
        if not recs:
            continue
        agg = dict(wm=wm, keys=[r['key'] for r in recs], cal_rate=[r['cal_rate'] for r in recs],
                   target_est_green_real=[r['target_est_green_real'] for r in recs],
                   est_green_release=[r['est_green_release'] for r in recs], orc_rate=[r.get('orc_rate') for r in recs],
                   cal_f_rate1=[(r.get('cal_trace') or [[None, None]])[0][1] for r in recs], variants={})
        for vn in ('clean', 'full', 'cal', 'orc', 'sel10'):
            a = {}
            for m in ('m100', 'm1000', 'm10000', 'whole'):
                allp = [p for r in recs for p in r['variants'].get(vn, {}).get(m, [])]
                if not allp:
                    continue
                e = {}
                for f in ('up', 'low', 'two', 'cal_up', 'cal_two'):
                    s = summ(allp, f)
                    if s:
                        e[f] = s
                        # per key: median -log10 p and detected share
                        e[f]['per_key_median'] = [float(-np.median([p[f] for p in r['variants'][vn][m]])) for r in recs if r['variants'].get(vn, {}).get(m)]
                        e[f]['per_key_frac'] = [float(np.mean([p[f] <= -3 for p in r['variants'][vn][m]])) for r in recs if r['variants'].get(vn, {}).get(m)]
                e['zf_median'] = float(np.median([p['zf'] for p in allp])); e['zm_median'] = float(np.median([p['zm'] for p in allp]))
                e['zf_per_key'] = [float(np.median([p['zf'] for p in r['variants'][vn][m]])) for r in recs if r['variants'].get(vn, {}).get(m)]
                e['zm_per_key'] = [float(np.median([p['zm'] for p in r['variants'][vn][m]])) for r in recs if r['variants'].get(vn, {}).get(m)]
                a[m] = e
            ut = [r['variants'].get(vn, {}).get('utility', {}) for r in recs]
            a['utility'] = {k: float(np.mean([u[k] for u in ut if k in u])) for k in ('jsd_bigram', 'jsd_length_ovf', 'jsd_gyration_ovf',
                                                                                      'mean_length_km_gen') if any(k in u for u in ut)}
            a['frac_changed'] = float(np.mean([r['variants'].get(vn, {}).get('frac_changed', 0.0) for r in recs]))
            a['est_green'] = [r['variants'].get(vn, {}).get('est_green') for r in recs]
            if any(vn in r['variants'] for r in recs):
                agg['variants'][vn] = a
        out['configs'][name] = agg
    fo = os.path.join(OUT3, 'scrub3_smoke.json' if SMOKE else 'scrub3.json')
    dump_json(out, fo)
    print('wrote', fo, flush=True)


def main():
    os.makedirs(D3, exist_ok=True)
    jobs = [(n, w, [0] if n == 'eosn' else KEYS) for n, w in CONFIGS]
    if SMOKE:
        jobs = [('k1', 'gm_kgw', [0])]
    with Pool(scaled(len(jobs), 1)) as pool:
        for n in pool.imap_unordered(run_config, jobs):
            print('done', n, flush=True)
    aggregate()


if __name__ == '__main__':
    main()
