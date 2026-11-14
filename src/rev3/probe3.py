"""Stage 10 probe (round-3 audit): verify the reviewers' recomputation of the scrubbed releases (fine / medium z-scores,
lower-tail p-values) and time the recolouring step. Read-only on results/rev2/steal2/*.pkl. Model-free.
Output: results/rev3/probe3.json. Usage: python src/rev3/probe3.py"""
import os, sys, time, json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rev2'))
from r2common import cfg_rev2, load_release, OUT2, load_pickle, LN10, ppath, dump_json
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, uniforms
from generate import wm_scheme

CORPORA = [('k1', 'gm_kgw'), ('k2', 'gm_kgw_k2'), ('k0', 'a_k0'), ('exb', 'exact_kgw_bin'), ('eosn', 'gm_kgw_eosn'), ('sn', 'gm_kgw_sn')]


def zt(pairs, prf, dom, gamma):
    u = uniforms(prf, pairs, dom)
    n = len(u); T = int((u < gamma).sum())
    if n == 0:
        return dict(n=0)
    return dict(n=n, frac=T / n, z=(T - n * gamma) / np.sqrt(n * gamma * (1 - gamma)),
                up=float(stats.binom.logsf(T - 1, n, gamma) / LN10), low=float(stats.binom.logcdf(T, n, gamma) / LN10))


def pool(pl, a, b):
    F, M = {}, {}
    for f, m in pl[a:b]:
        F.update(dict.fromkeys(f)); M.update(dict.fromkeys(m))
    return list(F), list(M)


def main():
    cfg = cfg_rev2()
    prf = PRF(0)
    out = {}
    for name, wm in CORPORA:
        sch = wm_scheme(cfg, wm, 9)
        f = os.path.join(OUT2, 'steal2', f'{name}_scrub_r1.0.pkl')
        scr = load_pickle(f)
        pl = [pair_sets(*trace_cells(t, sch), sch) for t in scr]
        rec = {}
        for m in (100, 1000, 10000):
            n_ = 2000 if m == 100 else len(pl)
            res = []
            for i in range(0, n_ - m + 1, m):
                F, M = pool(pl, i, i + m)
                rf = zt(F, prf, 'fine', sch.gamma)
                rm = zt(M, prf, 'mid', sch.gamma) if sch.uses_mid else dict(n=0)
                res.append((rf, rm))
            zf = [r[0]['z'] for r in res]
            zm = [r[1]['z'] for r in res if r[1].get('n')]
            lowf = [r[0]['low'] for r in res]
            rec[f'm{m}'] = dict(n_pools=len(res), zf_median=float(np.median(zf)), zf_min=float(np.min(zf)), zf_max=float(np.max(zf)),
                                zm_median=float(np.median(zm)) if zm else None,
                                fine_low_mlp_median=float(-np.median(lowf)),
                                frac_fine_low_le_m3_bonf2=float(np.mean(np.array(lowf) + np.log10(2) <= -3)),
                                fine_frac_green=float(np.median([r[0]['frac'] for r in res])), n_f_median=float(np.median([r[0]['n'] for r in res])))
        out[name] = rec
        print(name, {k: (round(v['zf_median'], 1), round(v['zm_median'], 1) if v['zm_median'] is not None else None,
                         round(v['fine_low_mlp_median'], 1), v['frac_fine_low_le_m3_bonf2']) for k, v in rec.items()}, flush=True)
    # timing of the recolouring step (k1, 200 traces)
    from geomark.posthoc import recolor_trace
    from run_steal import Estimate
    from geomark.data import real_traces
    sch = wm_scheme(cfg, 'gm_kgw', 9)
    R = load_release(cfg, 'gpt_porto_r9_s0', 'gm_kgw__k0__s10')
    RA = R['traces'][:10000]
    t0 = time.time()
    real = real_traces(cfg, 'porto', 'train', limit=80000, seed=7)
    prior = [pair_sets(*trace_cells(t, sch), sch) for t in real]
    t1 = time.time()
    est = Estimate(sch, [pair_sets(t['cells'], t['slot'], sch) for t in RA], prior)
    t2 = time.time()
    rng = np.random.default_rng(0)
    sc = est.scorer()
    xs = [recolor_trace(t, sch, sc, 'red', 1.0, rng) for t in RA[:300]]
    t3 = time.time()
    # estimated-green fraction (fine) of de-duplicated pairs: release, real test, full scrub
    def efrac(trs, from_points=True):
        g = []
        for t in trs:
            f, _ = pair_sets(*trace_cells(t, sch), sch) if from_points else pair_sets(t['cells'], t['slot'], sch)
            for s, c in f:
                v = est.green_fine(s, c)
                if not np.isnan(v):
                    g.append(v)
        return float(np.mean(g)), len(g)
    rt = real_traces(cfg, 'porto', 'test', limit=1000, seed=8)
    scr = load_pickle(os.path.join(OUT2, 'steal2', 'k1_scrub_r1.0.pkl'))
    out['timing'] = dict(prior_pairs_s=t1 - t0, estimate_s=t2 - t1, recolor_per_trace_ms=1000 * (t3 - t2) / 300)
    out['k1_est_green'] = dict(release=efrac(RA[:2000], False), real_test=efrac(rt), full_scrub=efrac(scr[:2000]))
    print(out['timing'], out['k1_est_green'], flush=True)
    dump_json(out, ppath('results', 'rev3', 'probe3.json'))


if __name__ == '__main__':
    main()
