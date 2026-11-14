"""Round 4 (post-audit review, issue V1): null of the reference-calibrated pooled test at 1,000 and 10^4 traces on
UNWATERMARKED generated data, clean and after boundary hopping. Reproduces the independent check of the round-4 verifier
(adapted from its vhopnull.py), with the paper's own machinery: the reference pools of src/rev3/recal3.py
(_calib_setup, same RNG streams as the hop / dilution / hopkeys parts) and r3common.score().

Data: traces 30,000-39,999 of the unwatermarked small-GPT Porto release none__k0__s20 (one 10^4-trace dataset and ten
disjoint 1,000-trace pools); boundary hop = src/rev2/breakatk._hop in mode 'hopfm' (as in recal3 hop / hopkeys).
Keys: 150 fresh keys 7000-7149 (disjoint from keys 0-4 of the releases and 3000-3199 of keycond2), so every test is a
null test. Detectors: the GeoMark-KGW and step-neutral pair sets with their recal3 reference pools ('k1', 'sn').
Per (scheme, data, size): FPR at 1e-2 / 1e-3 of the parametric upper tail and two-sided test and of the calibrated upper
tail and two-sided test, the largest calibrated -log10 p, and the variance of the calibrated t-statistic per level
(1 if the calibration carries over to these data).
Output: results/rev4/calnull4.json. Model-free, CPU only (about 2 min with 10 workers).
Usage (from the project folder): PYTHONPATH=src python src/rev4/calnull4.py"""
import os, sys, json, math, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for p_ in (os.path.join(SRC, 'rev3'), os.path.join(SRC, 'rev2'), SRC):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import recal3 as R3
from r3common import cfg_rev3, load_release, BREF, union, ref_stats, score, ppath, dump_json
from geomark.common import scaled
from geomark.detect import pair_sets, trace_cells
from generate import wm_scheme
import breakatk as BA

OUT4 = ppath('results', 'rev4')
OUTF = os.path.join(OUT4, 'calnull4.json')
KEYS = list(range(7000, 7150))
BLOCK = (30000, 40000)
SIZES = (1000, 10000)
NAMES = {'gm_kgw': 'k1', 'gm_kgw_sn': 'sn'}


def main():
    os.makedirs(OUT4, exist_ok=True)
    cfg = cfg_rev3()
    res = dict(meta=dict(release='gpt_porto_r9_s0/none__k0__s20', block=list(BLOCK), keys=[KEYS[0], KEYS[-1]],
                         n_keys=len(KEYS), sizes=list(SIZES), hop='breakatk._hop hopfm',
                         reference='recal3._calib_setup (same pools as recal3 hop/dil/hopkeys)'))
    t0 = time.time()
    for wm, name in NAMES.items():
        sch = wm_scheme(cfg, wm, 9)
        codec, rp, n_ref = R3._calib_setup(cfg, sch, name, SIZES)
        F = load_release(cfg, 'gpt_porto_r9_s0', 'none__k0__s20')['traces']
        blk = F[BLOCK[0]:BLOCK[1]]
        with Pool(scaled(10, 1), initializer=BA._init, initargs=(sch,)) as pool:
            hop = pool.map(BA._hop, [(t, 'hopfm', i) for i, t in enumerate(blk)], chunksize=50)
        sets = {'clean': [codec.enc(pair_sets(*trace_cells(t, sch), sch)) for t in blk],
                'hop': [codec.enc(pair_sets(*trace_cells(h[0], sch), sch)) for h in hop]}
        disp = float(np.mean([h[1] for h in hop]))
        print(wm, f'mean hop displacement {disp:.1f} m', f'{time.time() - t0:.0f}s', flush=True)
        pools = {v: {m: [union(E[i:i + m]) for i in range(0, len(E) - m + 1, m)] for m in SIZES} for v, E in sets.items()}
        rec = {v: {m: {f: [] for f in ('up', 'two', 'cal_up', 'cal_two', 'tf', 'tm')} for m in SIZES} for v in sets}
        for key in KEYS:
            Gf, Gm = codec.greens(key)
            rs = {m: ref_stats(rp[m], Gf, Gm, sch.gamma, BREF[m], n_ref, m) for m in rp}
            for v in sets:
                for m in SIZES:
                    st = rs[m]
                    for p in pools[v][m]:
                        s = score(p, Gf, Gm, sch.gamma, st)
                        r = rec[v][m]
                        for f in ('up', 'two', 'cal_up', 'cal_two'):
                            r[f].append(s[f])
                        r['tf'].append((s['zf'] - st['muf']) / (st['sdf'] / math.sqrt(st['fpc'])))
                        r['tm'].append((s['zm'] - st['mum']) / (st['sdm'] / math.sqrt(st['fpc'])))
        o = dict(mean_hop_disp_m=disp)
        for v in sets:
            for m in SIZES:
                r = {f: np.array(x) for f, x in rec[v][m].items()}
                e = dict(n_tests=int(len(r['up'])), pools_per_key=len(pools[v][m]))
                for f in ('up', 'two', 'cal_up', 'cal_two'):
                    e[f'{f}_fpr_1e2'] = float((r[f] <= -2).mean())
                    e[f'{f}_fpr_1e3'] = float((r[f] <= -3).mean())
                e['cal_up_max_mlp'] = float(-r['cal_up'].min())
                e['var_t_fine'], e['var_t_mid'] = float(np.var(r['tf'])), float(np.var(r['tm']))
                e['mean_t_fine'], e['mean_t_mid'] = float(np.mean(r['tf'])), float(np.mean(r['tm']))
                o[f'{v}|m{m}'] = e
                print(wm, v, m, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in e.items()}, flush=True)
        res[wm] = o
        dump_json(res, OUTF)
    res['meta']['seconds'] = round(time.time() - t0, 1)
    dump_json(res, OUTF)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
