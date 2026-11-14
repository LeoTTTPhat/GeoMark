"""E5 calibration of the null: per dataset / resolution / scheme, n_tests (trace, key) null tests on held-out REAL
traces (keys never used for generation, ids >= 100000)
and on UNWATERMARKED generator outputs (a fresh key per test); KS test of randomised p-values, empirical FPR at nominal alphas with exact
binomial CIs (conservative p-values), dataset-level pooled nulls, bin-shift nulls, anytime-valid e-process and SPRT
false-positive rates on streams of real traces. Output: <out_dir>/calibration/<ds>_r<res>_<scheme>_<source>.{json,npz}.
Usage: python src/run_calibration.py --cfg CFG"""
import os, sys, argparse, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, dump_json, load_pickle, Timer, seed_from
from geomark.prf import PRF
from geomark.detect import pair_sets, score_pairs, trace_cells, pooled, detect_cells, uniforms
from geomark.sequential import eprocess_for, scores_from_uniforms, SPRT
from geomark.stats import ks_uniform, empirical_fpr
from geomark.data import real_traces, release_dir
from generate import wm_scheme


def null_block(pairs, sch, n_tests, key0, rng):
    """n_tests null tests cycling over the traces' pair sets, each with a FRESH random key (key0 + i). A fresh key per test
    makes the p-values mutually independent (traces of the same user repeat routes, so p-values of different traces under
    one shared key are dependent, which invalidates the KS test although every marginal is exactly uniform)."""
    lp, lpr, lpf = [], [], []
    n = len(pairs)
    for i in range(n_tests):
        f, m = pairs[i % n]
        prf = PRF(key0 + i)
        a = score_pairs(f, m, sch, prf)
        b = score_pairs(f, m, sch, prf, rng) if sch.score == 'kgw' else a
        lp.append(a['lp']); lpr.append(b['lp']); lpf.append(a['lp_f'])
    return np.array(lp), np.array(lpr), np.array(lpf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ext', action='store_true', help='Stage 4: use cfg[calib_ext] overrides, write results/calibration_ext/')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    cc = dict(cfg['calib'])
    if a.ext:
        cc.update(cfg['calib_ext'])
    od = os.path.join(cfg['out_dir'], 'calibration_ext' if a.ext else 'calibration')
    KO = 10_000_000 if a.ext else 0      # ext run: fresh, disjoint null keys (independent replication of the r9 cells)
    os.makedirs(od, exist_ok=True)
    for ds in cfg['datasets']:
        for res in cc['res']:
            real = real_traces(cfg, ds, 'test', limit=cc['n_traces'], seed=3, min_pts=2)
            for sname in list(cc['schemes']) + (list(cc.get('schemes_r9_extra', [])) if res == 9 else []):
                sch = wm_scheme(cfg, sname, res)
                sources = {'real': real}
                gen = cc['gen_of'].get(f'{ds}_r{res}')
                if gen:
                    f = os.path.join(release_dir(cfg, gen), 'none__k0__s0.pkl')
                    if os.path.exists(f):
                        sources['gen_none'] = load_pickle(f)['traces']
                for src, trs in sources.items():
                    out = os.path.join(od, f'{ds}_r{res}_{sname}_{src}.json')
                    if os.path.exists(out):
                        continue
                    T = Timer()
                    rng = np.random.default_rng(seed_from(ds, res, sname, src, *([KO] if KO else [])))
                    cells = [trace_cells(tr, sch) for tr in trs]
                    pairs = [pair_sets(c, s, sch) for c, s in cells]
                    npairs = np.array([len(f) for f, _ in pairs])
                    n_tests = cc['n_tests'] if src == 'real' else cc.get('n_tests_gen', cc['n_tests'] // 2)
                    lp, lpr, lpf = null_block(pairs, sch, n_tests, KO + 100_000, rng)
                    pv = 10 ** lpr
                    rec = dict(ds=ds, res=res, scheme=sname, source=src, n_traces=len(trs), n_tests=n_tests,
                               median_pairs=float(np.median(npairs)), ks_randomized=ks_uniform(pv),
                               fpr_conservative=empirical_fpr(lp), fpr_randomized=empirical_fpr(lpr),
                               fpr_fine_only=empirical_fpr(lpf), min_lp=float(lp.min()))
                    # dataset-level pooled nulls
                    pl = []
                    for i in range(cc['pooled_tests']):
                        idx = rng.choice(len(pairs), size=min(cc['pooled_m'], len(pairs)), replace=False)
                        pl.append(pooled([pairs[j] for j in idx], sch, PRF(KO + 500_000 + i), rng if sch.score == 'kgw' else None)['lp'])
                    rec['pooled'] = dict(m=cc['pooled_m'], ks_randomized=ks_uniform(10 ** np.array(pl)), fpr=empirical_fpr(pl, (1e-1, 1e-2, 1e-3)))
                    # bin-shift (J=1) nulls on real traces
                    if src == 'real':
                        ls = [detect_cells(c, s, sch, PRF(KO + 700_000 + i), J=1)['lp_shift']
                              for i, (c, s) in zip(range(cc['shift_tests']), (cells * 100)[:cc['shift_tests']])]
                        rec['bin_shift_J1'] = empirical_fpr(ls, (1e-1, 1e-2, 1e-3))
                        # streaming: anytime-valid e-process and SPRT on streams of real traces under a random key
                        seq = {}
                        for alpha in cc['seq_alpha']:
                            cross, spr = 0, 0
                            for r in range(cc['seq_streams']):
                                prf, seen, ep = PRF(KO + 900_000 + r), set(), eprocess_for(sch)
                                sp = SPRT(sch.gamma, min(0.95, sch.gamma + 0.15), alpha, 0.1) if sch.score == 'kgw' else None
                                hit = False
                                for j in rng.choice(len(pairs), size=min(cc['seq_len'], len(pairs)), replace=False):
                                    new = [x for x in pairs[j][0] if x not in seen]; seen.update(new)
                                    if not new:
                                        continue
                                    sc = scores_from_uniforms(uniforms(prf, new, 'fine'), sch)
                                    if not hit and ep.update(sc) >= math.log(1 / alpha):
                                        hit = True
                                    if sp is not None and sp.decision == 0:
                                        sp.update(sc)
                                cross += hit
                                spr += int(sp is not None and sp.decision == 1)
                            seq[f'{alpha:g}'] = dict(eprocess_fpr=cross / cc['seq_streams'], sprt_fpr=spr / cc['seq_streams'] if sch.score == 'kgw' else None,
                                                     streams=cc['seq_streams'], stream_len=cc['seq_len'])
                        rec['sequential'] = seq
                    rec['sec'] = T()
                    sub = rng.choice(len(lpr), size=min(len(lpr), 100_000), replace=False)
                    np.savez_compressed(out[:-5] + '.npz', lp=lp[sub], lp_rand=lpr[sub], npairs=npairs, pooled=np.array(pl))
                    dump_json(rec, out)
                    print(f"{ds} r{res} {sname} {src}: KS p={rec['ks_randomized']['ks_p']:.3f}, FPR@1e-3={rec['fpr_conservative']['0.001']['fpr']:.5f}, "
                          f"pooled KS p={rec['pooled']['ks_randomized']['ks_p']:.3f} ({T():.0f}s)", flush=True)


if __name__ == '__main__':
    main()
