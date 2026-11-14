"""Stage 7: cryptographic PRF check (reviewers: the per-cell SplitMix64 mix is not a PRF; quantify the cost of a
cryptographic per-pair PRF and re-check calibration with it).

geomark/prf.py selects a per-pair keyed BLAKE2b-64 PRF for key ids >= CRYPTO_KEY0. This script (1) runs 10^5 per-trace
null tests on real Porto and GeoLife test traces with a fresh cryptographic key per test (randomised p-values; KS, FPR
at 1e-2 and 1e-3) and 2x10^4 pooled 100-trace null tests, for GeoMark-KGW-2R and Gumbel-H; (2) times detection of
2,000 traces with the fast and the cryptographic PRF. End-to-end detection with the cryptographic PRF is in the
rev1_crypto releases (evaluated by evaluate.py). Output: results/rev1/crypto_calib.json. Usage: python src/rev1/crypto_calib.py"""
import os, sys, time, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, jdump, real_cells, load_release
from geomark.prf import PRF, CRYPTO_KEY0
from geomark.detect import pair_sets, score_pairs, pooled, detect_cells
from generate import wm_scheme


def main():
    cfg = cfg_rev1()
    out = {}
    rng = np.random.default_rng(0)
    for ds in ('porto', 'geolife'):
        cells, slots = real_cells(cfg, ds, 9, 'test')
        for wm in ('gm_kgw', 'gm_gumbel'):
            sch = wm_scheme(cfg, wm, 9)
            pairs = [pair_sets(c, s, sch) for c, s in zip(cells, slots)]
            n = 100000
            base = CRYPTO_KEY0 + 1_000_000 + (0 if ds == 'porto' else 2_000_000) + (0 if wm == 'gm_kgw' else 500_000)
            lps = np.array([score_pairs(*pairs[i % len(pairs)], sch, PRF(base + i), rng)['lp'] for i in range(n)])
            pv = 10 ** lps
            pl = []
            for i in range(20000):
                idx = rng.choice(len(pairs), 100, replace=False)
                pl.append(pooled([pairs[j] for j in idx], sch, PRF(base + 200000 + i))['lp'])
            pl = np.array(pl)
            out[f'{ds}_r9|{wm}'] = dict(n=n, ks_p=float(stats.kstest(pv, 'uniform').pvalue), fpr_1e2=float((pv <= 1e-2).mean()),
                                        fpr_1e3=float((pv <= 1e-3).mean()), pooled_n=len(pl),
                                        pooled_fpr_1e3=float((pl <= -3).mean()), pooled_fpr_1e2=float((pl <= -2).mean()))
            print(ds, wm, out[f'{ds}_r9|{wm}'], flush=True)
    # timing
    R = load_release(cfg, 'gpt_porto_r9_s0', 'gm_kgw__k0__s0')['traces'][:2000]
    sch = wm_scheme(cfg, 'gm_kgw', 9)
    for name, key in (('fast', 0), ('crypto', CRYPTO_KEY0)):
        prf = PRF(key)
        t = time.time()
        for tr in R:
            detect_cells(tr['cells'], tr['slot'], sch, prf)
        dt = time.time() - t
        out[f'timing_{name}'] = dict(traces=len(R), sec=dt, traces_per_s=len(R) / dt)
        print(name, out[f'timing_{name}'], flush=True)
    jdump(out, 'crypto_calib.json')


if __name__ == '__main__':
    main()
