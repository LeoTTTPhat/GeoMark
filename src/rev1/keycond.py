"""Stage 7: key-CONDITIONAL calibration of the pooled (dataset-level) test (reviewers: the headline pooled test was
calibrated only marginally over keys; a single owner uses one key, and pools of one city share popular pairs).

For 200 fixed keys (3000..3199), every disjoint 100- and 1000-trace pool of (a) the full real test split, (b) the
unwatermarked small-GPT release none__k0__s0 (10,000 traces) and (c) the unwatermarked Qwen release (3,300 traces) is
tested with the pooled detector (GeoMark-KGW-2R conservative, Gumbel-H, exact-hash KGW). Reported per cell: total FPR
at 1e-3 and 1e-2, per-key FPR distribution (mean, sd, 90th/99th percentile, max), the binomial reference, an index-of-
dispersion test across keys, and the between-key variance ratio of the mean pooled z-score (1 = no key effect).
Output: results/rev1/keycond_pooled.json. Model-free (hashing only). Usage: python src/rev1/keycond.py"""
import os, sys, argparse, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import (cfg_rev1, OUT, load_release, jdump, PairIndex, uniforms_vec, log10_binom_sf, log10_gamma_sf,
                          fisher_log10, real_cells)
from geomark.common import Timer
from geomark.detect import pair_sets
from generate import wm_scheme

KEYS = list(range(3000, 3200))


def pools_of(cells_slots, sch, m):
    per = [pair_sets(c, s, sch) for c, s in cells_slots]
    pools = []
    for gi in range(len(per) // m):
        F, M = {}, {}
        for f, mm in per[gi * m:(gi + 1) * m]:
            F.update(dict.fromkeys(f)); M.update(dict.fromkeys(mm))
        pools.append((list(F), list(M)))
    return pools


def encode_pools(pools):
    idx = PairIndex()
    fs, fc, fp, ms, mc, mp = [], [], [], [], [], []
    for j, (F, M) in enumerate(pools):
        s, c = idx.encode(F); fs.append(s); fc.append(c); fp.append(np.full(len(s), j))
        s, c = idx.encode(M); ms.append(s); mc.append(c); mp.append(np.full(len(s), j))
    cat = lambda x: np.concatenate(x) if x else np.zeros(0, np.int64)
    return idx, (cat(fs), cat(fc), cat(fp)), (cat(ms), cat(mc), cat(mp))


def score_all(idx, F, M, sch, key, J):
    sf, cf, pf = F
    uf = uniforms_vec(key, 'fine', idx.seeds, sf, cf)
    nf = np.bincount(pf, minlength=J)
    if sch.score == 'kgw':
        Tf = np.bincount(pf, weights=(uf < sch.gamma).astype(float), minlength=J)
        lf = log10_binom_sf(Tf.astype(int), nf, sch.gamma)
        z = (Tf - nf * sch.gamma) / np.sqrt(np.maximum(nf, 1) * sch.gamma * (1 - sch.gamma))
    else:
        Tf = np.bincount(pf, weights=-np.log1p(-uf), minlength=J)
        lf = log10_gamma_sf(Tf, nf)
        z = (Tf - nf) / np.sqrt(np.maximum(nf, 1))
    if not sch.uses_mid:
        return lf, z
    sm, cm, pm = M
    um = uniforms_vec(key, 'mid', idx.seeds, sm, cm)
    nm = np.bincount(pm, minlength=J)
    if sch.score == 'kgw':
        Tm = np.bincount(pm, weights=(um < sch.gamma).astype(float), minlength=J)
        lm = log10_binom_sf(Tm.astype(int), nm, sch.gamma)
    else:
        Tm = np.bincount(pm, weights=-np.log1p(-um), minlength=J)
        lm = log10_gamma_sf(Tm, nm)
    return fisher_log10(lf, lm, nf, nm), z


def summarise(LP, Z, alpha):
    K, J = LP.shape
    k = (LP <= np.log10(alpha)).sum(1)
    fpr = k / J
    mean = k.mean()
    disp = float(((k - mean) ** 2).sum() / mean) if mean > 0 else float('nan')
    out = dict(n_keys=K, n_pools=J, n_tests=K * J, hits=int(k.sum()), fpr=float(k.sum() / (K * J)),
               per_key_mean=float(fpr.mean()), per_key_sd=float(fpr.std(ddof=1)),
               binom_sd=float(np.sqrt(alpha * (1 - alpha) / J)), per_key_p90=float(np.percentile(fpr, 90)),
               per_key_p99=float(np.percentile(fpr, 99)), per_key_max=float(fpr.max()),
               keys_above_2x=float((fpr > 2 * alpha).mean()),
               keys_above_2x_binom=float(stats.binom.sf(int(np.floor(2 * alpha * J)), J, alpha)),
               dispersion_stat=disp, dispersion_p=float(stats.chi2.sf(disp, K - 1)) if mean > 0 else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--nkeys', type=int, default=200)
    ap.add_argument('--out', default='keycond_pooled.json')
    a = ap.parse_args()
    global KEYS
    KEYS = KEYS[:a.nkeys]
    cfg = cfg_rev1()
    fn = os.path.join(OUT, a.out)
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    T = Timer()
    for ds, gpt, qwen in (('porto', 'gpt_porto_r9_s0', 'qwen15_porto_r9'), ('geolife', 'gpt_geolife_r9_s0', 'qwen15_geolife_r9')):
        cells, slots = real_cells(cfg, ds, 9, 'test')
        sources = {'real_test': list(zip(cells, slots))}
        for name, gen in (('gpt_none', gpt), ('qwen_none', qwen)):
            R = load_release(cfg, gen, 'none__k0__s0')
            if R is not None:
                sources[name] = [(tr['cells'], tr['slot']) for tr in R['traces']]
        for wm in ('gm_kgw', 'gm_gumbel', 'exact_kgw'):
            sch = wm_scheme(cfg, wm, 9)
            for src, cs in sources.items():
                for m in (100, 1000):
                    k = f'{ds}_r9|{wm}|{src}|m{m}'
                    if k in out:
                        continue
                    pools = pools_of(cs, sch, m)
                    if len(pools) < 2:
                        continue
                    idx, F, M = encode_pools(pools)
                    J = len(pools)
                    LP = np.zeros((len(KEYS), J)); Z = np.zeros((len(KEYS), J))
                    for i, key in enumerate(KEYS):
                        LP[i], Z[i] = score_all(idx, F, M, sch, key, J)
                    zbar = Z.mean(1)
                    rec = dict(n_traces=len(cs), pool_m=m, mean_pairs_per_pool=float(np.mean([len(f) for f, _ in pools])),
                               a1e3=summarise(LP, Z, 1e-3), a1e2=summarise(LP, Z, 1e-2),
                               z_between_key_var_ratio=float(J * zbar.var(ddof=1)),
                               z_mean=float(Z.mean()), z_sd=float(Z.std()),
                               ks_p_all=float(stats.kstest(np.clip(10 ** LP.ravel(), 0, 1), 'uniform').pvalue),
                               min_lp=float(LP.min()))
                    out[k] = rec
                    print(f"{k}: pools {J} x {len(KEYS)} keys, FPR@1e-3 {rec['a1e3']['fpr']:.2e} (p90 {rec['a1e3']['per_key_p90']:.3f}, "
                          f"max {rec['a1e3']['per_key_max']:.3f}), FPR@1e-2 {rec['a1e2']['fpr']:.3e} disp p {rec['a1e2']['dispersion_p']:.3g}, "
                          f"z var ratio {rec['z_between_key_var_ratio']:.2f} ({T():.0f}s)", flush=True)
                    jdump(out, a.out)


if __name__ == '__main__':
    main()
