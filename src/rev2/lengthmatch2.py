"""Stage 8: length-matched detection on BOTH datasets, re-weighted to the REAL trip-length distribution, and pooled
fine-only vs. two-resolution results (reviewers: the GeoLife generator makes trips about 3x too long, which inflates GeoLife
detection; length matching was done only for Porto and to the generator's distribution; the detector of the recommended
configuration was not specified).

Per trace: TPR at p <= 1e-3 (rows of results/eval/<gen>/<rid>__none.parquet, keys 0-2 seed 0) re-weighted by the ratio
of the real-test token-length histogram to the release's histogram (bins as src/rev1/stats.lengthmatch), and to the
unwatermarked generator's histogram. Pooled: 100-trace pools drawn by weighted sampling without replacement with these
weights (20 pools per key, from the first 5,000 traces), so pools mimic the real length distribution. Fine-only vs.
Fisher: pooled 100-trace tests (first 2,000 traces, and the 1,000 map-matched traces) with both detectors.
Output: results/rev2/lengthmatch2.json. Model-free. Usage: python src/rev2/lengthmatch2.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, real_cells, ppath, load_pickle
from geomark.common import seed_from, Timer
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, pooled
from generate import wm_scheme

BINS = np.array([0, 5, 8, 10, 12, 14, 16, 18, 20, 24, 28, 34, 40, 50, 70, 1000])
GENS = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']
WMS = ['gm_kgw', 'gm_kgw_eosn', 'gm_kgw_sn', 'exact_kgw', 'a_g25_d4', 'a_g25_d4_eosn', 'exact_kgw_bin', 'a_k0']


def hist(L):
    b = np.clip(np.digitize(L, BINS) - 1, 0, len(BINS) - 2)
    return np.bincount(b, minlength=len(BINS) - 1) / max(1, len(b)), b


def main():
    cfg = cfg_rev2()
    out = jload2('lengthmatch2.json')
    T = Timer()
    for gen in GENS:
        g = cfg['generators'][gen]; ds = g['dataset']
        rc, _ = real_cells(cfg, ds, 9, 'test')
        p_real, _ = hist(np.array([len(c) for c in rc]))
        R0 = load_release(cfg, gen, 'none__k0__s0')
        p_gen = hist(np.array([len(t['cells']) for t in R0['traces']]))[0] if R0 else None
        out[f'{gen}|lengths'] = dict(real_mean_tok=float(np.mean([len(c) for c in rc])),
                                    gen_none_mean_tok=float(np.mean([len(t['cells']) for t in R0['traces']])) if R0 else None)
        for wm in WMS:
            k = f'{gen}|{wm}'
            if k in out:
                continue
            sch = wm_scheme(cfg, wm, 9)
            rows, pools_raw, pools_lm, fo = [], [], [], {'none': ([], []), 'mm': ([], [])}
            for key in (0, 1, 2):
                rid = f'{wm}__k{key}__s0'
                f = ppath('results', 'eval', gen, f'{rid}__none.parquet')
                R = load_release(cfg, gen, rid)
                if R is None or not os.path.exists(f):
                    continue
                d = pd.read_parquet(f)
                rows.append(d[['len_tok', 'lp']])
                prf = PRF(key)
                trs = R['traces'][:5000]
                pl = [pair_sets(t['cells'], t['slot'], sch) for t in trs]
                L = np.array([len(t['cells']) for t in trs])
                ph, b = hist(L)
                w = np.where(ph[b] > 0, p_real[b] / np.maximum(ph[b], 1e-12), 0.0)
                rng = np.random.default_rng(seed_from('lm2', gen, rid))
                for j in range(20):
                    idx = rng.choice(len(pl), 100, replace=False, p=w / w.sum())
                    pools_lm.append(pooled([pl[i] for i in idx], sch, prf)['lp'])
                    pools_raw.append(pooled(pl[j * 100:(j + 1) * 100], sch, prf)['lp'])
                for att in ('none', 'mm'):
                    if att == 'none':
                        P = pl[:2000]
                    else:
                        fm = ppath('results', 'attacked', gen, f'{rid}__mm.pkl')
                        if not os.path.exists(fm):
                            continue
                        C = load_pickle(fm)['traces']
                        P = [pair_sets(*trace_cells(t, sch), sch) if t is not None and len(t['lat']) else ([], []) for t in C]
                    for j in range(0, len(P) - 99, 100):
                        r = pooled(P[j:j + 100], sch, prf)
                        fo[att][0].append(r['lp']); fo[att][1].append(r['lp_f'])
            if not rows:
                continue
            d = pd.concat(rows)
            Lt = d['len_tok'].to_numpy(); hit = (d['lp'] <= -3).to_numpy()
            ph, b = hist(Lt)
            rec = dict(n=len(d), tpr_raw=float(hit.mean()), mean_len_tok=float(Lt.mean()))
            for ref, pr in (('real', p_real), ('gen_none', p_gen)):
                if pr is None:
                    continue
                w = np.where(ph[b] > 0, pr[b] / np.maximum(ph[b], 1e-12), 0.0)
                rec[f'tpr_lm_{ref}'] = float((w * hit).sum() / w.sum())
            a, c = np.array(pools_raw), np.array(pools_lm)
            rec['pooled_raw'] = dict(median_mlp=float(-np.median(a)), frac_le_m3=float((a <= -3).mean()), n=len(a))
            rec['pooled_lm_real'] = dict(median_mlp=float(-np.median(c)), frac_le_m3=float((c <= -3).mean()), n=len(c))
            for att, (fi, ff) in fo.items():
                if fi:
                    fi, ff = np.array(fi), np.array(ff)
                    rec[f'fisher_vs_fine_{att}'] = dict(fisher_median_mlp=float(-np.median(fi)), fine_median_mlp=float(-np.median(ff)),
                                                        fisher_det=float((fi <= -3).mean()), fine_det=float((ff <= -3).mean()), n=len(fi))
            out[k] = rec
            print(f"{k}: tpr {rec['tpr_raw']:.4f} lm_real {rec.get('tpr_lm_real', float('nan')):.4f}; pooled {rec['pooled_raw']['median_mlp']:.1f} "
                  f"lm {rec['pooled_lm_real']['median_mlp']:.1f}; fisher/fine none {rec.get('fisher_vs_fine_none', {}).get('fisher_median_mlp')}/"
                  f"{rec.get('fisher_vs_fine_none', {}).get('fine_median_mlp')} ({T():.0f}s)", flush=True)
            jdump2(out, 'lengthmatch2.json')


if __name__ == '__main__':
    main()
