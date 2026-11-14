"""Stage 8: aggregation of the revision-2 releases (factorial completion, step-neutral and EOS-neutral schemes on all
generator-dataset pairs, 50-key paired comparisons with the key as the unit). Reads results/eval/<gen>/*.parquet and
pooled.jsonl and results/rev1/utility_fixed.jsonl (de-duplicated by key, last record wins).
Output: results/rev2/summary_rev2.json. Model-free. Usage: python src/rev2/stats2.py"""
import os, sys, math
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, wilson
from rev1.stats import pq, pooled_lps, pstat, tstat, rids_for, utility_lookup, util_summary, seed_count, FACT_ATT, ALL_ATT

FACT = {'exact_nobin_1r': 'exact_kgw', 'exact_nobin_1r_d1': 'exact_kgw_d1', 'exact_bin_1r': 'exact_kgw_bin', 'exact_bin_2r': 'exact_bin_2r',
        'exact_bin_1r_eosn': 'exact_bin_eosn', 'coarse_nobin_2r': 'a_bin0', 'coarse_bin_1r': 'a_1r', 'coarse_bin_2r': 'gm_kgw',
        'k0_bin_1r': 'k0_1r', 'k0_bin_2r': 'a_k0', 'coarse_bin_2r_eosn': 'gm_kgw_eosn', 'coarse_bin_2r_sn': 'gm_kgw_sn'}
MK_WMS = ['gm_kgw', 'exact_kgw', 'exact_kgw_bin', 'a_k0', 'gm_kgw_eosn', 'gm_kgw_sn']


def scheme_block(gen, wm, U, seeds=(0, 1, 2), atts=ALL_ATT):
    rids = rids_for(wm, seeds=seeds)
    rec = {}
    for a in atts:
        p = pstat(pooled_lps(gen, rids, a))
        t = tstat(gen, rids, a) if not a.startswith('mix') else None
        if p or t:
            rec[a] = dict(pooled=p, trace=t)
        p1 = pstat(pooled_lps(gen, rids, a, m=1000))
        if p1:
            rec.setdefault(a, {})['pooled_m1000'] = p1
    rec['utility'] = util_summary(U, gen, rids)
    return rec


def manykey(cfg):
    gen = 'gpt_porto_r9_s0'
    keys = cfg['releases']['rev2_manykey']['keys']
    out = {}
    for attack in ('none', 'mm'):
        per = {}
        for wm in MK_WMS:
            tp, pm, det, ns = {}, {}, [], {}
            for k in keys:
                rid = f'{wm}__k{k}__s0'
                d = pq(gen, rid, attack)
                if d is not None:
                    tp[k] = float((d['lp'] <= -3).mean()); ns[k] = len(d)
                lps = pooled_lps(gen, [rid], attack)
                if lps:
                    pm[k] = float(-np.median(lps)); det += [x <= -3 for x in lps]
            if not pm:
                continue
            arr = np.array(list(tp.values())) if tp else np.array([])
            mean = float(np.mean(arr)) if len(arr) else None
            per[wm] = dict(n_keys=len(pm), pooled_median_by_key={str(k): v for k, v in pm.items()},
                           pooled_mlp_mean_over_keys=float(np.mean(list(pm.values()))),
                           pooled_detect_frac=float(np.mean(det)), pooled_n=len(det), pooled_k=int(np.sum(det)),
                           wilson_lo=wilson(int(np.sum(det)), len(det))[0],
                           keys_all_detected=int(sum(1 for k in keys if pooled_lps(gen, [f'{wm}__k{k}__s0'], attack) and
                                                     all(x <= -3 for x in pooled_lps(gen, [f'{wm}__k{k}__s0'], attack)))),
                           tpr_mean=mean, tpr_by_key={str(k): v for k, v in tp.items()})
        comps = {}
        for a_, b_ in (('gm_kgw', 'exact_kgw_bin'), ('gm_kgw', 'a_k0'), ('gm_kgw', 'exact_kgw'), ('a_k0', 'exact_kgw_bin'),
                       ('gm_kgw_eosn', 'exact_kgw_bin'), ('gm_kgw_sn', 'gm_kgw_eosn'), ('gm_kgw_sn', 'exact_kgw_bin'), ('gm_kgw_sn', 'a_k0')):
            if a_ not in per or b_ not in per:
                continue
            ks = sorted(set(per[a_]['pooled_median_by_key']) & set(per[b_]['pooled_median_by_key']), key=int)
            e = np.array([per[a_]['pooled_median_by_key'][k] - per[b_]['pooled_median_by_key'][k] for k in ks])
            if len(e) < 2:
                continue
            w = stats.wilcoxon(e) if np.any(e != 0) else None
            h = stats.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / math.sqrt(len(e))
            rec = dict(n_keys=len(e), mean=float(e.mean()), ci=[float(e.mean() - h), float(e.mean() + h)],
                       wilcoxon_p=float(w.pvalue) if w else None, n_a_better=int((e > 0).sum()))
            kt = sorted(set(per[a_]['tpr_by_key']) & set(per[b_]['tpr_by_key']), key=int)
            if len(kt) >= 2:
                d = np.array([per[a_]['tpr_by_key'][k] - per[b_]['tpr_by_key'][k] for k in kt])
                w2 = stats.wilcoxon(d) if np.any(d != 0) else None
                h2 = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
                rec['tpr'] = dict(mean=float(d.mean()), ci=[float(d.mean() - h2), float(d.mean() + h2)], wilcoxon_p=float(w2.pvalue) if w2 else None)
            comps[f'{a_}-{b_}'] = rec
        out[attack] = dict(per=per, paired=comps)
    return out


def main():
    cfg = cfg_rev2()
    U = utility_lookup()
    out = {}
    gen = 'gpt_porto_r9_s0'
    fac = {}
    for name, wm in FACT.items():
        rids = rids_for(wm)
        rec = dict(wm=wm, seeds=seed_count(cfg, gen, rids[0]))
        for a in FACT_ATT:
            rec[a] = dict(pooled=pstat(pooled_lps(gen, rids, a)), trace=tstat(gen, rids, a))
        rec['utility'] = util_summary(U, gen, rids)
        fac[name] = rec
        print('factorial', name, rec['seeds'], {a: (rec[a]['pooled'] or {}).get('median_mlp') for a in ('none', 'mm', 'mmrs', 'noise100')}, flush=True)
    out['factorial'] = fac
    sch = {}
    for g in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9'):
        for wm in ('gm_kgw', 'gm_kgw_eosn', 'gm_kgw_sn', 'exact_kgw', 'gm_gumbel', 'a_g25_d4', 'a_g25_d4_eosn', 'gm_gumbel_m2', 'none'):
            if wm == 'none':
                r = dict(utility=util_summary(U, g, ['none__k0__s0']))
            else:
                r = scheme_block(g, wm, U)
            if any(k != 'utility' for k in r) or r.get('utility'):
                sch[f'{g}|{wm}'] = r
    out['schemes'] = sch
    out['manykey'] = manykey(cfg)
    jdump2(out, 'summary_rev2.json')


if __name__ == '__main__':
    main()
