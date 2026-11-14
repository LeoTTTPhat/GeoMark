"""Stage 7: aggregation of the revision experiments and re-analyses of existing outputs -> results/rev1/summary_rev1.json.

Sections (each robust to missing inputs):
  keylevel      key-level re-analysis of the 3-key releases (per_replicate of results/summary.json): per-key TPRs,
                chi-square homogeneity, paired key-level t-interval and sign test (GeoMark-KGW vs exact-hash KGW)
  manykey       50 keys x 1,000 traces (rev1_manykey): per-key TPR / pooled evidence for exact-hash vs GeoMark-KGW,
                Wilcoxon signed-rank over keys (key = experimental unit), homogeneity chi-square with power
  factorial     {exact fine cell, coarse r-2} x {no bin, 30-min bin} (+ k = 0) at none / mm / noise30+mm / mmrs /
                retime60 / noise30 / noise100 / down3 / crop50, per trace and pooled, seed-space size, utility
  newschemes    EOS-neutral KGW (default, delta = 4), Gumbel-H with seed-level masking (M2), crypto-PRF and Qwen
                releases, all attacks, next to the original schemes
  posthoc       post-hoc rate sweep (strength-utility frontier)
  lengthmatch   per-trace TPR re-weighted to the unwatermarked generator's token-length distribution
  gumbel_distinct  Gumbel-H per-trace TPR on distinct traces only
  wilson        n and Wilson 95% bounds of the 'every map-matched pool detected' claims
  nullpools     distinct unwatermarked null pools behind the '7,890 pooled tests' statement
  mapmatch      per-scheme full-match rates
  road          road-snapped decoding (D3) results
  nextloc       next-location accuracy grouped by training-set cap (fixes the 4,999-trace grouping bug)
Usage: python src/rev1/stats.py"""
import os, sys, json, glob, re, math
from collections import defaultdict
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, jdump, ppath, read_jsonl, wilson, load_release
from geomark.detect import pair_sets

EV = ppath('results', 'eval')


def pq(gen, rid, attack):
    f = os.path.join(EV, gen, f'{rid}__{attack}.parquet')
    return pd.read_parquet(f) if os.path.exists(f) else None


_POOL = {}


def pooled_db(gen):
    if gen not in _POOL:
        d = {}
        for r in read_jsonl(os.path.join(EV, gen, 'pooled.jsonl')):
            d[r['key']] = r['lps']
        _POOL[gen] = d
    return _POOL[gen]


def pooled_lps(gen, rids, attack, m=100, T='None'):
    db = pooled_db(gen)
    out = []
    for rid in rids:
        out += db.get(f'{rid}|{attack}|{T}|m{m}', [])
    return out


def pstat(lps):
    if not lps:
        return None
    a = np.array(lps)
    k = int((a <= -3).sum())
    return dict(n_pools=len(a), median_mlp=float(-np.median(a)), q25_mlp=float(-np.percentile(a, 75)), q75_mlp=float(-np.percentile(a, 25)),
                frac_le_m3=float(k / len(a)), k_le_m3=k, wilson_lo=wilson(k, len(a))[0])


def tstat(gen, rids, attack):
    ds = [pq(gen, r, attack) for r in rids]
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    per = [float((d['lp'] <= -3).mean()) for d in ds]
    allr = pd.concat(ds)
    k = int((allr['lp'] <= -3).sum()); n = len(allr)
    out = dict(n_traces=n, tpr=k / n, wilson=wilson(k, n), per_release=per, median_n_f=float(allr['n_f'].median()))
    if 'surv_f' in allr:
        out['surv_f'] = float(allr['surv_f'].mean())
    return out


def rids_for(wm, keys=(0, 1, 2), seeds=(0,)):
    return [f'{wm}__k{k}__s{s}' for k in keys for s in seeds]


# ------------------------------------------------------------------------------------------------------------------
def keylevel(S):
    out = {}
    det = S['detection']
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9'):
        for attack in ('none', 'mm'):
            a, b = det.get(f'{gen}|gm_kgw|{attack}'), det.get(f'{gen}|exact_kgw|{attack}')
            if not a or not b:
                continue
            ra = {(r['key'], r['seed']): r for r in a['per_replicate']}
            rb = {(r['key'], r['seed']): r for r in b['per_replicate']}
            common = sorted(set(ra) & set(rb))
            diffs = np.array([rb[c]['tpr_1e3'] - ra[c]['tpr_1e3'] for c in common])
            # per-key aggregated TPRs and chi-square homogeneity across keys
            def by_key(rr):
                agg = defaultdict(lambda: [0, 0])
                for (k_, s_), r in rr.items():
                    agg[k_][0] += round(r['tpr_1e3'] * r['n']); agg[k_][1] += r['n']
                return {k_: v for k_, v in sorted(agg.items())}
            rec = {}
            for name, rr in (('gm_kgw', ra), ('exact_kgw', rb)):
                bk = by_key(rr)
                tab = np.array([[v[0], v[1] - v[0]] for v in bk.values()])
                chi = stats.chi2_contingency(tab)
                rec[name] = dict(per_key_tpr={str(k_): v[0] / v[1] for k_, v in bk.items()}, chi2=float(chi[0]), chi2_p=float(chi[1]),
                                 n_per_key=[v[1] for v in bk.values()])
            # key-level paired comparison: average replicates within key, t over keys; sign test over all replicates
            kd = defaultdict(list)
            for c, d in zip(common, diffs):
                kd[c[0]].append(d)
            kmeans = np.array([np.mean(v) for v in kd.values()])
            if len(kmeans) >= 2:
                t = stats.ttest_1samp(kmeans, 0.0)
                h = stats.t.ppf(0.975, len(kmeans) - 1) * kmeans.std(ddof=1) / math.sqrt(len(kmeans))
                rec['key_level'] = dict(n_keys=len(kmeans), mean_diff_exact_minus_gm=float(kmeans.mean()),
                                        ci=[float(kmeans.mean() - h), float(kmeans.mean() + h)], t_p=float(t.pvalue))
            npos = int((diffs > 0).sum())
            rec['sign_test'] = dict(n=len(diffs), n_exact_better=npos,
                                    p_two_sided=float(stats.binomtest(npos, len(diffs), 0.5).pvalue) if len(diffs) else None)
            out[f'{gen}|{attack}'] = rec
    return out


def manykey(cfg):
    gen = 'gpt_porto_r9_s0'
    keys = cfg['releases']['rev1_manykey']['keys']
    out = {}
    for attack in ('none', 'noise30', 'down3', 'crop50', 'mm'):
        per = {}
        for wm in ('exact_kgw', 'gm_kgw'):
            tp, pm, ns = {}, {}, {}
            for k in keys:
                d = pq(gen, f'{wm}__k{k}__s0', attack)
                if d is None:
                    continue
                tp[k] = float((d['lp'] <= -3).mean()); ns[k] = len(d)
                lps = pooled_lps(gen, [f'{wm}__k{k}__s0'], attack)
                if lps:
                    pm[k] = float(-np.median(lps))
            if not tp:
                continue
            arr = np.array(list(tp.values())); n = np.array(list(ns.values()))
            kk = np.round(arr * n).astype(int)
            chi = stats.chi2_contingency(np.c_[kk, n - kk]) if len(kk) > 1 else (np.nan, np.nan)
            mean = float(kk.sum() / n.sum())
            per[wm] = dict(n_keys=len(tp), traces_per_key=int(np.median(n)), tpr_mean=mean, tpr_sd_keys=float(arr.std(ddof=1)),
                           binom_sd=float(math.sqrt(mean * (1 - mean) / np.median(n))), var_ratio=float(arr.var(ddof=1) / (mean * (1 - mean) / np.median(n))) if mean > 0 else None,
                           chi2_p=float(chi[1]), tpr_by_key={str(k): v for k, v in tp.items()},
                           pooled_median_by_key={str(k): v for k, v in pm.items()},
                           pooled_mlp_mean_over_keys=float(np.mean(list(pm.values()))) if pm else None,
                           pooled_detect_frac=float(np.mean([x <= -3 for k in keys for x in pooled_lps(gen, [f'{wm}__k{k}__s0'], attack)])) if pm else None)
        if 'exact_kgw' in per and 'gm_kgw' in per:
            ks = sorted(set(map(int, per['exact_kgw']['tpr_by_key'])) & set(map(int, per['gm_kgw']['tpr_by_key'])))
            d = np.array([per['exact_kgw']['tpr_by_key'][str(k)] - per['gm_kgw']['tpr_by_key'][str(k)] for k in ks])
            w = stats.wilcoxon(d) if np.any(d != 0) else None
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
            per['tpr_diff_exact_minus_gm'] = dict(n_keys=len(ks), mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)],
                                                  wilcoxon_p=float(w.pvalue) if w else None, n_exact_better=int((d > 0).sum()))
            kp = sorted(set(map(int, per['exact_kgw']['pooled_median_by_key'])) & set(map(int, per['gm_kgw']['pooled_median_by_key'])))
            if kp:
                e = np.array([per['gm_kgw']['pooled_median_by_key'][str(k)] - per['exact_kgw']['pooled_median_by_key'][str(k)] for k in kp])
                w2 = stats.wilcoxon(e) if np.any(e != 0) else None
                h2 = stats.t.ppf(0.975, len(e) - 1) * e.std(ddof=1) / math.sqrt(len(e)) if len(e) > 1 else float('nan')
                per['pooled_diff_gm_minus_exact'] = dict(n_keys=len(kp), mean=float(e.mean()), ci=[float(e.mean() - h2), float(e.mean() + h2)],
                                                         wilcoxon_p=float(w2.pvalue) if w2 else None, n_gm_better=int((e > 0).sum()))
        out[attack] = per
    return out


def seed_count(cfg, gen, rid, n=5000):
    R = load_release(cfg, gen, rid)
    if R is None:
        return None
    from generate import wm_scheme
    wm = rid.split('__')[0]
    sch = wm_scheme(cfg, wm, cfg['generators'][gen]['res'])
    seeds = set()
    for tr in R['traces'][:n]:
        f, _ = pair_sets(tr['cells'], tr['slot'], sch)
        seeds.update(s for s, _ in f)
    return len(seeds)


def utility_lookup():
    U = {}
    f = os.path.join(OUT, 'utility_fixed.jsonl')
    for r in read_jsonl(f):
        U[r['key']] = r
    return U


def util_summary(U, gen, rids):
    rs = [U.get(f'{gen}|{r}') for r in rids]
    rs = [r for r in rs if r]
    if not rs:
        return None
    keys = ['jsd_bigram', 'jsd_od', 'jsd_cellpop', 'jsd_length_coded', 'jsd_length_ovf', 'jsd_gyration_ovf', 'w1_length_km',
            'ks_length', 'mean_length_km_gen', 'median_length_km_gen', 'mean_tok_gen', 'median_tok_gen', 'frac_above_p99_length',
            'duplicate_trace_frac', 'distinct_trigram_ratio', 'nn_jaccard', 'memorised_exact_frac']
    out = {'n_releases': len(rs)}
    for k in keys:
        v = [r[k] for r in rs if k in r and r[k] is not None]
        if v:
            out[k] = float(np.mean(v)); out[k + '_sd'] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    out['mean_length_km_real'] = rs[0].get('mean_length_km_real'); out['mean_tok_real'] = rs[0].get('mean_tok_real')
    return out


FACT_ATT = ['none', 'mm', 'noise30+mm', 'mmrs', 'retime60', 'noise30', 'noise100', 'down3', 'crop50']


def factorial(cfg, U):
    gen = 'gpt_porto_r9_s0'
    cells = {'exact_nobin': 'exact_kgw', 'exact_bin': 'exact_kgw_bin', 'coarse_nobin': 'a_bin0', 'coarse_bin': 'gm_kgw',
             'bin_only_k0': 'a_k0', 'exact_nobin_d1': 'exact_kgw_d1', 'coarse_bin_1r': 'a_1r'}
    out = {}
    for name, wm in cells.items():
        rids = rids_for(wm)
        rec = dict(wm=wm, seeds=seed_count(cfg, gen, rids[0]))
        for a in FACT_ATT:
            rec[a] = dict(pooled=pstat(pooled_lps(gen, rids, a)), trace=tstat(gen, rids, a))
        rec['utility'] = util_summary(U, gen, rids)
        out[name] = rec
        print('factorial', name, {a: (rec[a]['pooled'] or {}).get('median_mlp') for a in FACT_ATT}, flush=True)
    return out


ALL_ATT = ['none', 'noise10', 'noise30', 'noise100', 'down2', 'down3', 'down5', 'retime15', 'retime60', 'crop25', 'crop50',
           'splice30', 'mm', 'mmrs', 'noise30+mm', 'mix30', 'mix50']


def scheme_block(gen, wm, U, seeds=(0, 1, 2)):
    rids = rids_for(wm, seeds=seeds)
    rec = {}
    for a in ALL_ATT:
        p = pstat(pooled_lps(gen, rids, a))
        t = tstat(gen, rids, a) if not a.startswith('mix') else None
        if p or t:
            rec[a] = dict(pooled=p, trace=t)
    rec['utility'] = util_summary(U, gen, rids)
    return rec


def newschemes(cfg, U):
    out = {}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0'):
        for wm in ('gm_kgw', 'gm_kgw_eosn', 'a_g25_d4', 'a_g25_d4_eosn', 'gm_gumbel', 'gm_gumbel_m2', 'exact_kgw', 'none'):
            r = scheme_block(gen, wm, U) if wm != 'none' else dict(utility=util_summary(U, gen, ['none__k0__s0', 'none__k0__s1', 'none__k0__s2']))
            out[f'{gen}|{wm}'] = r
    for wm in ('gm_kgw', 'gm_kgw_eosn', 'exact_kgw', 'gm_gumbel', 'a_g25_d4'):
        out[f'qwen15_porto_r9|{wm}'] = scheme_block('qwen15_porto_r9', wm, U)
    for wm in ('gm_kgw', 'exact_kgw', 'gm_gumbel'):
        out[f'qwen15_geolife_r9|{wm}'] = scheme_block('qwen15_geolife_r9', wm, U)
    out['qwen15_porto_r9|none'] = dict(utility=util_summary(U, 'qwen15_porto_r9', ['none__k0__s0']))
    # crypto PRF releases vs the standard PRF on the same generator (first 2,000 traces)
    gen = 'gpt_porto_r9_s0'
    cr = [f'gm_kgw__k{k}__s0' for k in (10000000, 10000001, 10000002)]
    out['crypto'] = dict(crypto={a: dict(pooled=pstat(pooled_lps(gen, cr, a)), trace=tstat(gen, cr, a)) for a in ('none', 'noise30', 'crop50')},
                         standard={a: dict(pooled=pstat(pooled_lps(gen, rids_for('gm_kgw'), a)), trace=tstat(gen, rids_for('gm_kgw'), a))
                                   for a in ('none', 'noise30', 'crop50')})
    # Gumbel M2 key mix duplicates / utility
    out['gumbel_m2_keymix_utility'] = U.get('gpt_porto_r9_s0|keymix:gm_gumbel_m2')
    out['gumbel_m1_keymix_utility'] = U.get('gpt_porto_r9_s0|keymix:gm_gumbel')
    out['kgw_keymix_utility'] = U.get('gpt_porto_r9_s0|keymix:gm_kgw')
    return out


def posthoc(cfg, U):
    gen = 'gpt_porto_r9_s0'
    out = {}
    for wm, rate in (('posthoc_r10', 0.1), ('posthoc_r25', 0.25), ('posthoc_r50', 0.5), ('posthoc', 1.0)):
        rids = rids_for(wm)
        out[wm] = dict(rate=rate, none=pstat(pooled_lps(gen, rids, 'none')), mm=pstat(pooled_lps(gen, rids, 'mm')),
                       noise30mm=pstat(pooled_lps(gen, rids, 'noise30+mm')), trace_none=tstat(gen, rids, 'none'),
                       trace_mm=tstat(gen, rids, 'mm'), utility=util_summary(U, gen, rids))
    for wm in ('a_g25_d1', 'gm_kgw', 'a_g25_d4', 'gm_kgw_eosn', 'a_g25_d4_eosn', 'exact_kgw', 'exact_kgw_d1', 'exact_kgw_bin'):
        rids = rids_for(wm)
        out[wm] = dict(none=pstat(pooled_lps(gen, rids, 'none')), mm=pstat(pooled_lps(gen, rids, 'mm')),
                       noise30mm=pstat(pooled_lps(gen, rids, 'noise30+mm')), trace_none=tstat(gen, rids, 'none'),
                       trace_mm=tstat(gen, rids, 'mm'), utility=util_summary(U, gen, rids))
    return out


def lengthmatch():
    """Per-trace TPR re-weighted to the unwatermarked generator's token-length distribution (Porto small GPT, none)."""
    gen = 'gpt_porto_r9_s0'
    ref = None
    R = load_release(cfg_rev1(), gen, 'none__k0__s0')
    if R is None:
        return {}
    ref = np.array([len(t['cells']) for t in R['traces']])
    bins = np.array([0, 5, 8, 10, 12, 14, 16, 18, 20, 24, 28, 34, 40, 50, 70, 1000])
    pref = np.histogram(ref, bins)[0] / len(ref)
    out = {}
    for wm in ('gm_kgw', 'gm_kgw_eosn', 'a_g25_d4', 'a_g25_d4_eosn', 'a_g50_d4', 'exact_kgw', 'gm_gumbel', 'gm_gumbel_m2'):
        ds = [pq(gen, r, 'none') for r in rids_for(wm)]
        ds = [d for d in ds if d is not None]
        if not ds:
            continue
        d = pd.concat(ds)
        L = d['len_tok'].to_numpy(); hit = (d['lp'] <= -3).to_numpy()
        b = np.clip(np.digitize(L, bins) - 1, 0, len(bins) - 2)
        pw = np.bincount(b, minlength=len(bins) - 1) / len(b)
        w = np.where(pw[b] > 0, pref[b] / np.maximum(pw[b], 1e-12), 0)
        out[wm] = dict(tpr_raw=float(hit.mean()), tpr_length_matched=float((w * hit).sum() / w.sum()),
                       mean_len_tok=float(L.mean()), ref_mean_len_tok=float(ref.mean()))
    return out


def gumbel_distinct(cfg):
    out = {}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9'):
        seeds = (0, 1, 2) if gen.startswith('gpt') else (0,)
        tot_h = tot_n = 0
        per = []
        for k in (0, 1, 2):
            for s in seeds:
                rid = f'gm_gumbel__k{k}__s{s}'
                d = pq(gen, rid, 'none'); R = load_release(cfg, gen, rid)
                if d is None or R is None:
                    continue
                seqs = [tuple(int(x) for x in t['cells']) for t in R['traces']]
                first = {}
                for i, q in enumerate(seqs):
                    first.setdefault(q, i)
                idx = sorted(first.values())
                dd = d.set_index('i').loc[[i for i in idx if i in set(d['i'])]]
                h = int((dd['lp'] <= -3).sum()); n = len(dd)
                per.append(dict(rid=rid, distinct=n, total=len(seqs), tpr_distinct=h / max(n, 1), tpr_all=float((d['lp'] <= -3).mean())))
                tot_h += h; tot_n += n
        if per:
            out[gen] = dict(per=per, tpr_distinct=tot_h / tot_n, n_distinct=tot_n, wilson=wilson(tot_h, tot_n))
    return out


def wilson_claims():
    out = {}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9'):
        for wm in ('gm_kgw', 'gm_gumbel', 'exact_kgw'):
            for a in ('mm', 'noise30+mm', 'mmrs'):
                for m in (100, 1000):
                    p = pstat(pooled_lps(gen, rids_for(wm), a, m=m))
                    if p:
                        out[f'{gen}|{wm}|{a}|m{m}'] = p
    return out


def nullpools():
    """Distinct pools behind the unwatermarked pooled tests (pools are shared across attacks)."""
    out = {}
    tot_tests = tot_hits = 0
    for gen in sorted(os.listdir(EV)):
        if gen == 'real':
            continue
        db = pooled_db(gen)
        for key, lps in db.items():
            rid, attack, T, m = key.split('|')
            if not rid.startswith('none__') or m != 'm100':
                continue
            a = np.array(lps)
            tot_tests += len(a); tot_hits += int((a <= -3).sum())
            o = out.setdefault(f'{gen}|{T}', dict(pools_per_attack=[], hits=[], attacks=[], per_rid={}))
            o['pools_per_attack'].append(len(a)); o['hits'].append(int((a <= -3).sum())); o['attacks'].append(attack)
            o['per_rid'][rid] = max(o['per_rid'].get(rid, 0), len(a))
    summ = {k: dict(distinct_pools=int(sum(v['per_rid'].values())), releases=len(v['per_rid']), attacks=len(set(v['attacks'])),
                    tests=int(sum(v['pools_per_attack'])), hits=int(sum(v['hits'])),
                    hit_attacks=[a for a, h in zip(v['attacks'], v['hits']) if h]) for k, v in out.items()}
    return dict(cells=summ, total_tests=tot_tests, total_hits=tot_hits,
                total_distinct_pools=int(sum(v['distinct_pools'] for v in summ.values())))


def mapmatch_rates(S):
    out = defaultdict(list)
    for r in S['mapmatch']:
        tag = r['tag']
        if '/' in tag:
            gen, rid = tag.split('/')
            wm = rid.split('__')[0]
            out[f"{gen}|{wm}|{r['prefix']}"].append(r['full_match'])
        else:
            out[f"{tag}|real|{r['prefix']}"].append(r['full_match'])
    for r in read_jsonl(ppath('results', 'attacked', 'mm_log.jsonl')):
        tag = r['tag']
        if '/' in tag and any(x in tag for x in ('eosn', 'm2', 'exact_kgw_bin', 'exact_kgw_d1', 'posthoc_r')):
            gen, rid = tag.split('/')
            out[f"{gen}|{rid.split('__')[0]}|{r['prefix']}"].append(r['full_match'])
    return {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in out.items()}


def road(S):
    d = S['detection']
    out = {}
    for k, v in d.items():
        if '|road|' in k or '|centroid|' in k:
            out[k] = dict(tpr=float(np.mean(v['tpr_1e3'][:1])) if isinstance(v['tpr_1e3'], list) else v['tpr_1e3'],
                          surv_f=v.get('surv_f', [None])[0] if isinstance(v.get('surv_f'), list) else v.get('surv_f'), n=v.get('n_traces'))
    return out


def nextloc():
    nl = read_jsonl(ppath('results', 'utility', 'nextloc.jsonl'))
    g = defaultdict(list)
    for r in nl:
        cap = 5000 if r['n_train'] <= 5000 else (10000 if r['n_train'] <= 10000 else 20000)
        g[(r['gen'], r['rid'].split('__')[0], cap)].append(r['acc1'])
    out = {}
    for (gen, wm, cap), v in g.items():
        base = g.get((gen, 'none', cap))
        rec = dict(mean=float(np.mean(v)), n=len(v), values=v)
        if base and wm not in ('none', 'real') and len(v) >= 2 and len(base) >= 2:
            rec['welch_p'] = float(stats.ttest_ind(v, base, equal_var=False).pvalue); rec['mean_none'] = float(np.mean(base))
        out[f'{gen}|{wm}|cap{cap}'] = rec
    return out


def fullmatch(cfg):
    """Map-matching results restricted to the traces that matched completely (frac_matched >= 0.999)."""
    from geomark.prf import PRF
    from geomark.detect import pooled, trace_cells
    from generate import wm_scheme
    from geomark.common import load_pickle
    out = {}
    for gen in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0'):
        for wm in ('gm_kgw', 'exact_kgw', 'a_g25_d4', 'gm_kgw_eosn'):
            sch = wm_scheme(cfg, wm, 9)
            hits = n = 0; pools = []; nfull = ntot = 0
            for k in (0, 1, 2):
                rid = f'{wm}__k{k}__s0'
                f = ppath('results', 'attacked', gen, f'{rid}__mm.pkl')
                d = pq(gen, rid, 'mm')
                if d is None or not os.path.exists(f):
                    continue
                C = load_pickle(f)
                full = [i for i, inf in enumerate(C['info']) if inf.get('frac', 0) >= 0.999]
                nfull += len(full); ntot += len(C['info'])
                dd = d[d['i'].isin(full)]
                hits += int((dd['lp'] <= -3).sum()); n += len(dd)
                prf = PRF(k)
                pl = [pair_sets(*trace_cells(C['traces'][i], sch), sch) for i in full if C['traces'][i] is not None and len(C['traces'][i]['lat'])]
                pools += [pooled(pl[j:j + 100], sch, prf)['lp'] for j in range(0, len(pl) - 99, 100)]
            if n:
                out[f'{gen}|{wm}'] = dict(full_frac=nfull / ntot, tpr_full=hits / n, n_full=n, pooled=pstat(pools))
    return out


def main():
    cfg = cfg_rev1()
    S = json.load(open(ppath('results', 'summary.json')))
    U = utility_lookup()
    out = {}
    for name, fn in (('keylevel', lambda: keylevel(S)), ('manykey', lambda: manykey(cfg)), ('factorial', lambda: factorial(cfg, U)),
                     ('newschemes', lambda: newschemes(cfg, U)), ('posthoc', lambda: posthoc(cfg, U)), ('lengthmatch', lengthmatch),
                     ('gumbel_distinct', lambda: gumbel_distinct(cfg)), ('wilson', wilson_claims), ('nullpools', nullpools),
                     ('mapmatch', lambda: mapmatch_rates(S)), ('road', lambda: road(S)), ('nextloc', nextloc),
                     ('fullmatch', lambda: fullmatch(cfg))):
        try:
            out[name] = fn()
            print('done', name, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[name] = dict(error=str(e))
    jdump(out, 'summary_rev1.json')


if __name__ == '__main__':
    main()
