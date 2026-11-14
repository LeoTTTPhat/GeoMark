"""Stage 5 statistical analysis layer: significance tests, effect sizes and CIs for every claim in PLAN.md.
Reads results/summary.json (aggregated cells) and the raw per-trace / per-group files it was built from
(results/eval/*/*.parquet, eval/*/pooled.jsonl, keyvar/*.jsonl, utility/*.jsonl, steal/results_spoof5000.jsonl).
Writes results/stats_tests.json (consumed by src/make_tables.py, src/make_figures.py and results/analysis.md).
No model is loaded; runtime < 1 min. Usage: python src/stats_tests.py [--root PROJECT_DIR]

Statistical units (see results/GAPS.md Sec. D):
* KGW-type schemes: between-key variance of per-trace TPR equals the binomial within-key variance (keyvar_tpr), so traces
  are treated as units (two-proportion tests, Newcombe intervals, McNemar on the same traces).
* Gumbel-H: the key effect dominates, so keys are the unit (50-key / 12-key tables; chi-square homogeneity test).
* Release-level comparisons: Wilcoxon signed-rank over releases paired by (key, seed) with Holm correction, as planned
  (with 3 or 9 pairs the minimum attainable two-sided p is 0.25 / 0.0039, stated with every result).
"""
import os, sys, re, glob, json, math, argparse
from collections import defaultdict
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out.tolist()


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float('nan')] * 3
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [ph, max(0.0, c - h), min(1.0, c + h)]


def newcombe_diff(k1, n1, k2, n2):
    """Difference p1 - p2 with Newcombe (hybrid Wilson) 95% interval and a two-proportion z-test p."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else 0.0
    pz = float(2 * stats.norm.sf(abs(d) / se)) if se > 0 else (1.0 if d == 0 else 0.0)
    return dict(p1=p1, p2=p2, diff=d, lo=lo, hi=hi, p=pz, n1=n1, n2=n2, k1=k1, k2=k2)


def mcnemar(a, b):
    """Exact McNemar on paired booleans a, b (same traces). Returns diff = mean(a) - mean(b) and p."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    nd = n01 + n10
    p = float(stats.binomtest(n10, nd, 0.5).pvalue) if nd else 1.0
    return dict(n=int(len(a)), tpr_a=float(a.mean()), tpr_b=float(b.mean()), diff=float(a.mean() - b.mean()),
                n10=n10, n01=n01, p=p)


def wilcoxon(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    n = int(len(d))
    pmin = 2.0 / 2 ** n if n else float('nan')
    if n == 0 or np.allclose(d, 0):
        return dict(n=n, stat=float('nan'), p=1.0, p_min_attainable=pmin, mean_diff=float(np.mean(d)) if n else float('nan'))
    r = stats.wilcoxon(x, y, method='exact' if n <= 25 else 'auto')
    return dict(n=n, stat=float(r.statistic), p=float(r.pvalue), p_min_attainable=pmin, mean_diff=float(d.mean()),
                frac_pos=float((d > 0).mean()))


def mannwhitney(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return dict(p=float('nan'))
    r = stats.mannwhitneyu(x, y, alternative='two-sided')
    # common-language effect size P(X < Y) (X more significant = smaller log10 p)
    cles = float(r.statistic / (len(x) * len(y)))
    return dict(nx=int(len(x)), ny=int(len(y)), U=float(r.statistic), p=float(r.pvalue), cles_x_gt_y=cles,
                median_x=float(np.median(x)), median_y=float(np.median(y)))


def clip_lp(v, floor=-300.0):
    v = np.asarray(v, float)
    return np.where(np.isfinite(v), v, floor)


# ----------------------------------------------------------------------------------------------------------------------
def calibration(S):
    """P1: KS uniformity of randomised p-values on 1e5 / 2e5 nulls; one-sided binomial exceedance tests of empirical FPR."""
    ce = S['calibration_ext']
    rows = []
    for cell, v in sorted(ce.items()):
        r = dict(cell=cell, ds=v['ds'], res=v['res'], scheme=v['scheme'], source=v['source'], n_tests=v['n_tests'],
                 median_pairs=v['median_pairs'], ks_stat=v['ks_randomized']['ks_stat'], ks_p=v['ks_randomized']['ks_p'])
        for kind in ('fpr_conservative', 'fpr_randomized'):
            for a in ('0.01', '0.001', '0.0001'):
                e = v[kind][a]
                r[f'{kind}_{a}'] = e['fpr']
                r[f'{kind}_{a}_k'] = e['k']
                r[f'{kind}_{a}_ci'] = e['ci']
                # exceedance test: H0 FPR <= alpha
                r[f'{kind}_{a}_p_exceed'] = float(stats.binomtest(e['k'], e['n'], float(a), alternative='greater').pvalue)
        pe = v['pooled']['fpr']['0.001']
        r['pooled_fpr_0.001'] = pe['fpr']; r['pooled_fpr_0.001_ci'] = pe['ci']; r['pooled_n'] = pe['n']
        r['pooled_fpr_0.001_p_exceed'] = float(stats.binomtest(pe['k'], pe['n'], 0.001, alternative='greater').pvalue)
        r['pooled_ks_p'] = v['pooled']['ks_randomized']['ks_p']
        r['sec'] = v.get('sec')
        rows.append(r)
    for key in ('ks_p', 'fpr_conservative_0.001_p_exceed', 'fpr_randomized_0.001_p_exceed', 'pooled_fpr_0.001_p_exceed', 'pooled_ks_p'):
        adj = holm([r[key] for r in rows])
        for r, a in zip(rows, adj):
            r[key + '_holm'] = a
    ks = np.array([r['ks_p'] for r in rows])
    summ = dict(n_cells=len(rows), n_ks_below_005=int((ks < 0.05).sum()), min_ks_p=float(ks.min()),
                min_ks_p_holm=float(min(r['ks_p_holm'] for r in rows)),
                max_fpr_rand_1e3=float(max(r['fpr_randomized_0.001'] for r in rows)),
                max_fpr_cons_1e3=float(max(r['fpr_conservative_0.001'] for r in rows)),
                min_exceed_p_holm_rand_1e3=float(min(r['fpr_randomized_0.001_p_exceed_holm'] for r in rows)),
                min_exceed_p_holm_pooled_1e3=float(min(r['pooled_fpr_0.001_p_exceed_holm'] for r in rows)),
                total_tests=int(sum(r['n_tests'] for r in rows)))
    # detection throughput (tests per second) from the calibration runs (hashing-only detector)
    thr = [r['n_tests'] / r['sec'] for r in rows if r.get('sec')]
    summ['detect_tests_per_sec_median'] = float(np.median(thr)) if thr else None
    return dict(cells=rows, summary=summ)


def keycond_fpr(root, S):
    """P1 for a single owner: per-key FPR over 200 keys x 3,000 real traces; quantiles and dispersion."""
    out = {}
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'keyvar', 'null.jsonl'))]
    by = defaultdict(list)
    for r in recs:
        by[f"{r['cell']}|{r['scheme']}"].append(r)
    for cell, rs in sorted(by.items()):
        n = rs[0]['n']
        e = dict(n_keys=len(rs), n_per_key=n)
        for t, a in ((2, 0.01), (3, 0.001), (4, 0.0001)):
            k = np.array([r[f'k_le_m{t}'] for r in rs])
            f = k / n
            # fraction of keys whose own exact binomial test rejects FPR <= alpha at 5% (would be ~5% or less if per-key binomial)
            prej = np.array([stats.binomtest(int(x), n, a, alternative='greater').pvalue for x in k])
            e[f'{a:g}'] = dict(mean=float(f.mean()), sd=float(f.std(ddof=1)), q50=float(np.quantile(f, .5)), q90=float(np.quantile(f, .9)),
                               max=float(f.max()), frac_gt_2x=float((f > 2 * a).mean()),
                               frac_keys_reject_5pct=float((prej < 0.05).mean()),
                               expected_frac_gt_2x_binom=float(stats.binom.sf(math.floor(2 * a * n), n, f.mean())))
        out[cell] = e
    out['_from_summary'] = {k: {a: S['keyvar']['null'][k][f'fpr_{a}']['dispersion_p'] for a in ('0.01', '0.001', '0.0001')}
                            for k in S['keyvar']['null']}
    return out


def key_effect(root, S):
    """Is per-trace TPR key-dependent? Chi-square homogeneity over keys (counts = TPR x 200) and variance ratio."""
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'keyvar', 'tpr.jsonl'))]
    by = defaultdict(list)
    for r in recs:
        by[f"{r['gen']}|{r['wm']}|{r['attack']}"].append(r)
    out = {}
    for cell, rs in sorted(by.items()):
        k = np.array([round(r['tpr_1e3'] * r['n']) for r in rs]); n = np.array([r['n'] for r in rs])
        tab = np.vstack([k, n - k])
        if tab[0].sum() == 0:
            out[cell] = dict(n_keys=len(rs), chi2=float('nan'), p=1.0); continue
        tab = tab[:, tab.sum(0) > 0]
        chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
        f = k / n
        pbar = k.sum() / n.sum()
        out[cell] = dict(n_keys=len(rs), mean=float(f.mean()), sd_keys=float(f.std(ddof=1)), min=float(f.min()), max=float(f.max()),
                         var_ratio=float(f.var(ddof=1) / (pbar * (1 - pbar) / n.mean())) if 0 < pbar < 1 else float('nan'),
                         chi2=float(chi2), dof=int(dof), p=float(p))
    # paired-by-key attack effects for Gumbel-H and KGW (Wilcoxon over keys, Holm within table)
    paired = {}
    for cell in list(by):
        gen, wm, atk = cell.split('|')
        if atk == 'none':
            continue
        base = {r['wm_key']: r['tpr_1e3'] for r in by[f'{gen}|{wm}|none']}
        att = {r['wm_key']: r['tpr_1e3'] for r in by[cell]}
        ks = sorted(set(base) & set(att))
        paired[cell] = wilcoxon([att[k] for k in ks], [base[k] for k in ks])
    adj = holm([v['p'] for v in paired.values()])
    for (k, v), a in zip(paired.items(), adj):
        v['p_holm'] = a
    # Gumbel-H vs KGW over keys (unpaired: different keys' traces; Mann-Whitney)
    gvk = {}
    for gen in sorted({c.split('|')[0] for c in by}):
        for atk in ('none', 'noise30', 'down3', 'crop50'):
            g = [r['tpr_1e3'] for r in by.get(f'{gen}|gm_gumbel|{atk}', [])]
            kk = [r['tpr_1e3'] for r in by.get(f'{gen}|gm_kgw|{atk}', [])]
            if g and kk:
                gvk[f'{gen}|{atk}'] = mannwhitney(g, kk)
    adj = holm([v['p'] for v in gvk.values()])
    for (k, v), a in zip(gvk.items(), adj):
        v['p_holm'] = a
    return dict(homogeneity=out, attack_vs_none_by_key=paired, gumbel_vs_kgw_over_keys=gvk)


def release_comparisons(S):
    """Planned protocol: Wilcoxon signed-rank over releases paired by (key, seed), Holm over attacks per comparison family.
    Plus trace-level two-proportion tests (valid for KGW-type schemes) with Newcombe intervals."""
    det = S['detection']
    fams = {
        'gm_kgw_vs_exact_kgw': ('gm_kgw', 'exact_kgw'),
        'gm_gumbel_vs_gm_kgw': ('gm_gumbel', 'gm_kgw'),
        'a_g25_d4_vs_gm_kgw': ('a_g25_d4', 'gm_kgw'),
        'a_g50_d4_vs_a_g25_d4': ('a_g50_d4', 'a_g25_d4'),
        'gm_kgw_vs_posthoc': ('gm_kgw', 'posthoc'),
    }
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']
    out = {}
    for fam, (a, b) in fams.items():
        res = {}
        for gen in gens:
            atks = sorted({k.split('|')[2] for k in det if k.startswith(f'{gen}|{a}|') and len(k.split('|')) == 3})
            for atk in atks:
                ca, cb = det.get(f'{gen}|{a}|{atk}'), det.get(f'{gen}|{b}|{atk}')
                if not ca or not cb or 'per_replicate' not in ca or 'per_replicate' not in cb:
                    continue
                ra = {(r['key'], r['seed']): r for r in ca['per_replicate']}
                rb = {(r['key'], r['seed']): r for r in cb['per_replicate']}
                ks = sorted(set(ra) & set(rb))
                e = dict(n_pairs=len(ks))
                if ks:
                    e['wilcoxon_tpr_1e3'] = wilcoxon([ra[k]['tpr_1e3'] for k in ks], [rb[k]['tpr_1e3'] for k in ks])
                    e['wilcoxon_median_lp'] = wilcoxon([ra[k]['median_lp'] for k in ks], [rb[k]['median_lp'] for k in ks])
                # trace-level: total positives over all releases
                ka = sum(round(r['tpr_1e3'] * r['n']) for r in ca['per_replicate']); na = sum(r['n'] for r in ca['per_replicate'])
                kb = sum(round(r['tpr_1e3'] * r['n']) for r in cb['per_replicate']); nb = sum(r['n'] for r in cb['per_replicate'])
                e['trace_level'] = newcombe_diff(ka, na, kb, nb)
                res[f'{gen}|{atk}'] = e
        for key in ('wilcoxon_tpr_1e3', 'wilcoxon_median_lp'):
            items = [(k, v[key]) for k, v in res.items() if key in v]
            adj = holm([v['p'] for _, v in items])
            for (k, v), x in zip(items, adj):
                v['p_holm'] = x
        items = [(k, v['trace_level']) for k, v in res.items()]
        adj = holm([v['p'] for _, v in items])
        for (k, v), x in zip(items, adj):
            v['p_holm'] = x
        out[fam] = res
    return out


def pooled_comparisons(root):
    """Dataset-level (100-trace pooled) tests: GeoMark-KGW vs exact-hash KGW per attack; Mann-Whitney on group log10 p."""
    out = {}
    for gen in ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']:
        f = os.path.join(root, 'results', 'eval', gen, 'pooled.jsonl')
        if not os.path.exists(f):
            continue
        acc = defaultdict(list)
        for l in open(f):
            r = json.loads(l)
            if r['T'] is not None or r['m'] != 100:
                continue
            acc[(r['rid'].split('__')[0], r['attack'])].extend(r['lps'])
        for (wm, atk) in sorted(acc):
            if wm != 'gm_kgw':
                continue
            for other in ('exact_kgw', 'posthoc', 'gm_gumbel'):
                if (other, atk) in acc:
                    e = mannwhitney(clip_lp(acc[('gm_kgw', atk)]), clip_lp(acc[(other, atk)]))
                    e['frac_le_m3_x'] = float((np.asarray(acc[('gm_kgw', atk)]) <= -3).mean())
                    e['frac_le_m3_y'] = float((np.asarray(acc[(other, atk)]) <= -3).mean())
                    out[f'{gen}|gm_kgw_vs_{other}|{atk}'] = e
    adj = holm([v['p'] for v in out.values()])
    for (k, v), a in zip(out.items(), adj):
        v['p_holm'] = a
    return out


def same_trace_tests(root):
    """McNemar on the same traces: (i) attack vs none for each scheme (Porto / GeoLife small GPT, seed-0 releases, keys 0-2);
    (ii) two-resolution Fisher detector vs fine-only detector; (iii) bin-shift search (J=1) vs none."""
    th = -3.0
    out = dict(attack_vs_none={}, twores_vs_fine={}, shift_vs_plain={})
    for gen in ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9']:
        d = os.path.join(root, 'results', 'eval', gen)
        for wm in ['gm_kgw', 'gm_gumbel', 'exact_kgw', 'a_g25_d4']:
            base = {}
            for f in glob.glob(os.path.join(d, f'{wm}__k*__s0__none.parquet')):
                key = re.search(r'__k(\d+)__', f).group(1)
                base[key] = pd.read_parquet(f, columns=['i', 'lp', 'lp_f', 'lp_shift'])
            if not base:
                continue
            # (ii) / (iii) on the no-attack traces
            allb = pd.concat(base.values())
            out['twores_vs_fine'][f'{gen}|{wm}|none'] = mcnemar(allb.lp.values <= th, allb.lp_f.values <= th)
            out['shift_vs_plain'][f'{gen}|{wm}|none'] = mcnemar(allb.lp_shift.values <= th, allb.lp.values <= th)
            for atk in ['noise10', 'noise30', 'noise100', 'down2', 'down3', 'down5', 'crop25', 'crop50', 'splice30', 'retime15',
                        'retime60', 'mm', 'noise30+mm', 'mmrs', 'regen']:
                A, B, Af, Bf = [], [], [], []
                for key, b in base.items():
                    f = os.path.join(d, f'{wm}__k{key}__s0__{atk}.parquet')
                    if not os.path.exists(f):
                        continue
                    a = pd.read_parquet(f, columns=['i', 'lp', 'lp_f'])
                    m = a.merge(b, on='i', suffixes=('_a', '_b'))
                    A.append(m.lp_a.values <= th); B.append(m.lp_b.values <= th)
                    Af.append(m.lp_a.values); Bf.append(m.lp_f_a.values)
                if A:
                    out['attack_vs_none'][f'{gen}|{wm}|{atk}'] = mcnemar(np.concatenate(A), np.concatenate(B))
                    if atk in ('mm', 'noise30', 'noise30+mm'):
                        out['twores_vs_fine'][f'{gen}|{wm}|{atk}'] = mcnemar(np.concatenate(Af) <= th, np.concatenate(Bf) <= th)
    for fam in out:
        adj = holm([v['p'] for v in out[fam].values()])
        for (k, v), a in zip(out[fam].items(), adj):
            v['p_holm'] = a
    return out


def lemma3(S):
    """Lemma 3: measured surviving fine-pair fraction vs the closed-form prediction (noise attacks); geometry MC accuracy."""
    rows = []
    for k, v in S['detection'].items():
        if 'lemma3_pred_surv_f' in v and isinstance(v.get('surv_f'), list) and not math.isnan(v.get('lemma3_pred_surv_f', float('nan'))):
            gen, wm, atk = k.split('|')[:3]
            if len(k.split('|')) > 3:
                continue
            rows.append(dict(cell=k, gen=gen, wm=wm, attack=atk, pred_f=v['lemma3_pred_surv_f'], meas_f=v['surv_f'][0],
                             pred_m=v.get('lemma3_pred_surv_m'), meas_m=v['surv_m'][0]))
    df = pd.DataFrame(rows)
    out = dict(n_cells=len(df))
    if len(df):
        df['sigma'] = df.attack.str.extract(r'noise(\d+)').astype(float)
        out['frac_meas_ge_pred_f'] = float((df.meas_f >= df.pred_f - 1e-9).mean())
        out['frac_meas_ge_pred_m'] = float((df.meas_m >= df.pred_m - 1e-9).mean())
        out['by_sigma'] = {f'{int(s)}': dict(n=int(len(g)), pred_f_mean=float(g.pred_f.mean()), meas_f_mean=float(g.meas_f.mean()),
                                              gap_mean=float((g.meas_f - g.pred_f).mean()), gap_min=float((g.meas_f - g.pred_f).min()))
                           for s, g in df.groupby('sigma')}
        # KGW cells only (Gumbel-H has a different survival because of repeated-seed masking)
        kg = df[df.wm.str.contains('kgw') | df.wm.str.startswith('a_g')]
        out['kgw_by_sigma'] = {f'{int(s)}': dict(n=int(len(g)), pred_f_mean=float(g.pred_f.mean()), meas_f_mean=float(g.meas_f.mean()))
                               for s, g in kg.groupby('sigma')}
        out['rows'] = df.to_dict('records')
    geo = json.load(open(os.path.join(ROOT, 'results', 'geometry_mc.json')))
    g = pd.DataFrame(geo)
    g['rel_err'] = (g.closed_form - g.empirical) / g.empirical
    out['geometry_mc'] = dict(n=int(len(g)), max_abs_rel_err=float(g.rel_err.abs().max()),
                              median_abs_rel_err=float(g.rel_err.abs().median()),
                              worst_case_bound_holds=bool((g.dropna(subset=['worst_case_bound']).empirical <=
                                                           g.dropna(subset=['worst_case_bound']).worst_case_bound).all()),
                              by_kind={k: float(x.rel_err.abs().max()) for k, x in g.groupby('kind')})
    return out


def utility_tests(root):
    """Utility JSDs of watermarked releases vs the unwatermarked generator (Mann-Whitney over releases, Holm per generator),
    and next-location acc@1 (Welch t over 3 releases)."""
    metrics = ['jsd_length', 'jsd_gyration', 'jsd_od', 'jsd_cellpop', 'jsd_bigram', 'duplicate_trace_frac', 'distinct_trigram_ratio']
    out = {}
    for gen in ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']:
        f = os.path.join(root, 'results', 'utility', f'{gen}.jsonl')
        if not os.path.exists(f):
            continue
        recs = [json.loads(l) for l in open(f)]
        by = defaultdict(list)
        for r in recs:
            if 'wm' in r:
                by[r['wm']].append(r)
        real = [r for r in recs if r.get('key') == 'real_val']
        res = {}
        for wm in by:
            if wm == 'none':
                continue
            for m in metrics:
                x = [r[m] for r in by[wm] if m in r and r[m] is not None]
                y = [r[m] for r in by['none'] if m in r and r[m] is not None]
                if len(x) < 2 or len(y) < 2:
                    continue
                e = mannwhitney(x, y)
                e.update(mean_wm=float(np.mean(x)), mean_none=float(np.mean(y)), ratio=float(np.mean(x) / np.mean(y)) if np.mean(y) else float('nan'),
                         welch_p=float(stats.ttest_ind(x, y, equal_var=False).pvalue))
                res[f'{wm}|{m}'] = e
        # Holm within each metric across the main schemes (the family a utility claim is made about); ablations unadjusted
        main = ('exact_kgw', 'gm_kgw', 'gm_gumbel', 'a_g25_d4', 'a_g50_d4', 'posthoc', 'wtrace')
        for m in metrics:
            items = [(k, v) for k, v in res.items() if k.endswith('|' + m) and k.split('|')[0] in main]
            for key in ('p', 'welch_p'):
                adj = holm([v[key] for _, v in items])
                for (k, v), a in zip(items, adj):
                    v[key + '_holm'] = a
        out[gen] = dict(tests=res, real_val=real[0] if real else None, n_releases={k: len(v) for k, v in by.items()},
                        note='Mann-Whitney exact p (min attainable 0.009 for 9 vs 3 releases) and Welch t; Holm within metric over main schemes')
    # next-location
    nl = [json.loads(l) for l in open(os.path.join(root, 'results', 'utility', 'nextloc.jsonl'))]
    groups = defaultdict(list)
    for r in nl:
        rid = r['rid']
        wm = rid.split('__')[0]
        n = r.get('n_train')
        groups[(r['gen'], wm, n)].append(r['acc1'])
    nres = {}
    for (gen, wm, n), v in groups.items():
        if wm in ('none', 'real'):
            continue
        base = groups.get((gen, 'none', n))
        if base and len(base) >= 2 and len(v) >= 2:
            t = stats.ttest_ind(v, base, equal_var=False)
            nres[f'{gen}|{wm}|n{n}'] = dict(mean=float(np.mean(v)), mean_none=float(np.mean(base)), diff=float(np.mean(v) - np.mean(base)),
                                             n=len(v), n_none=len(base), welch_p=float(t.pvalue))
    adj = holm([v['welch_p'] for v in nres.values()])
    for (k, v), a in zip(nres.items(), adj):
        v['p_holm'] = a
    out['nextloc'] = nres
    return out


def spoofing(root):
    """P5: per-trace spoof success (lp <= -3) with Wilson CIs from 5,000 attempts; dataset-level spoof success
    (fraction of 100-trace forged pools with pooled p <= 1e-3); comparison with unmodified real traces (Fisher exact)."""
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'steal', 'results_spoof5000.jsonl'))]
    out = {}
    ref = {r['corpus']: r for r in recs if r['key'].endswith('|ref')}
    for r in recs:
        if r['key'].endswith('|ref'):
            c = r['corpus']
            rr = r['real_unmodified']
            k = round(rr['frac_lp_le_m3'] * rr['n'])
            out[f'{c}|real_unmodified'] = dict(per_trace=wilson(k, rr['n']), k=k, n=rr['n'],
                                               pooled_frac_le_m3=float((np.asarray(rr['pooled_lps']) <= -3).mean()),
                                               pooled_median=float(np.median(rr['pooled_lps'])), n_pools=len(rr['pooled_lps']))
            so = r['spoof_oracle']
            k = round(so['frac_lp_le_m3'] * so['n'])
            out[f'{c}|spoof_oracle'] = dict(per_trace=wilson(k, so['n']), k=k, n=so['n'],
                                            pooled_frac_le_m3=float((np.asarray(so['pooled_lps']) <= -3).mean()),
                                            pooled_median=float(np.median(so['pooled_lps'])))
            continue
        c, N = r['corpus'], r['N']
        for kk, sp in r['spoof'].items():
            k = round(sp['frac_lp_le_m3'] * sp['n'])
            base = ref[c]['real_unmodified']
            kb = round(base['frac_lp_le_m3'] * base['n'])
            fe = stats.fisher_exact([[k, sp['n'] - k], [kb, base['n'] - kb]], alternative='greater')
            pl = np.asarray(sp['pooled_lps'])
            out[f'{c}|N{N}|{kk}'] = dict(per_trace=wilson(k, sp['n']), k=k, n=sp['n'], fisher_vs_real_p=float(fe.pvalue),
                                         pooled_frac_le_m3=float((pl <= -3).mean()), pooled_frac_le_m10=float((pl <= -10).mean()),
                                         pooled_median=float(np.median(pl)), n_pools=int(len(pl)),
                                         pooled_wilson_le_m3=wilson(int((pl <= -3).sum()), len(pl)),
                                         n_seeds=r['accuracy']['n_seeds'], acc=r['accuracy']['acc'], frac_changed=r['spoof_frac_changed'])
        for rate in (0.25, 0.5, 1.0):
            s = r.get(f'scrub_r{rate}')
            if s:
                pl = np.asarray(s['pooled_lps'])
                out[f'{c}|N{N}|scrub_r{rate}'] = dict(frac_changed=s['frac_changed'], pooled_median=float(np.median(pl)),
                                                      pooled_frac_le_m3=float((pl <= -3).mean()), per_trace_le_m3=s['frac_lp_le_m3'])
        for rate in (0.25, 0.5, 1.0):
            s = ref[c].get(f'scrub_random_r{rate}')
            if s:
                pl = np.asarray(s['pooled_lps'])
                out[f'{c}|random_scrub_r{rate}'] = dict(frac_changed=s['frac_changed'], pooled_median=float(np.median(pl)),
                                                        pooled_frac_le_m3=float((pl <= -3).mean()), per_trace_le_m3=s['frac_lp_le_m3'])
    # k1 vs k2 at N=1e5: two-proportion test
    a, b = out.get('k1|N100000|key0'), out.get('k2|N100000|key0')
    if a and b:
        out['k1_vs_k2_N1e5'] = newcombe_diff(a['k'], a['n'], b['k'], b['n'])
    return out


def radio(S):
    """P4: rank p-values of the teacher key among 500 (200 for the power re-read) fresh keys; Fisher combination over the 3
    independent students as a secondary (population-level) statistic; resolution 1/(K+1)."""
    out = {}
    for k, v in S['radio']['keynull_by_rho'].items():
        rp = [float(x) for x in str(v['rank_p']).split(';')] if isinstance(v['rank_p'], str) else list(v['rank_p'])
        chi = -2 * np.sum(np.log(rp))
        out[k] = dict(rank_p=rp, max_rank_p=float(max(rp)), fisher_p=float(stats.chi2.sf(chi, 2 * len(rp))),
                      lp_true=v['lp_true'], min_null_ks_p=v.get('min_null_ks_p'), n_seeds=v['n_seeds'])
    return out


def shift_null(S):
    """Bin-shift search: is the null FPR at 1e-3 above nominal? (one-sided binomial over 9,000 real-trace tests)."""
    out = {}
    for k, v in S['shift'].items():
        n = v.get('null_n')
        if not n:
            continue
        kk = round(v['null_fpr_1e3'] * n)
        out[k] = dict(null_fpr_1e3=v['null_fpr_1e3'], k=kk, n=n, p_exceed=float(stats.binomtest(kk, n, 1e-3, alternative='greater').pvalue))
    adj = holm([v['p_exceed'] for v in out.values()])
    for (k, v), a in zip(out.items(), adj):
        v['p_holm'] = a
    return out


def cost(root, S):
    """Generation cost per 1k traces by scheme (median over releases) and relative overhead vs unwatermarked."""
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'gen', 'timing.jsonl'))]
    by = defaultdict(list)
    for r in recs:
        if r.get('n') and r['n'] >= 2000:   # exclude 200-trace many-key releases (fixed start-up overhead dominates)
            by[(r['gen'], r['rid'].split('__')[0])].append(1000 * r['sec'] / r['n'])
    out = {}
    for (gen, wm), v in sorted(by.items()):
        base = by.get((gen, 'none'))
        out[f'{gen}|{wm}'] = dict(n_rel=len(v), sec_per_1k_median=float(np.median(v)),
                                  rel_to_none=float(np.median(v) / np.median(base)) if base else None)
    out['_training'] = S['timing']['training']
    return out


def gumbel_duplicates(root):
    """Diagnostic for the Gumbel-H key effect: a single-key Gumbel-H release is dominated by repeated traces (the sampler is
    deterministic given key + coarse context). Gumbel-H p-values are continuous, so identical detection signatures
    (lp, lp_f, lp_m, n_f, n_m, len_tok) identify repeated traces. Reports per release: duplicate fraction and per-trace TPR
    over all traces vs over distinct traces. (Not applied to KGW: binomial p-values collide for distinct traces.)"""
    out = {}
    for gen in ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']:
        for f in sorted(glob.glob(os.path.join(root, 'results', 'eval', gen, 'gm_gumbel__k*__s*__none.parquet'))):
            m = re.search(r'gm_gumbel__k(\d+)__s(\d+)__none\.parquet$', f)
            if not m:                       # decode variants (road / centroid)
                continue
            x = pd.read_parquet(f)
            if len(x) < 1000:                # 200-trace many-key releases
                continue
            sig = x[['lp', 'lp_f', 'lp_m', 'n_f', 'n_m', 'len_tok']].round(9)
            u = x.loc[~sig.duplicated()]
            top = sig.value_counts().iloc[0] / len(x)
            out[f'{gen}|k{m.group(1)}|s{m.group(2)}'] = dict(n=int(len(x)), dup_frac=float(1 - len(u) / len(x)), n_distinct=int(len(u)),
                                                           top_trace_share=float(top), tpr_1e3_all=float((x.lp <= -3).mean()),
                                                           tpr_1e3_distinct=float((u.lp <= -3).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.root, 'results', 'summary.json')))
    R = dict(generated_by='src/stats_tests.py', source='results/summary.json + raw per-trace files')
    for name, fn in [('calibration', lambda: calibration(S)), ('keycond_fpr', lambda: keycond_fpr(a.root, S)),
                     ('key_effect', lambda: key_effect(a.root, S)), ('release_comparisons', lambda: release_comparisons(S)),
                     ('pooled_comparisons', lambda: pooled_comparisons(a.root)), ('same_trace', lambda: same_trace_tests(a.root)),
                     ('lemma3', lambda: lemma3(S)), ('utility', lambda: utility_tests(a.root)), ('spoofing', lambda: spoofing(a.root)),
                     ('radio', lambda: radio(S)), ('shift_null', lambda: shift_null(S)), ('gumbel_duplicates', lambda: gumbel_duplicates(a.root)), ('cost', lambda: cost(a.root, S))]:
        R[name] = fn()
        print('done', name, flush=True)
    path = os.path.join(a.root, 'results', 'stats_tests.json')

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(R, f, indent=1, default=conv)
    print('wrote', path)


if __name__ == '__main__':
    main()
