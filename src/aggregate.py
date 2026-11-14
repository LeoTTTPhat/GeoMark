"""Aggregate all raw outputs under <out_dir> into <out_dir>/summary.json (+ CSV tables in <out_dir>/tables/).
Every number reported in the paper must come from this file or the raw files it reads.
Usage: python src/aggregate.py --cfg CFG"""
import os, sys, glob, json, argparse, re, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, dump_json, read_jsonl, p
from geomark.stats import tpr_at, auc, mean_ci_t, boot_ci

RID = re.compile(r'^(?P<wm>.+?)__k(?P<key>\d+)__s(?P<seed>\d+)(?:__(?P<dec>road|centroid))?__(?P<attack>[a-z0-9+]+)(?:__T(?P<T>.+))?\.parquet$')


def lemma3_pred(sigma, res, k, coarse_off=2):
    import h3
    q = lambda r: min(1.0, 0.921 * sigma / h3.average_hexagon_edge_length(r, 'm'))
    return (1 - q(res)) * (1 - q(res - coarse_off)) ** (k + 1), (1 - q(res - 1)) * (1 - q(res - coarse_off)) ** (k + 1)


def per_file_metrics(df, null_df):
    lp = df['lp'].values
    m = dict(n=int(len(df)), tpr_1e3=tpr_at(lp, 1e-3), tpr_1e5=tpr_at(lp, 1e-5), tpr_1e2=tpr_at(lp, 1e-2),
             median_lp=float(np.median(lp)), median_n_f=float(df['n_f'].median()), median_n_m=float(df['n_m'].median()),
             tpr_1e3_fine_only=tpr_at(df['lp_f'].values, 1e-3))
    if 'lp_shift' in df:
        m['tpr_1e3_shift'] = tpr_at(df['lp_shift'].values, 1e-3)
    if 'n_f0' in df:
        sel = df['n_f0'] >= 20
        m['tpr_1e3_ge20pairs'] = tpr_at(lp[sel.values], 1e-3) if sel.any() else float('nan')
        m['frac_ge20pairs'] = float(sel.mean())
        m['surv_f'] = float(np.nanmean(df['surv_f'])) if 'surv_f' in df else float('nan')
        m['surv_m'] = float(np.nanmean(df['surv_m'])) if 'surv_m' in df else float('nan')
    if 'failed' in df:
        m['failed_frac'] = float(df['failed'].fillna(False).mean())
    if null_df is not None:
        m['auc'] = auc(-lp, -null_df['lp'].values)
        m['null_fpr_1e3'] = tpr_at(null_df['lp'].values, 1e-3)
    return m


PROP_COLS = ('tpr_1e3', 'tpr_1e5', 'tpr_1e2', 'tpr_1e3_fine_only', 'tpr_1e3_shift', 'tpr_1e3_ge20pairs', 'surv_f', 'surv_m', 'auc',
             'null_fpr_1e3', 'failed_frac', 'frac_ge20pairs')


def det_signature(cfg, name, res):
    """Everything the DETECTOR depends on (embedding strengths delta / delta_mid do not matter); posthoc == kgw."""
    if name not in cfg['wm']:
        return None
    from geomark.scheme import Scheme
    sc = Scheme.from_dict(dict(cfg['wm'][name], res=res))
    cls = 'exact' if sc.method == 'exact_kgw' else sc.score
    return (cls, sc.gamma if sc.score == 'kgw' else None, sc.k, sc.bin_min, sc.ctx, sc.mask, sc.uses_mid, sc.coarse_off, res)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float('nan')] * 3
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [ph, max(0.0, c - h), min(1.0, c + h)]


def clip01(ci):
    """Stage 4b: t-interval of a proportion clipped to [0, 1]."""
    f = lambda v, op, b: v if math.isnan(v) else op(b, v)
    return ci if math.isnan(ci[0]) else [ci[0], f(ci[1], max, 0.0), f(ci[2], min, 1.0)]


def eval_section(cfg):
    out, rows = {}, []
    root = os.path.join(cfg['out_dir'], 'eval')
    nulls = {}
    for f in glob.glob(os.path.join(root, 'real', '*', '*.parquet')):
        dsr = os.path.basename(os.path.dirname(f))
        nulls[(dsr, os.path.basename(f)[:-8])] = f
    # Stage 4: a release whose detector equals that of a null scheme (e.g. a_g25_d4 == gm_kgw, posthoc == gm_kgw) uses that
    # scheme's real-trace null (exact same p-value computation)
    null_schemes = sorted({k[1].split('__k')[0] for k in nulls})
    _eq = {}

    def null_name(sname, res):
        if (sname, res) not in _eq:
            sig = det_signature(cfg, sname, res)
            cands = [sname] if sname in null_schemes else [x for x in null_schemes if sig is not None and det_signature(cfg, x, res) == sig]
            _eq[(sname, res)] = cands[0] if cands else sname
        return _eq[(sname, res)]
    for gdir in sorted(glob.glob(os.path.join(root, '*'))):
        gen = os.path.basename(gdir)
        if gen == 'real' or gen not in cfg['generators']:
            continue
        g = cfg['generators'][gen]
        dsr = f"{g['dataset']}_r{g['res']}"
        for f in sorted(glob.glob(os.path.join(gdir, '*.parquet'))):
            m = RID.match(os.path.basename(f))
            if not m:
                continue
            d = m.groupdict()
            df = pd.read_parquet(f)
            sname = null_name(d['T'] or d['wm'], g['res'])
            nf = nulls.get((dsr, f"{sname}__k{d['key'] if not d['T'] else 0}__{d['attack']}"))
            met = per_file_metrics(df, pd.read_parquet(nf) if nf else None)
            met['k_1e3'] = int((df['lp'].values <= -3).sum()); met['k_1e5'] = int((df['lp'].values <= -5).sum())
            rows.append(dict(gen=gen, wm=d['wm'], key=int(d['key']), seed=int(d['seed']), decode=d['dec'] or 'uniform',
                             attack=d['attack'], T=d['T'], null_scheme=sname if nf else None, **met))
    if not rows:
        return {}, pd.DataFrame()
    R = pd.DataFrame(rows)
    agg = {}
    for (gen, wm, dec, attack, T), grp in R.groupby(['gen', 'wm', 'decode', 'attack', R['T'].fillna('-')]):
        k = f'{gen}|{wm}' + (f'|{dec}' if dec != 'uniform' else '') + f'|{attack}' + (f'|T{T}' if T != '-' else '')
        e = dict(n_files=len(grp), n_traces=int(grp['n'].sum()))
        for col in ['tpr_1e3', 'tpr_1e5', 'tpr_1e2', 'median_lp', 'median_n_f', 'tpr_1e3_fine_only', 'tpr_1e3_shift', 'tpr_1e3_ge20pairs',
                    'surv_f', 'surv_m', 'auc', 'null_fpr_1e3', 'failed_frac', 'frac_ge20pairs']:
            if col in grp:
                e[col] = mean_ci_t(grp[col].values)
                if col in PROP_COLS:          # Stage 4: t-intervals of proportions clipped to [0, 1]
                    e[col] = clip01(e[col])
        # Stage 4: pooled over all traces of the cell (Wilson CI; ignores clustering by release, so it complements the t-CI)
        e['tpr_1e3_alltraces'] = wilson(int(grp['k_1e3'].sum()), int(grp['n'].sum()))
        e['tpr_1e5_alltraces'] = wilson(int(grp['k_1e5'].sum()), int(grp['n'].sum()))
        e['null_scheme'] = grp['null_scheme'].dropna().iloc[0] if grp['null_scheme'].notna().any() else None
        # Stage 4b: the release is not the independent unit when keys repeat across sampling seeds (key effect dominates for
        # Gumbel-H): key-level interval = t-interval over per-key means (n_keys units). Single-release cells get NaN
        # t-interval bounds (no replicate; use *_alltraces Wilson intervals instead).
        e['n_keys'] = int(grp['key'].nunique())
        km = grp.groupby('key')
        for col in ('tpr_1e3', 'tpr_1e5', 'median_lp', 'auc'):
            if col in grp:
                ci = mean_ci_t(km[col].mean().values)
                e[col + '_keylevel'] = clip01(ci) if col != 'median_lp' else ci
        if len(grp) == 1:
            for col in list(e):
                if isinstance(e[col], list) and len(e[col]) == 3 and not col.endswith('_alltraces'):
                    e[col] = [e[col][0], float('nan'), float('nan')]
            e['ci_note'] = 'single release: no t-interval; see tpr_1e3_alltraces / tpr_1e5_alltraces (Wilson, traces as units)'
        rep_cols = [c for c in ['key', 'seed', 'n', 'tpr_1e3', 'tpr_1e5', 'median_lp', 'median_n_f', 'tpr_1e3_ge20pairs', 'auc', 'null_fpr_1e3']
                    if c in grp]
        e['per_replicate'] = [{c: (None if isinstance(v, float) and math.isnan(v) else v) for c, v in r.items()}
                              for r in grp[rep_cols].to_dict('records')]   # one entry per (key, seed) release
        sch = cfg['wm'].get(wm, {})
        mm = re.fullmatch(r'noise(\d+)', attack)
        if mm and sch.get('method') not in (None, 'none', 'exact_kgw'):
            pf, pm = lemma3_pred(int(mm.group(1)), cfg['generators'][gen]['res'], sch.get('k', 1))
            e['lemma3_pred_surv_f'], e['lemma3_pred_surv_m'] = pf, pm
        agg[k] = e
    return agg, R


def tpr_vs_pairs(cfg, gens, wms, attacks=('none', 'mm', 'noise30')):
    out = {}
    edges = [0, 5, 10, 15, 20, 25, 30, 40, 60, 1000]
    for gen in gens:
        for wm in wms:
            for atk in attacks:
                fs = glob.glob(os.path.join(cfg['out_dir'], 'eval', gen, f'{wm}__k*__s*__{atk}.parquet'))
                if not fs:
                    continue
                df = pd.concat([pd.read_parquet(f) for f in fs])
                bins = pd.cut(df['n_f'], edges, right=False)
                out[f'{gen}|{wm}|{atk}'] = {str(b): dict(n=int(len(x)), tpr_1e3=tpr_at(x['lp'], 1e-3), tpr_1e5=tpr_at(x['lp'], 1e-5))
                                           for b, x in df.groupby(bins, observed=True)}
    return out


def tpr_vs_length(cfg, gens, wms, attacks=('none',)):
    """Stage 4: TPR vs RAW trace length (points) and collapsed fine tokens (PLAN Sec. 6 metric (a))."""
    out = {}
    e_pts = [0, 10, 20, 40, 60, 100, 150, 200, 10000]
    e_tok = [0, 5, 10, 15, 20, 30, 40, 60, 1000]
    for gen in gens:
        for wm in wms:
            for atk in attacks:
                fs = glob.glob(os.path.join(cfg['out_dir'], 'eval', gen, f'{wm}__k*__s*__{atk}.parquet'))
                if not fs:
                    continue
                df = pd.concat([pd.read_parquet(f, columns=['lp', 'len_pts', 'len_tok']) for f in fs])
                for col, edges in (('len_pts', e_pts), ('len_tok', e_tok)):
                    bins = pd.cut(df[col], edges, right=False)
                    out[f'{gen}|{wm}|{atk}|{col}'] = {str(b): dict(n=int(len(x)), tpr_1e3=tpr_at(x['lp'], 1e-3), tpr_1e5=tpr_at(x['lp'], 1e-5))
                                                     for b, x in df.groupby(bins, observed=True)}
    return out


def shift_section(cfg):
    """Stage 4: bin-shift radius J vs re-timestamping (results/eval_shift/)."""
    out = {}
    nulls = {}
    for f in glob.glob(os.path.join(cfg['out_dir'], 'eval_shift', 'real_*.jsonl')):
        dsr = os.path.basename(f)[5:-6]
        for r in read_jsonl(f):
            wm = r['tag'].split('/')[1].split('__k')[0]
            for J in [k for k in r if re.fullmatch(r'J\d+', k)]:
                nulls.setdefault((dsr, wm, r['attack'], J), []).extend(r[J])
    for f in glob.glob(os.path.join(cfg['out_dir'], 'eval_shift', '*.jsonl')):
        gen = os.path.basename(f)[:-6]
        if gen.startswith('real_') or gen not in cfg['generators']:
            continue
        g = cfg['generators'][gen]
        dsr = f"{g['dataset']}_r{g['res']}"
        acc = {}
        for r in read_jsonl(f):
            wm = r['tag'].split('/')[1].split('__k')[0]
            for J in [k for k in r if re.fullmatch(r'J\d+', k)]:
                acc.setdefault((wm, r['attack'], J), []).append(tpr_at(r[J], 1e-3))
        for (wm, atk, J), v in acc.items():
            nl = nulls.get((dsr, wm, atk, J), [])
            out[f'{gen}|{wm}|{atk}|{J}'] = dict(n_releases=len(v), tpr_1e3=clip01(mean_ci_t(v)),   # Stage 4b: clipped per_key=';'.join(f'{x:.4f}' for x in v),
                                                null_n=len(nl), null_fpr_1e3=tpr_at(nl, 1e-3) if nl else None,
                                                null_fpr_1e2=tpr_at(nl, 1e-2) if nl else None)
    return out


def pooled_section(cfg):
    out = {}
    for f in glob.glob(os.path.join(cfg['out_dir'], 'eval', '*', 'pooled.jsonl')):
        gen = os.path.basename(os.path.dirname(f))
        acc = {}
        for r in read_jsonl(f):
            wm = r['rid'].split('__')[0]
            acc.setdefault(f"{gen}|{wm}|{r['attack']}|{r['T'] or '-'}|m{r['m']}", []).extend(r['lps'])
        for k, v in acc.items():
            v = np.array(v)
            out[k] = dict(n_groups=len(v), median_lp=float(np.median(v)), frac_le_m3=float((v <= -3).mean()), frac_le_m10=float((v <= -10).mean()),
                          max_lp=float(v.max()))
    return out


def json_dir(path_glob):
    out = {}
    for f in sorted(glob.glob(path_glob)):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def utility_section(cfg):
    out = {}
    for f in glob.glob(os.path.join(cfg['out_dir'], 'utility', '*.jsonl')):
        name = os.path.basename(f)[:-6]
        recs = read_jsonl(f)
        if name == 'nextloc':
            out['nextloc'] = {r['key']: r for r in recs}
            # Stage 4: mean / 95% t-CI over replicate releases per (gen, wm, training cap)
            grp = {}
            for r in recs:
                wm = r['rid'].split('__')[0]
                nb = int(round(r['n_train'] / 1000.0)) * 1000      # training-set size bucket (e.g. 1,933 -> 2000; 9,990 -> 10000)
                grp.setdefault(f"{r['gen']}|{wm}|n{nb}", []).append(r)
            out['nextloc_summary'] = {k: dict(n_rep=len(v), acc1=mean_ci_t([x['acc1'] for x in v]), acc5=mean_ci_t([x['acc5'] for x in v]),
                                              n_train=';'.join(str(x['n_train']) for x in v), rids=';'.join(x['rid'] for x in v)) for k, v in grp.items()}
            continue
        if name == 'keymix':                   # Stage 4: key-averaged vs single-key distributions (run_keymix.py)
            df = pd.DataFrame(recs)
            out['keymix'] = {f'{g}|{w}|{k}': dict(n_rep=len(x), n_traces=int(x['n'].iloc[0]),
                                                 **{c: mean_ci_t(x[c].values) for c in x.columns if c.startswith(('jsd_', 'distinct_', 'duplicate_', 'memorised', 'mean_len'))})
                             for (g, w, k), x in df.groupby(['gen', 'wm', 'kind'])}
            continue
        df = pd.DataFrame(recs)
        res = {}
        if 'wm' in df:
            for wm, grp in df[df['key'] != 'real_val'].groupby('wm'):
                res[wm] = {c: mean_ci_t(grp[c].values) for c in grp.columns if c.startswith(('jsd_', 'distinct_', 'duplicate_', 'memorised', 'nn_', 'mean_', 'posthoc_'))}
        rv = df[df['key'] == 'real_val']
        if len(rv):
            res['real_val'] = {c: float(rv[c].iloc[0]) for c in rv.columns if c != 'key' and not isinstance(rv[c].iloc[0], str)}
        out[name] = res
    return out


def wtrace_section(cfg):
    out = {}
    for f in glob.glob(os.path.join(cfg['out_dir'], 'baselines', 'wtrace_*.parquet')):
        df = pd.read_parquet(f)
        for atk, grp in df.groupby('attack'):
            null = np.sort(grp['ncc_null'].values)[::-1]
            r = dict(n=len(grp), mean_ncc=float(grp['ncc'].mean()), tpr_tau085=float((grp['ncc'] > 0.85).mean()),
                     fpr_tau085=float((grp['ncc_null'] > 0.85).mean()), auc=auc(grp['ncc'].values, grp['ncc_null'].values))
            for a in (1e-2, 1e-3):
                k = max(1, int(a * len(null)))
                r[f'tpr_at_emp_fpr{a:g}'] = float((grp['ncc'].values > null[k - 1]).mean())
            out.setdefault(os.path.basename(f)[:-8], {})[atk] = r
    return out


def steal_section(cfg, fname='results.jsonl'):
    f = os.path.join(cfg['out_dir'], 'steal', fname)
    if not os.path.exists(f):
        return {}
    recs = read_jsonl(f)
    out = {}
    for r in recs:
        r = dict(r)
        for k, v in list(r.items()):
            if isinstance(v, dict) and 'pooled_lps' in v:
                v['pooled_median_lp'] = float(np.median(v['pooled_lps'])) if v['pooled_lps'] else None
                v.pop('pooled_lps')
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and 'pooled_lps' in vv:
                        vv['pooled_median_lp'] = float(np.median(vv['pooled_lps'])) if vv['pooled_lps'] else None
                        vv.pop('pooled_lps')
        out[r['key']] = r
    return out


def radio_section(cfg):
    recs = read_jsonl(os.path.join(cfg['out_dir'], 'radio', 'results.jsonl'))
    out = {r['key']: r for r in recs}
    rc = cfg.get('radio', {})
    agg = {}
    for r in recs:
        agg.setdefault(f"{r['wm']}|rho{r['rho']}", []).append(r)
    summ = {}
    for k, v in agg.items():
        key = rc.get('key', 0); wk = rc.get('wrong_key', 1)
        summ[k] = dict(n=len(v), read_lp=[x[f'read_k{key}']['lp'] for x in v], read_lp_wrongkey=[x[f'read_k{wk}']['lp'] for x in v],
                       sample_lp=[x[f'sample_k{key}']['lp'] for x in v], sample_lp_wrongkey=[x[f'sample_k{wk}']['lp'] for x in v])
    kn = {}
    for r in read_jsonl(os.path.join(cfg['out_dir'], 'radio', 'keynull.jsonl')):
        kn[r['key']] = {**{c: r[c] for c in ('student', 'wm', 'rho', 'seed', 'n_f', 'n_m', 'null_keys')},
                        **{m: {c: v for c, v in r[m].items() if c != 'null_lps'} for m in ('read', 'sample')}}
    kp = {}
    pf = os.path.join(cfg['out_dir'], 'radio', 'keynull_power.jsonl')
    for r in (read_jsonl(pf) if os.path.exists(pf) else []):    # Stage 4: more contexts / samples
        kp[r['key']] = {**{c: r.get(c) for c in ('student', 'wm', 'rho', 'seed', 'n_f', 'n_m', 'null_keys', 'n_contexts', 'n_sample')},
                        **{m: {c: v for c, v in r[m].items() if c != 'null_lps'} for m in ('read', 'sample')}}
    # mean / CI of the teacher-key log10 p and rank p over student seeds
    by = {}
    for src, d in (('keynull', kn), ('keynull_power', kp)):
        for r in d.values():
            for m in ('read', 'sample'):
                by.setdefault(f"{src}|{r['wm']}|rho{r['rho']}|{m}", []).append(r[m])
    by = {k: dict(n_seeds=len(v), lp_true=mean_ci_t([x['lp_true'] for x in v]), rank_p=';'.join(f"{x['rank_p']:.4f}" for x in v),
                  max_rank_p=max(x['rank_p'] for x in v), min_null_ks_p=min(x['null_ks_p'] for x in v)) for k, v in by.items()}
    return dict(per_student=out, by_method_rho=summ, keynull=kn, keynull_power=kp, keynull_by_rho=by)


def keyvar_section(cfg):
    """Stage 4b: per-key TPR distribution over many keys and key-conditional null FPR (results/keyvar/, run_keyvar.py)."""
    out = dict(tpr={}, null={})
    tp = read_jsonl(os.path.join(cfg['out_dir'], 'keyvar', 'tpr.jsonl')) if os.path.exists(
        os.path.join(cfg['out_dir'], 'keyvar', 'tpr.jsonl')) else []
    acc = {}
    for r in tp:
        acc.setdefault((r['gen'], r['wm'], r['attack']), []).append(r)
    for (gen, wm, atk), rs in sorted(acc.items()):
        t = np.array([r['tpr_1e3'] for r in rs]); n = np.array([r['n'] for r in rs])
        plp = np.array([x for r in rs for x in r['pooled_lps']])
        sd = float(t.std(ddof=1)) if len(t) > 1 else float('nan')
        # binomial within-key variance vs observed between-key variance (excess = key effect)
        binom_var = float(np.mean(t * (1 - t) / n))
        out['tpr'][f'{gen}|{wm}|{atk}'] = dict(
            n_keys=len(rs), n_per_key=int(np.median(n)), tpr_1e3=clip01(mean_ci_t(t)), tpr_1e3_sd_keys=sd,
            tpr_1e3_q=[float(np.quantile(t, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)],
            between_key_var=float(t.var(ddof=1)) if len(t) > 1 else float('nan'), within_key_binom_var=binom_var,
            tpr_1e2=clip01(mean_ci_t([r['tpr_1e2'] for r in rs])), median_lp=mean_ci_t([r['median_lp'] for r in rs]),
            pooled_m=rs[0]['pooled_m'], pooled_n=int(len(plp)),
            pooled_median_lp=float(np.median(plp)) if len(plp) else None,
            pooled_frac_le_m3=float((plp <= -3).mean()) if len(plp) else None,
            pooled_frac_le_m10=float((plp <= -10).mean()) if len(plp) else None,
            per_key=';'.join(f"{r['wm_key']}:{r['tpr_1e3']:.4f}" for r in sorted(rs, key=lambda r: r['wm_key'])))
    nf = os.path.join(cfg['out_dir'], 'keyvar', 'null.jsonl')
    nl = read_jsonl(nf) if os.path.exists(nf) else []
    acc = {}
    for r in nl:
        acc.setdefault((r['cell'], r['scheme']), []).append(r)
    from scipy import stats as _st
    for (cell, sch), rs in sorted(acc.items()):
        n = rs[0]['n']; e = dict(n_keys=len(rs), n_traces_per_key=n)
        for t, alpha in ((2, 1e-2), (3, 1e-3), (4, 1e-4)):
            k = np.array([r[f'k_le_m{t}'] for r in rs])
            f = k / n
            # dispersion test: are per-key counts more variable than Binomial(n, mean)? (chi^2 index of dispersion)
            pm = k.sum() / (n * len(k))
            disp = float(((k - n * pm) ** 2).sum() / (n * pm * (1 - pm))) if 0 < pm < 1 else float('nan')
            e[f'fpr_{alpha:g}'] = dict(mean=float(f.mean()), pooled_wilson=wilson(int(k.sum()), int(n * len(k))),
                                       sd_keys=float(f.std(ddof=1)), q90=float(np.quantile(f, 0.9)), max=float(f.max()),
                                       frac_keys_gt_2x=float((f > 2 * alpha).mean()),
                                       binom_sd=float(np.sqrt(alpha * (1 - alpha) / n)),
                                       dispersion_chi2=disp, dispersion_df=len(k) - 1,
                                       dispersion_p=float(_st.chi2.sf(disp, len(k) - 1)) if not math.isnan(disp) else None)
        npl = rs[0]['n_pooled']
        for t, alpha in ((2, 1e-2), (3, 1e-3)):
            k = np.array([r['pooled'][f'k_le_m{t}'] for r in rs])
            e[f'pooled_fpr_{alpha:g}'] = dict(mean=float(k.sum() / (npl * len(k))), pooled_wilson=wilson(int(k.sum()), int(npl * len(k))),
                                              max=float((k / npl).max()), n_tests=int(npl * len(k)))
        out['null'][f'{cell}|{sch}'] = e
    return out


def timing_section(cfg):
    out = {}
    for f in glob.glob(os.path.join(cfg['out_dir'], 'gen', 'timing.jsonl')):
        df = pd.DataFrame(read_jsonl(f))
        if len(df):
            df['sec_per_1k'] = df['sec'] / df['n'] * 1000
            out['generation_sec_per_1k'] = df.groupby('gen')['sec_per_1k'].median().to_dict()
    tr = {}
    for f in glob.glob(os.path.join(cfg['model_dir'], '*', 'done.json')):
        d = json.load(open(f))
        tr[os.path.basename(os.path.dirname(f))] = dict(sec=d.get('sec'), n_train=d.get('n_train'), val_loss=d.get('val_loss', d.get('history', [{}])[-1].get('val_loss') if d.get('history') else None),
                                                        n_params=d.get('n_params'))
    out['training'] = tr
    return out


def flat(d, prefix=''):
    """Flatten nested dicts into {a.b.c: scalar}; lists of 3 numbers (mean, lo, hi) become _mean/_lo/_hi columns."""
    out = {}
    for k, v in d.items():
        kk = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flat(v, kk + '.'))
        elif isinstance(v, list):
            if len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                out[kk + '_mean'], out[kk + '_lo'], out[kk + '_hi'] = v
            elif len(v) <= 8 and all(isinstance(x, (int, float)) for x in v):
                out[kk] = ';'.join(f'{x:.6g}' for x in v)
        else:
            out[kk] = v
    return out


def write_tables(S, td):
    """Secondary CSV tables (one row per cell) derived only from the summary dict."""
    def dump(name, rows):
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(td, f'{name}.csv'), index=False)
    dump('calibration', [{'cell': k, **flat(v)} for k, v in S['calibration'].items() if isinstance(v, dict)])
    dump('pooled', [{'cell': k, **v} for k, v in S['pooled'].items()])
    U = S['utility']
    dump('utility', [{'gen': g, 'wm': w, **{c: x for c, x in flat(m).items() if c not in ('gen', 'wm')}} for g, d in U.items() if g != 'nextloc' for w, m in d.items()])
    dump('nextloc', [flat(v) for v in U.get('nextloc', {}).values()])
    dump('wtrace', [{'release': r, 'attack': a, **v} for r, d in S['baselines']['wtrace'].items() for a, v in d.items()])
    dump('classifier', [{'gen': g, 'wm': w, **{c: x for c, x in flat(v).items() if c not in ('gen', 'wm')}} for g, d in S['baselines']['classifier'].items() for w, v in d.items()
                        if isinstance(v, dict)])
    dump('steal', [flat(v) for v in S['steal'].values()])
    dump('radio', [flat(v) for v in S['radio']['per_student'].values()])
    dump('radio_keynull', [{'key': k, **flat(v)} for k, v in S['radio'].get('keynull', {}).items()])
    dump('calibration_ext', [{'cell': k, **flat(v)} for k, v in S.get('calibration_ext', {}).items() if isinstance(v, dict)])
    dump('shift', [{'cell': k, **flat(v)} for k, v in S.get('shift', {}).items()])
    dump('keymix', [{'cell': k, **flat(v)} for k, v in U.get('keymix', {}).items()])
    dump('nextloc_summary', [{'cell': k, **flat({c: x for c, x in v.items() if c not in ('n_train', 'rids')})} for k, v in U.get('nextloc_summary', {}).items()])
    dump('radio_power', [{'key': k, **flat(v)} for k, v in S['radio'].get('keynull_power', {}).items()])
    dump('keyvar_tpr', [{'cell': k, **flat(v)} for k, v in S.get('keyvar', {}).get('tpr', {}).items()])      # Stage 4b
    dump('keyvar_null', [{'cell': k, **flat(v)} for k, v in S.get('keyvar', {}).get('null', {}).items()])    # Stage 4b
    dump('radio_keynull_by_rho', [{'cell': k, **flat(v)} for k, v in S['radio'].get('keynull_by_rho', {}).items()])
    dump('steal_spoof5000', [flat(v) for v in S.get('steal_spoof5000', {}).values()])
    dump('tpr_vs_length', [{'cell': k, 'bin': b, **v} for k, d in S.get('tpr_vs_length', {}).items() for b, v in d.items()])
    tv = S['tpr_vs_pairs']
    dump('tpr_vs_pairs', [{'cell': k, 'bin': b, **v} for k, d in tv.items() for b, v in d.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    S = dict(config=cfg.get('name'), generated_by='src/aggregate.py')
    S['prep'] = json_dir(os.path.join(cfg['out_dir'], 'prep', '*.json'))
    ev, R = eval_section(cfg)
    S['detection'] = ev
    main_gens = [g for g in cfg['generators'] if g in cfg['releases'].get('main', {}).get('gens', []) + cfg['releases'].get('llm', {}).get('gens', [])]
    S['tpr_vs_pairs'] = tpr_vs_pairs(cfg, main_gens, ['gm_kgw', 'gm_gumbel', 'exact_kgw', 'posthoc', 'a_g25_d4', 'a_g50_d4'])
    S['pooled'] = pooled_section(cfg)
    S['calibration'] = json_dir(os.path.join(cfg['out_dir'], 'calibration', '*.json'))
    S['calibration_ext'] = json_dir(os.path.join(cfg['out_dir'], 'calibration_ext', '*.json'))   # Stage 4
    S['shift'] = shift_section(cfg)                                                                 # Stage 4
    S['tpr_vs_length'] = tpr_vs_length(cfg, main_gens, ['gm_kgw', 'gm_gumbel', 'exact_kgw', 'a_g25_d4'])  # Stage 4
    S['baselines'] = dict(wtrace=wtrace_section(cfg), classifier=json_dir(os.path.join(cfg['out_dir'], 'baselines', 'classifier_*.json')))
    S['utility'] = utility_section(cfg)
    S['steal'] = steal_section(cfg)
    S['steal_spoof5000'] = steal_section(cfg, 'results_spoof5000.jsonl')     # Stage 4: 10x more spoof attempts
    S['radio'] = radio_section(cfg)
    cap = os.path.join(cfg['out_dir'], 'capacity', 'capacity.json')
    S['capacity'] = json.load(open(cap)) if os.path.exists(cap) else {}
    S['mapmatch'] = read_jsonl(os.path.join(cfg['out_dir'], 'attacked', 'mm_log.jsonl'))
    for name in ('geometry_mc', 'seedspace', 'data_stats'):
        f = p('results', f'{name}.json')
        if os.path.exists(f):
            S[f'stage1_{name}'] = f'results/{name}.json'
    S['keyvar'] = keyvar_section(cfg)                                                              # Stage 4b
    S['timing'] = timing_section(cfg)
    dump_json(S, os.path.join(cfg['out_dir'], 'summary.json'))
    td = os.path.join(cfg['out_dir'], 'tables'); os.makedirs(td, exist_ok=True)
    if len(R):
        R.to_csv(os.path.join(td, 'detection_per_release.csv'), index=False)
        rows = [dict(key=k, **flat(e)) for k, e in ev.items()]   # mean, 95% t-CI over keys x seeds
        pd.DataFrame(rows).to_csv(os.path.join(td, 'detection_summary.csv'), index=False)
    write_tables(S, td)
    print(f"summary: {len(ev)} detection cells, {len(S['calibration'])} calibration files, {len(S['pooled'])} pooled cells, "
          f"utility {list(S['utility'])}, steal {len(S['steal'])}, radio {len(S['radio']['per_student'])}")


if __name__ == '__main__':
    main()
