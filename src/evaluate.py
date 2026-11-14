"""E2/E3/E4 detection under attacks. For every release of the given sets and every attack of the set's attack list,
re-tokenise the (attacked) traces, detect with the release key and scheme, and write per-trace rows:
  <out_dir>/eval/<gen>/<rid>__<attack>[__T<scheme>].parquet
Dataset-level pooled tests (release sizes m, and mixing with real traces) go to <out_dir>/eval/<gen>/pooled.jsonl.
Attack nulls (real test traces, keys 0..2): <out_dir>/eval/real/<ds>_r<res>/<scheme>__k<key>__<attack>.parquet.
Everything is resumable (existing outputs are skipped). Usage:
  python src/evaluate.py --cfg CFG --sets main,llm [--gens g1,g2] [--nulls]"""
import os, sys, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, scaled, seed_from, load_pickle, append_jsonl, read_jsonl, Timer
from geomark.scheme import Scheme
from geomark.prf import PRF
from geomark.detect import detect_trace, pair_sets, survival, pooled, trace_cells, score_pairs
from geomark import attacks
from geomark.data import release_dir, real_traces
from generate import release_jobs, wm_scheme

CFG = None


def _init(cfg_path):
    global CFG
    CFG = load_cfg(cfg_path)


def load_cache(cfg, gen, rid, prefix):
    if gen.startswith('real_'):
        f = os.path.join(cfg['out_dir'], 'attacked', gen, f'test__{prefix}.pkl')
    else:
        f = os.path.join(cfg['out_dir'], 'attacked', gen, f'{rid}__{prefix}.pkl')
    return load_pickle(f)['traces'] if os.path.exists(f) else None


def eval_traces(traces, attack, sch, key, J, tag, cfg, gen, rid, pool_real, n_max):
    """Per-trace detection rows for one (trace list, attack, scheme, key)."""
    prf = PRF(key)
    pre = attacks.cache_prefix(attack)
    cached = load_cache(cfg, gen, rid, pre) if pre else None
    if pre and cached is None:
        return None, None
    n = min(n_max, len(traces), len(cached) if cached is not None else 10 ** 9)
    rows, pairs = [], []
    for i in range(n):
        tr = traces[i]
        c = {pre: cached[i]} if cached is not None else None
        if c is not None and (c[pre] is None or len(c[pre]['lat']) == 0):
            rows.append(dict(i=i, failed=True, lp=0.0, lp_f=0.0, lp_m=0.0, n_f=0, n_m=0, lp_shift=0.0)); pairs.append(([], []))
            continue
        x = attacks.apply(tr, attack, np.random.default_rng(seed_from(tag, attack, i)), pool=pool_real, cached=c)
        d = detect_trace(x, sch, prf, J=J)
        cells, slot = trace_cells(x, sch)
        fp, mp = pair_sets(cells, slot, sch)
        pairs.append((fp, mp))
        if 'cells' in tr:   # surviving fraction of the original (seed, cell) pairs
            f0, m0 = pair_sets(tr['cells'], tr['slot'], sch)
            d.update(n_f0=len(f0), n_m0=len(m0), surv_f=survival(f0, fp), surv_m=survival(m0, mp) if m0 else np.nan)
        d['i'] = i
        rows.append(d)
    return pd.DataFrame(rows), pairs


def _release_task(args):
    gen, rid, attacks_list, test_schemes, key, pooled_cfg = args
    cfg = CFG
    g = cfg['generators'][gen]
    R = load_pickle(os.path.join(release_dir(cfg, gen), rid + '.pkl'))
    traces = R['traces']
    pool_real = real_traces(cfg, g['dataset'], 'test', limit=2000, seed=2)
    J = cfg['eval'].get('J', 1)
    T = Timer()
    done = 0
    for tname, sch in test_schemes:
        for attack in attacks_list:
            suffix = '' if tname is None else f'__T{tname}'
            out = os.path.join(cfg['out_dir'], 'eval', gen, f'{rid}__{attack}{suffix}.parquet')
            if os.path.exists(out):
                continue
            n_max = len(traces) if attack == 'none' else cfg['eval']['n_eval']
            df, pairs = eval_traces(traces, attack, sch, key, J, f'{gen}/{rid}', cfg, gen, rid, pool_real, n_max)
            if df is None:
                continue
            df['attack'] = attack
            # dataset-level pooled tests
            pj = os.path.join(cfg['out_dir'], 'eval', gen, 'pooled.jsonl')
            prf = PRF(key)
            recs = []
            for m in pooled_cfg['sizes']:
                G = min(pooled_cfg['groups'], len(pairs) // m)
                lps = [pooled(pairs[gi * m:(gi + 1) * m], sch, prf)['lp'] for gi in range(G)]
                if lps:
                    recs.append(dict(key=f'{rid}|{attack}|{tname}|m{m}', rid=rid, attack=attack, T=tname, m=m, lps=lps))
            if attack == 'none' and tname is None:   # dataset-level mixing with real traces
                rp = [pair_sets(*trace_cells(tr, sch), sch) for tr in pool_real[:1000]]
                rng = np.random.default_rng(seed_from(rid, 'mix'))
                for f in pooled_cfg['mix_fracs']:
                    for m in pooled_cfg['sizes']:
                        nr = int(round(m * f / 100)); nw = m - nr
                        lps = []
                        for gi in range(min(pooled_cfg['groups'], len(pairs) // max(nw, 1))):
                            grp = pairs[gi * nw:(gi + 1) * nw] + [rp[j] for j in rng.choice(len(rp), size=nr, replace=False)]
                            lps.append(pooled(grp, sch, prf)['lp'])
                        recs.append(dict(key=f'{rid}|mix{f}|{tname}|m{m}', rid=rid, attack=f'mix{f}', T=tname, m=m, lps=lps))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for r in recs:
                append_jsonl(pj, r)
            df.to_parquet(out + '.tmp', index=False)
            os.replace(out + '.tmp', out)
            done += 1
    return f'{gen}/{rid}: {done} files ({T():.0f}s)'


def _null_task(args):
    ds, res, sname, sch, key, attack = args
    cfg = CFG
    out = os.path.join(cfg['out_dir'], 'eval', 'real', f'{ds}_r{res}', f'{sname}__k{key}__{attack}.parquet')
    if os.path.exists(out):
        return None
    trs = real_traces(cfg, ds, 'test', limit=cfg['eval']['null_n'], seed=1, min_pts=2)
    pool_real = real_traces(cfg, ds, 'test', limit=2000, seed=2)
    df, _ = eval_traces(trs, attack, sch, key, cfg['eval'].get('J', 1), f'real_{ds}', cfg, f'real_{ds}', 'test', pool_real,
                        cfg['eval']['null_n'])
    if df is None:
        return f'null {ds} {attack}: cache missing'
    df['attack'] = attack
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    return f'null {ds} r{res} {sname} k{key} {attack}: {len(df)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--sets', default='main')
    ap.add_argument('--gens', default=None)
    ap.add_argument('--nulls', action='store_true')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    ev = cfg['eval']
    tasks = []
    gens = a.gens.split(',') if a.gens else list(cfg['generators'])
    for sn in a.sets.split(','):
        atk = ev['attack_sets'].get(sn, ev['attack_sets']['default'])
        for gen in gens:
            g = cfg['generators'][gen]
            for j in release_jobs(cfg, gen, [sn]):
                rid = f"{j['wm']}__k{j['key']}__s{j['seed']}" + (f"__{j['decode']}" if j['decode'] != 'uniform' else '')
                if not os.path.exists(os.path.join(release_dir(cfg, gen), rid + '.pkl')):
                    print('missing release', gen, rid); continue
                meth = cfg['wm'][j['wm']]['method']
                if meth == 'wtrace':
                    continue
                if meth == 'none':
                    ts = [(t, wm_scheme(cfg, t, g['res'])) for t in ev['null_schemes']]
                else:
                    ts = [(None, Scheme.from_dict(load_pickle_meta(cfg, gen, rid)))]
                tasks.append((gen, rid, atk, ts, j['key'], ev['pooled']))
    null_tasks = []
    if a.nulls:
        for ds in cfg['datasets']:
            for res in sorted({g['res'] for g in cfg['generators'].values() if g['dataset'] == ds}):
                # Stage 4: extra detector configurations (ablations / strong operating points) get their own real nulls;
                # map-matching nulls at every resolution (the real-trace mm cache holds points, so it is resolution-free)
                extra = ev.get('null_schemes_extra', {}).get(f'{ds}_r{res}', [])
                for sname in list(ev['null_schemes']) + [x for x in extra if x not in ev['null_schemes']]:
                    for key in ev.get('null_keys', [0, 1, 2]):
                        for attack in ev['attack_sets']['main']:
                            if attack == 'regen':
                                continue
                            null_tasks.append((ds, res, sname, wm_scheme(cfg, sname, res), key, attack))
    print(f'{len(tasks)} release tasks, {len(null_tasks)} null tasks', flush=True)
    with Pool(scaled(ev.get('workers', 8), 1), initializer=_init, initargs=(a.cfg,)) as pool:
        for msg in pool.imap_unordered(_release_task, tasks):
            print(msg, flush=True)
        for msg in pool.imap_unordered(_null_task, null_tasks):
            if msg:
                print(msg, flush=True)


_META = {}


def load_pickle_meta(cfg, gen, rid):
    k = (gen, rid)
    if k not in _META:
        import json
        mf = os.path.join(release_dir(cfg, gen), rid + '.meta.json')
        _META[k] = json.load(open(mf))['scheme'] if os.path.exists(mf) else \
            load_pickle(os.path.join(release_dir(cfg, gen), rid + '.pkl'))['meta']['scheme']
    return _META[k]


if __name__ == '__main__':
    main()
