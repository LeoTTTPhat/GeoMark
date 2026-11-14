"""E4 map-matching cache: runs HMM map-matching (optionally after a pre-attack such as noise30) on the first
`mapmatch.n` traces of selected releases and on real test traces (attack nulls). Output (chunked, resumable):
  <out_dir>/attacked/<gen>/<rid>__<prefix>.pkl  and  ...__<prefix minus 'mm' + 'mmrs'>.pkl
  <out_dir>/attacked/real_<ds>/test__<prefix>.pkl
Usage: python src/mapmatch_run.py --cfg CFG [--sets main,llm] [--real]"""
import os, sys, argparse, glob
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, scaled, seed_from, save_pickle, load_pickle, Timer, append_jsonl
from geomark import attacks
from geomark.mapmatch import match_trace
from geomark.data import release_dir, real_traces
from generate import release_jobs

_MC = None


def _init(mc):
    global _MC
    _MC = mc


def _work(args):
    tr, pre, seed = args
    rng = np.random.default_rng(seed)
    x = attacks.apply(tr, pre, rng) if pre else tr
    try:
        r = match_trace(x, _MC)
    except Exception as e:  # never kill the pool on one trace
        return None, None, dict(err=str(e)[:200])
    return r['mm'], r['mmrs'], dict(frac=r['frac_matched'], breaks=r['n_breaks'])


def rs_name(prefix):
    return prefix[:-2] + 'mmrs'


def run_one(traces, out_mm, tag, prefix, mc, pool):
    out_rs = out_mm.replace(f'__{prefix}.pkl', f'__{rs_name(prefix)}.pkl')
    if os.path.exists(out_mm) and os.path.exists(out_rs):
        return
    pre = prefix[:-len('+mm')] if prefix != 'mm' else ''
    chunk = mc.get('chunk', 250)
    T = Timer()
    mm, rs, info, parts = [], [], [], []
    for a in range(0, len(traces), chunk):
        pf = out_mm + f'.part{a // chunk}'
        if os.path.exists(pf):
            m_, r_, i_ = load_pickle(pf)
        else:
            res = pool.map(_work, [(tr, pre, seed_from(tag, prefix, a + i)) for i, tr in enumerate(traces[a:a + chunk])], chunksize=4)
            m_, r_, i_ = [x[0] for x in res], [x[1] for x in res], [x[2] for x in res]
            save_pickle((m_, r_, i_), pf)
        parts.append(pf); mm += m_; rs += r_; info += i_
    fr = [i.get('frac', 0) for i in info]
    meta = dict(prefix=prefix, n=len(traces), mean_frac_matched=float(np.mean(fr)), full_match=float(np.mean([f >= 0.999 for f in fr])),
                errors=sum('err' in i for i in info), sec=T())
    save_pickle(dict(meta=meta, traces=mm, info=info), out_mm)
    save_pickle(dict(meta=dict(meta, prefix=rs_name(prefix)), traces=rs, info=info), out_rs)
    for pf in parts:
        os.remove(pf)
    print(f'  {tag} {prefix}: {meta}', flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--sets', default=None)
    ap.add_argument('--real', action='store_true', help='also match real test traces (attack nulls)')
    ap.add_argument('--gens', default=None, help='restrict to these generators (comma-separated; Stage 4)')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    mc = cfg['mapmatch']
    sets = a.sets.split(',') if a.sets else mc['sets']
    n = mc['n']
    with Pool(scaled(mc.get('workers', 16), 1), initializer=_init, initargs=(mc,)) as pool:
        if a.real:
            for ds in cfg['datasets']:
                if ds not in mc.get('datasets', cfg['datasets']):
                    continue
                trs = real_traces(cfg, ds, 'test', limit=cfg['eval']['null_n'], seed=1, min_pts=2)
                for prefix in mc['prefixes']:
                    out = os.path.join(cfg['out_dir'], 'attacked', f'real_{ds}', f'test__{prefix}.pkl')
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    m = run_one(trs, out, f'real_{ds}', prefix, mc, pool)
                    if m:
                        append_jsonl(os.path.join(cfg['out_dir'], 'attacked', 'mm_log.jsonl'), dict(tag=f'real_{ds}', **m))
        for gen, g in cfg['generators'].items():
            if g['dataset'] not in mc.get('datasets', cfg['datasets']):
                continue
            if a.gens and gen not in a.gens.split(','):
                continue
            for j in release_jobs(cfg, gen, sets):
                if j['seed'] not in mc.get('seeds', [j['seed']]):
                    continue
                rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
                f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
                if not os.path.exists(f):
                    continue
                R = load_pickle(f)
                trs = R['traces'][:n]
                for prefix in mc['prefixes']:
                    out = os.path.join(cfg['out_dir'], 'attacked', gen, f'{rid}__{prefix}.pkl')
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    m = run_one(trs, out, f'{gen}/{rid}', prefix, mc, pool)
                    if m:
                        append_jsonl(os.path.join(cfg['out_dir'], 'attacked', 'mm_log.jsonl'), dict(tag=f'{gen}/{rid}', **m))


if __name__ == '__main__':
    main()
