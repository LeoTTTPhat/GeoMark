"""Stage 4b (2026-10-17): key variability of detection power and of the key-CONDITIONAL false-positive rate.

Why: the main detection cells have 3 keys x 3 sampling seeds; for Gumbel-H the key effect dominates (Porto small GPT
TPR@1e-3 0.077 / 0.20 / 0.21 for keys 0 / 1 / 2, near-identical across seeds), so a t-interval over the 9 (key, seed)
releases treats 3 independent units as 9 (pseudo-replication). Lemma 1 (exactness) holds over a RANDOM key; a single
owner holds ONE key, so the FPR conditional on the key also matters (it only equals the nominal level on average).

(a) per-key TPR: for each generator in cfg[keyvar][tpr_sets], every release of that set (200 traces per key, keys 100..)
    is detected with its own key under attacks cfg[keyvar][tpr_attacks] (non-map-matching attacks; same attack RNG tags
    as evaluate.py). Output: <out_dir>/keyvar/tpr.jsonl (one record per gen x release x attack).
(b) key-conditional null: the SAME real test traces as the eval nulls (eval.null_n, seed 1) are scored under
    cfg[keyvar][null_n_keys] fresh keys (null_key0 ..) with each null scheme, attack none, per trace (exact p-values, as
    evaluate.py) and pooled over disjoint groups of m traces. Output: <out_dir>/keyvar/null.jsonl (one record per
    cell x scheme x key).
Both parts are resumable (records keyed; finished keys are skipped). Hashing only (no model). Reads MEMGUARD_SCALE via
scaled() for the worker count. Usage: python src/run_keyvar.py --cfg CFG [--part tpr,null]"""
import os, sys, argparse
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, scaled, load_pickle, append_jsonl, done_keys, Timer
from geomark.scheme import Scheme
from geomark.prf import PRF
from geomark.detect import pooled
from geomark.data import release_dir, real_traces
from generate import release_jobs, wm_scheme
import evaluate as EV

CFG = None
_REAL = {}


def _init(cfg_path):
    global CFG
    CFG = load_cfg(cfg_path)
    EV.CFG = CFG


def _real(ds, kind):
    if (ds, kind) not in _REAL:
        if kind == 'null':
            _REAL[(ds, kind)] = real_traces(CFG, ds, 'test', limit=CFG['eval']['null_n'], seed=1, min_pts=2)   # == eval nulls
        else:
            _REAL[(ds, kind)] = real_traces(CFG, ds, 'test', limit=2000, seed=2)                               # == eval pool
    return _REAL[(ds, kind)]


def _counts(lp):
    lp = np.asarray(lp, dtype=float)
    return {f'k_le_m{t}': int((lp <= -t).sum()) for t in (2, 3, 4, 5)}


def _tpr_task(args):
    gen, rid, key, attacks_list = args
    cfg = CFG
    g = cfg['generators'][gen]
    R = load_pickle(os.path.join(release_dir(cfg, gen), rid + '.pkl'))
    sch = Scheme.from_dict(EV.load_pickle_meta(cfg, gen, rid))
    pool_real = _real(g['dataset'], 'pool')
    J = cfg['eval'].get('J', 1)
    m = cfg['keyvar'].get('pooled_m', 100)
    recs = []
    for attack in attacks_list:
        df, pairs = EV.eval_traces(R['traces'], attack, sch, key, J, f'{gen}/{rid}', cfg, gen, rid, pool_real, len(R['traces']))
        if df is None:
            continue
        lp = df['lp'].values
        sel = (df['n_f0'] >= 20).values if 'n_f0' in df else np.zeros(len(df), bool)
        prf = PRF(key)
        plp = [pooled(pairs[i * m:(i + 1) * m], sch, prf)['lp'] for i in range(len(pairs) // m)]
        recs.append(dict(key=f'{gen}|{rid}|{attack}', gen=gen, rid=rid, wm=rid.split('__k')[0], wm_key=key, attack=attack,
                         n=int(len(df)), tpr_1e3=float((lp <= -3).mean()), tpr_1e2=float((lp <= -2).mean()),
                         tpr_1e5=float((lp <= -5).mean()), median_lp=float(np.median(lp)), median_n_f=float(df['n_f'].median()),
                         n_ge20=int(sel.sum()), tpr_1e3_ge20pairs=float((lp[sel] <= -3).mean()) if sel.any() else None,
                         pooled_m=m, pooled_lps=[round(float(x), 3) for x in plp]))
    return recs


def _null_task(args):
    cell, sname, keys = args
    cfg = CFG
    ds, res = cell.split('_r')[0], int(cell.split('_r')[1])
    sch = wm_scheme(cfg, sname, res)
    trs = _real(ds, 'null')
    pool_real = _real(ds, 'pool')
    J = cfg['eval'].get('J', 1)
    m = cfg['keyvar'].get('pooled_m', 100)
    recs = []
    for key in keys:
        df, pairs = EV.eval_traces(trs, 'none', sch, key, J, f'real_{ds}', cfg, f'real_{ds}', 'test', pool_real, len(trs))
        lp = df['lp'].values
        prf = PRF(key)
        plp = np.array([pooled(pairs[i * m:(i + 1) * m], sch, prf)['lp'] for i in range(len(pairs) // m)])
        recs.append(dict(key=f'{cell}|{sname}|k{key}', cell=cell, scheme=sname, null_key=key, n=int(len(lp)),
                         median_lp=float(np.median(lp)), min_lp=float(lp.min()), **_counts(lp),
                         fine_only=_counts(df['lp_f'].values), n_pooled=int(len(plp)), pooled_m=m,
                         pooled={f'k_le_m{t}': int((plp <= -t).sum()) for t in (2, 3)},
                         pooled_min_lp=float(plp.min()) if len(plp) else None))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--part', default='tpr,null')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    kc = cfg['keyvar']
    od = os.path.join(cfg['out_dir'], 'keyvar')
    os.makedirs(od, exist_ok=True)
    parts = a.part.split(',')
    T = Timer()
    with Pool(scaled(kc.get('workers', 8), 1), initializer=_init, initargs=(a.cfg,)) as pool:
        if 'tpr' in parts:
            out = os.path.join(od, 'tpr.jsonl')
            done = done_keys(out)
            tasks = []
            for gen, sn in kc['tpr_sets'].items():
                for j in release_jobs(cfg, gen, [sn]):
                    rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
                    if not os.path.exists(os.path.join(release_dir(cfg, gen), rid + '.pkl')):
                        print('missing release', gen, rid, flush=True); continue
                    todo = [x for x in kc['tpr_attacks'] if f'{gen}|{rid}|{x}' not in done]
                    if todo:
                        tasks.append((gen, rid, j['key'], todo))
            print(f'keyvar tpr: {len(tasks)} release tasks', flush=True)
            for recs in pool.imap_unordered(_tpr_task, tasks):
                for r in recs:
                    append_jsonl(out, r)
                if recs:
                    print(f"tpr {recs[0]['gen']}/{recs[0]['rid']}: " +
                          ' '.join(f"{r['attack']} {r['tpr_1e3']:.3f}" for r in recs) + f' ({T():.0f}s)', flush=True)
        if 'null' in parts:
            out = os.path.join(od, 'null.jsonl')
            done = done_keys(out)
            keys = list(range(kc['null_key0'], kc['null_key0'] + kc['null_n_keys']))
            tasks = []
            for cell in kc['null_cells']:
                for sname in kc['null_schemes']:
                    todo = [k for k in keys if f'{cell}|{sname}|k{k}' not in done]
                    for b in range(0, len(todo), 10):          # 10 keys per task (checkpoint granularity)
                        tasks.append((cell, sname, todo[b:b + 10]))
            print(f'keyvar null: {len(tasks)} tasks', flush=True)
            for recs in pool.imap_unordered(_null_task, tasks):
                for r in recs:
                    append_jsonl(out, r)
                r = recs[-1]
                print(f"null {r['cell']} {r['scheme']} keys ..{r['null_key']}: FPR@1e-3 {r['k_le_m3'] / r['n']:.4f} "
                      f"@1e-2 {r['k_le_m2'] / r['n']:.4f} ({T():.0f}s)", flush=True)


if __name__ == '__main__':
    main()
