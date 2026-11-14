"""Stage 4: bin-shift search radius J vs re-timestamping. The default detector searches +-J=1 time bins (Bonferroni
factor 3), which cannot recover a +-60 min shift of 30-min bins (2 bins). For seed-0 releases of the given generators /
watermarks (keys 0..2) and attacks none / retime15 / retime60, compute lp_shift for J in cfg.eval.shift.J on the first
n traces, plus real-trace nulls (same attacks, keys 0..2) for the empirical FPR at each J.
Output: <out_dir>/eval_shift/<gen>.jsonl and <out_dir>/eval_shift/real_<ds>_r<res>.jsonl (one record per release x
attack, per-trace lp lists; resumable). Usage: python src/run_shift.py --cfg CFG --gens g1,g2 --wms gm_kgw,gm_gumbel"""
import os, sys, argparse
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, load_pickle, append_jsonl, done_keys, seed_from, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark.detect import trace_cells, detect_cells
from geomark import attacks
from geomark.data import release_dir, real_traces
from generate import wm_scheme

CFG = None


def _init(p):
    global CFG
    CFG = load_cfg(p)


def _task(args):
    tag, traces, sch_d, key, attack, Js = args
    sch = Scheme.from_dict(sch_d)
    prf = PRF(key)
    out = {f'J{J}': [] for J in Js}
    out['J0'] = []
    for i, tr in enumerate(traces):
        x = attacks.apply(tr, attack, np.random.default_rng(seed_from(tag, attack, i)))
        cells, slot = trace_cells(x, sch)
        for J in Js:
            d = detect_cells(cells, slot, sch, prf, J=J)
            out[f'J{J}'].append(round(float(d['lp_shift']), 4))
            if J == Js[0]:
                out['J0'].append(round(float(d['lp']), 4))
    return tag, key, attack, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--gens', required=True)
    ap.add_argument('--wms', default='gm_kgw,gm_gumbel')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    sc = cfg['eval']['shift']
    Js, n = sc['J'], sc['n']
    od = os.path.join(cfg['out_dir'], 'eval_shift')
    os.makedirs(od, exist_ok=True)
    tasks = []
    dsr_done = set()
    for gen in a.gens.split(','):
        g = cfg['generators'][gen]
        out = os.path.join(od, f'{gen}.jsonl')
        done = done_keys(out)
        for wm in a.wms.split(','):
            for key in (0, 1, 2):
                rid = f'{wm}__k{key}__s0'
                f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
                if not os.path.exists(f):
                    print('missing', gen, rid, flush=True); continue
                R = load_pickle(f)
                trs = R['traces'][:n]
                for atk in sc['attacks']:
                    if f'{rid}|{atk}' in done:
                        continue
                    tasks.append((out, f'{gen}/{rid}', trs, R['meta']['scheme'], key, atk))
            # real nulls for this dataset/resolution/scheme
            dsr = f"{g['dataset']}_r{g['res']}"
            if (dsr, wm) in dsr_done:
                continue
            dsr_done.add((dsr, wm))
            nout = os.path.join(od, f'real_{dsr}.jsonl')
            ndone = done_keys(nout)
            real = real_traces(cfg, g['dataset'], 'test', limit=cfg['eval']['null_n'], seed=1, min_pts=2)
            sch = wm_scheme(cfg, wm, g['res']).to_dict()
            for key in (0, 1, 2):
                for atk in sc['attacks']:
                    if f'{wm}__k{key}|{atk}' not in ndone:
                        tasks.append((nout, f'real_{g["dataset"]}/{wm}__k{key}', real, sch, key, atk))
    print(f'{len(tasks)} shift tasks', flush=True)
    with Pool(scaled(cfg['eval'].get('workers', 8), 1), initializer=_init, initargs=(a.cfg,)) as pool:
        res = pool.imap_unordered(_task, [(t[1], t[2], t[3], t[4], t[5], Js) for t in tasks])
        outs = {t[1] + '|' + t[5]: t[0] for t in tasks}
        for tag, key, atk, o in res:
            rid = tag.split('/')[1]
            rec = dict(key=f'{rid}|{atk}', tag=tag, wm_key=key, attack=atk, n=len(o['J0']), **{k: v for k, v in o.items()})
            append_jsonl(outs[tag + '|' + atk], rec)
            print(f'shift {tag} {atk}: ' + ', '.join(f"{k} tpr@1e-3={np.mean(np.array(v) <= -3):.3f}" for k, v in o.items()), flush=True)


if __name__ == '__main__':
    main()
