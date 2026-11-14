"""Stage 7: post-hoc recolouring at several rates (strength-utility frontier of the post-hoc baseline).
Builds gen/<gen>/posthoc_rXX__k<key>__s0.pkl from the first n traces of the unwatermarked release none__k0__s0
(true key, GeoMark's coarse hashing, recolouring probability = rate). Usage: python src/rev1_posthoc_prep.py --cfg CFG"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, save_pickle, load_pickle, dump_json, seed_from, Timer
from geomark.prf import PRF
from geomark.posthoc import recolor_trace, true_key_scorer
from geomark.data import release_dir
from generate import wm_scheme, release_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--set', default='rev1_posthoc')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    rs = cfg['releases'][a.set]
    for gen in rs['gens']:
        g = cfg['generators'][gen]
        for j in release_jobs(cfg, gen, [a.set]):
            rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
            out = os.path.join(release_dir(cfg, gen), rid + '.pkl')
            src = os.path.join(release_dir(cfg, gen), f"none__k0__s{j['seed']}.pkl")
            if os.path.exists(out):
                continue
            T = Timer()
            S = load_pickle(src)['traces'][:j['n']]
            sch = wm_scheme(cfg, j['wm'], g['res'])
            rate = float(cfg['wm'][j['wm']]['rate'])
            scorer = true_key_scorer(sch.with_(method='kgw'), PRF(j['key']))
            rng = np.random.default_rng(seed_from(gen, rid))
            trs = []
            for tr in S:
                x = recolor_trace(tr, sch, scorer, 'green', rate, rng)
                trs.append(x if x is not None else dict(tr, frac_changed=0.0))
            meta = dict(gen=gen, wm=j['wm'], scheme=sch.to_dict(), key=j['key'], seed=j['seed'], n=len(trs), decode='uniform',
                        dataset=g['dataset'], res=g['res'], sec=T(), source=os.path.basename(src), rate=rate)
            save_pickle(dict(meta=meta, traces=trs), out)
            dump_json(meta, out[:-4] + '.meta.json')
            print(f'{gen}/{rid}: {len(trs)} traces, rate {rate}, changed {np.mean([t["frac_changed"] for t in trs]):.3f} ({T():.0f}s)', flush=True)


if __name__ == '__main__':
    main()
