"""Stage 10 (round-3 audit): small numbers quoted in the text that were not stored before: distinct scored fine pairs per
generated trip and per 100-trace pool (no attack) for GeoMark-KGW and the step-neutral default (Porto / small GPT, seed-0
releases of keys 0-2; all disjoint 100-trace pools of the first 2,000 traces).
Output: results/rev3/numbers3.json. Model-free. Usage: python src/rev3/numbers3.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r3common import cfg_rev3, load_release, dump_json, OUT3
from geomark.detect import pair_sets
from generate import wm_scheme


def main():
    cfg = cfg_rev3()
    out = {}
    for wm in ('gm_kgw', 'gm_kgw_sn'):
        sch = wm_scheme(cfg, wm, 9)
        per_trip, per_pool = [], []
        for key in (0, 1, 2):
            R = load_release(cfg, 'gpt_porto_r9_s0', f'{wm}__k{key}__s0')
            pl = [pair_sets(t['cells'], t['slot'], sch) for t in R['traces'][:2000]]
            per_trip += [len(f) for f, _ in pl]
            for i in range(0, 2000, 100):
                F = set()
                for f, _ in pl[i:i + 100]:
                    F.update(f)
                per_pool.append(len(F))
        out[wm] = dict(median_pairs_per_trip=float(np.median(per_trip)), mean_pairs_per_trip=float(np.mean(per_trip)),
                       median_pairs_per_100pool=float(np.median(per_pool)), n_trips=len(per_trip), n_pools=len(per_pool))
        print(wm, out[wm], flush=True)
    dump_json(out, os.path.join(OUT3, 'numbers3.json'))


if __name__ == '__main__':
    main()
