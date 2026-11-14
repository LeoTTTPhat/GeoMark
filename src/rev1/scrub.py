"""Stage 7: the launderer IS a stealer (reviewers: a licensee holding one release of 10^4 traces can run the stealing
attack on its own release and then scrub it; scrub + map-matching and the utility cost of scrubbing were not measured).

Attacker A holds its own release R_A = the first 10,000 traces of the stealing corpus (GeoMark-KGW k = 1, and k = 2),
estimates green lists from R_A against 80,000 real training traces (as run_steal.py), and scrubs R_A itself towards
estimated red cells (rates 0.5 and 1.0; random recolouring at rate 1.0 as a key-agnostic reference). Then:
  * pooled detection (true key) of 100- and 1,000-trace pools of the scrubbed release, and after map-matching (mm and
    noise30+mm; first 1,000 scrubbed traces, same matcher settings as the main study);
  * utility of the scrubbed release (corrected metrics of src/rev1/utility.py, first 5,000 traces) next to R_A itself.
Outputs: results/rev1/scrub.json, scrubbed/matched traces cached in results/rev1/scrub/. No model is loaded.
Usage: python src/rev1/scrub.py"""
import os, sys, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump, save_pickle, load_pickle
from rev1.utility import fixed_metrics
from geomark.common import seed_from, Timer, scaled
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, detect_trace, pooled
from geomark.posthoc import recolor_trace
from geomark.utility import utility
from geomark.data import real_traces, load_ds
from geomark.hexgrid import tokenize
from geomark import attacks
from geomark.mapmatch import match_trace
from run_steal import Estimate
from run_utility import real_cells
from generate import wm_scheme

D = os.path.join(OUT, 'scrub')
_MC = None


def _init(mc):
    global _MC
    _MC = mc


def _mm(args):
    tr, pre, seed = args
    x = attacks.apply(tr, pre, np.random.default_rng(seed)) if pre else tr
    try:
        r = match_trace(x, _MC)
        return r['mm'], r['frac_matched']
    except Exception:
        return None, 0.0


def det(trs, sch, prf):
    ok = [t is not None and len(t['lat']) > 0 for t in trs]          # failed matches count as zero evidence (as in evaluate.py)
    lps = np.array([detect_trace(t, sch, prf)['lp'] if o else 0.0 for t, o in zip(trs, ok)])
    pl = [pair_sets(*trace_cells(t, sch), sch) if o else ([], []) for t, o in zip(trs, ok)]
    out = dict(n=len(trs), tpr_m3=float((lps <= -3).mean()))
    for m in (100, 1000):
        ps = [pooled(pl[i:i + m], sch, prf)['lp'] for i in range(0, len(pl) - m + 1, m)]
        out[f'm{m}'] = dict(lps=[round(float(x), 3) for x in ps], median=float(np.median(ps)) if ps else None,
                            frac_le_m3=float(np.mean(np.array(ps) <= -3)) if ps else None, n_pools=len(ps))
    return out


def main():
    cfg = cfg_rev1()
    os.makedirs(D, exist_ok=True)
    fn = os.path.join(OUT, 'scrub.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    sc = cfg['steal']
    gen = sc['gen']
    g = cfg['generators'][gen]
    ds, res = g['dataset'], g['res']
    T = Timer()
    real_train = real_traces(cfg, ds, 'train', limit=sc['n_prior'], seed=7)
    real_u = real_cells(cfg, ds, res, 'test', 5000, 0)
    Dd = load_ds(cfg, ds)
    train_set = {tuple(int(x) for x in Dd['tok'][res][i]) for i, t in enumerate(Dd['traces']) if t['split'] == 'train'}
    mc = cfg['mapmatch']
    pool = Pool(scaled(mc.get('workers', 16), 1), initializer=_init, initargs=(mc,))
    for name, wm, rid in (('k1', 'gm_kgw', 'gm_kgw__k0__s10'), ('k2', 'gm_kgw_k2', 'gm_kgw_k2__k0__s10')):
        sch = wm_scheme(cfg, wm, res)
        prf = PRF(0)
        RA = load_release(cfg, gen, rid)['traces'][:10000]
        est = None
        variants = [('clean', None, 0.0), ('scrub_r0.5', 'red', 0.5), ('scrub_r1.0', 'red', 1.0), ('random_r1.0', 'random', 1.0)]
        for vname, target, rate in variants:
            k = f'{name}|N10000|{vname}'
            if k in out and 'mm' in out[k]:
                continue
            cf = os.path.join(D, f'{name}_{vname}.pkl')
            if os.path.exists(cf):
                trs = load_pickle(cf)
            else:
                if target is None:
                    trs = RA[:5000]
                else:
                    rng = np.random.default_rng(seed_from('rev1-scrub', name, vname))
                    if target == 'red' and est is None:
                        prior_pairs = [pair_sets(*trace_cells(t, sch), sch) for t in real_train]
                        est = Estimate(sch, [pair_sets(t['cells'], t['slot'], sch) for t in RA], prior_pairs)
                        out[f'{name}|N10000|accuracy'] = est.accuracy(prf)
                    scorer = est.scorer() if target == 'red' else (lambda s, c: rng.random(len(c)))
                    trs = []
                    for t in RA[:5000]:
                        x = recolor_trace(t, sch, scorer, 'red' if target == 'red' else 'green', rate, rng)
                        trs.append(x if x is not None else dict(t, frac_changed=0.0))
                save_pickle(trs, cf)
            rec = dict(frac_changed=float(np.mean([t.get('frac_changed', 0.0) for t in trs])))
            rec['none'] = det(trs[:2000], sch, prf)
            cells = [tokenize(t['lat'], t['lon'], res) for t in trs[:5000]]
            u = utility(cells, real_u, ds, res, train_set)
            u.update(fixed_metrics(cells, real_u, ds, None, None))
            rec['utility'] = u
            for pre in ('mm', 'noise30+mm'):
                mf = os.path.join(D, f'{name}_{vname}__{pre}.pkl')
                if os.path.exists(mf):
                    mm = load_pickle(mf)
                else:
                    p0 = pre[:-len('+mm')] if pre != 'mm' else ''
                    res_ = pool.map(_mm, [(t, p0, seed_from('rev1-scrub-mm', name, vname, pre, i)) for i, t in enumerate(trs[:1000])], chunksize=4)
                    mm = [r[0] for r in res_]
                    save_pickle(mm, mf)
                rec[pre] = det(mm, sch, prf)
            out[k] = rec
            print(f"{k}: changed {rec['frac_changed']:.3f}; pooled m100 median none {rec['none']['m100']['median']:.1f} "
                  f"(det {rec['none']['m100']['frac_le_m3']:.2f}), mm {rec['mm']['m100']['median']:.1f} "
                  f"(det {rec['mm']['m100']['frac_le_m3']:.2f}); bigram JSD {u['jsd_bigram']:.3f} ({T():.0f}s)", flush=True)
            jdump(out, 'scrub.json')
    pool.close()


if __name__ == '__main__':
    main()
