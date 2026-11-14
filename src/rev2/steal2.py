"""Stage 8: stealing, forging, scrubbing and SELECTION for every seeding configuration (reviewers: the seed-space argument
for coarse context was asserted, not measured; k = 0 and exact+bin were never attacked; the EOS-/step-neutral default was
absent from the security experiments; breaking attacks were reported only for 100-trace pools; a selection attack was
missing).

Corpora (101,000 traces each, key 0, sampling seed 10, Porto small GPT): k1 = GeoMark-KGW (k=1), k2 (k=2), k0 (bin only),
exb = exact previous cell + bin (single resolution), eosn = EOS-neutral GeoMark-KGW, sn = step-neutral GeoMark-KGW.
Stealer = run_steal.Estimate (per-seed smoothed frequency ratio against 80,000 real training traces).
  (1) Per-pair learnability: for N = 10^3, 10^4, 10^5 observed traces, green precision / recall / mass-weighted accuracy of
      the estimate, and the MODEL prediction for the same pairs (exact binomial error of the threshold test given the
      number of visits of the seed, the prior frequency p(c|s) and the four KGW-2R classes of (fine, medium) coins;
      see stealbound2.py), so that the model is validated against the implemented stealer.
  (2) Recoloring forger: 1,000 real test traces recolored towards estimated green cells (N = 10^4, 10^5), tested with the
      true key: per-trace forgery rate and share of 100-trace pools at p <= 1e-3.
  (3) Launderer = stealer on its own release R_A (first 10^4 traces, N = 10^4): recolor all of R_A towards estimated
      red cells (rate 1); pooled detection at 100 / 1,000 / 10,000 traces, after map-matching (first 1,000 scrubbed
      traces), and utility.
  (4) Selection attack: keep the 50 / 25 / 10% of R_A with the lowest estimated green fraction (no trace is modified);
      pooled detection of 100-trace pools and of the whole kept subset.
Output: results/rev2/steal2.json; scrubbed / matched traces cached in results/rev2/steal2/. No model is loaded.
Usage: python src/rev2/steal2.py"""
import os, sys, math, json
from collections import Counter
from multiprocessing import Pool
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, OUT2, save_pickle, load_pickle
from geomark.common import seed_from, Timer, scaled
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, detect_trace, pooled
from geomark.posthoc import recolor_trace
from geomark.utility import utility
from geomark.data import real_traces, load_ds
from geomark.hexgrid import tokenize, parent
from geomark import attacks
from geomark.mapmatch import match_trace
from run_steal import Estimate
from run_utility import real_cells
from rev1.utility import fixed_metrics
from generate import wm_scheme

D = os.path.join(OUT2, 'steal2')
CORPORA = [('k1', 'gm_kgw'), ('k2', 'gm_kgw_k2'), ('k0', 'a_k0'), ('exb', 'exact_kgw_bin'), ('eosn', 'gm_kgw_eosn'), ('sn', 'gm_kgw_sn')]
_MC = None


def _init(mc):
    global _MC
    _MC = mc


def _mm(args):
    tr, seed = args
    try:
        r = match_trace(tr, _MC)
        return r['mm']
    except Exception:
        return None


def det(trs, sch, prf, sizes=(100, 1000, 10000), n100=2000):
    ok = [t is not None and len(t['lat']) > 0 for t in trs]
    pl = [pair_sets(*trace_cells(t, sch), sch) if o else ([], []) for t, o in zip(trs, ok)]
    lps = np.array([pooled([p], sch, prf)['lp'] for p in pl])
    out = dict(n=len(trs), tpr_m3=float((lps <= -3).mean()))
    for m in sizes:
        src = pl[:n100] if m == 100 else pl
        ps = [pooled(src[i:i + m], sch, prf)['lp'] for i in range(0, len(src) - m + 1, m)]
        out[f'm{m}'] = dict(n_pools=len(ps), median_mlp=float(-np.median(ps)) if ps else None,
                            frac_le_m3=float(np.mean(np.array(ps) <= -3)) if ps else None, lps=[round(float(x), 2) for x in ps])
    return out


def model_error(n, p, rate_factor):
    """Exact binomial error of the threshold test 'observed frequency of c after s > p' (vectorised).
    n visits of s; the observation 'c follows' has mean mu = p * rate_factor; truth green iff rate_factor > 1."""
    mu = np.clip(p * rate_factor, 1e-12, 1 - 1e-12)
    thr = np.floor(n * p)                                     # estimate green iff count > n p
    green = rate_factor > 1
    p_le = stats.binom.cdf(thr, n, mu)                        # P[count <= n p]
    return np.where(green, p_le, 1 - p_le)


def learnability(est, sch, prf, prior_counts, prior_tot, delta, dmid, gamma):
    """Measured vs model-predicted accuracy of the estimate over the observed fine pairs (mass-weighted)."""
    keys = list(est.w)
    s_, c_ = zip(*keys)
    true_f = prf.pairs('fine', s_, c_) < gamma
    if sch.uses_mid:
        mids = [parent(c, sch.mid_res) for c in c_]
        true_m = prf.pairs('mid', s_, mids) < gamma
    else:
        true_m = np.zeros(len(keys), bool)
    estg = np.array([est.green_fine(s, c) for s, c in keys])
    w = np.array([est.w[k] for k in keys], float)
    nvis = np.array([est.wm_tot[s] for s in s_], float)
    pp = np.array([(prior_counts.get(k, 0) + 0.5) / (prior_tot.get(k[0], 0) + 0.5) for k in keys])
    Zf = 1 + (math.exp(delta) - 1) * gamma
    Zm = 1 + (math.exp(dmid) - 1) * gamma if sch.uses_mid else 1.0
    fac = np.exp(delta * true_f + (dmid if sch.uses_mid else 0) * true_m) / (Zf * Zm)
    err_model = model_error(nvis, pp, fac)
    ok = ~np.isnan(estg)
    acc_meas = float((((estg == 1) == true_f) * w)[ok].sum() / w[ok].sum())
    acc_model = float(((1 - err_model) * w)[ok].sum() / w[ok].sum())
    learn_meas = float((w * (((estg == 1) == true_f) & ok)).sum() / w.sum())
    return dict(n_pairs=len(keys), acc_measured=acc_meas, acc_model=acc_model,
                frac_mass_err_le_0p1_model=float((w * (err_model <= 0.1)).sum() / w.sum()),
                precision=float(true_f[estg == 1].mean()) if (estg == 1).any() else None,
                recall=float((estg[true_f] == 1).mean()) if true_f.any() else None, base_rate=float(true_f.mean()),
                mass_correct=learn_meas)


def main():
    cfg = cfg_rev2()
    os.makedirs(D, exist_ok=True)
    out = jload2('steal2.json')
    sc = cfg['steal']
    gen = sc['gen']
    ds, res = 'porto', 9
    T = Timer()
    real_train = real_traces(cfg, ds, 'train', limit=sc['n_prior'], seed=7)
    real_test = real_traces(cfg, ds, 'test', limit=1000, seed=8)
    real_u = real_cells(cfg, ds, res, 'test', 5000, 0)
    Dd = load_ds(cfg, ds)
    train_set = {tuple(int(x) for x in Dd['tok'][res][i]) for i, t in enumerate(Dd['traces']) if t['split'] == 'train'}
    mc = cfg['mapmatch']
    pool = Pool(scaled(mc.get('workers', 16), 1), initializer=_init, initargs=(mc,))
    for name, wm in CORPORA:
        R = load_release(cfg, gen, f'{wm}__k0__s10')
        if R is None:
            print('missing corpus', name, flush=True); continue
        sch = wm_scheme(cfg, wm, res)
        prf = PRF(0)
        trs = R['traces']
        prior_pairs = None
        wm_pairs = None
        rec = out.get(name, {})
        need = [k for k in ('learn', 'spoof', 'scrub', 'select') if k not in rec]
        if not need:
            continue
        prior_pairs = [pair_sets(*trace_cells(t, sch), sch) for t in real_train]
        pc, pt = Counter(), Counter()
        for f, _ in prior_pairs:
            for s, c in f:
                pc[(s, c)] += 1; pt[s] += 1
        wm_pairs = [pair_sets(t['cells'], t['slot'], sch) for t in trs[:100000]]
        ests = {}
        if 'learn' not in rec:
            rec['learn'] = {}
            for N in (1000, 10000, 100000):
                ests[N] = Estimate(sch, wm_pairs[:N], prior_pairs)
                rec['learn'][str(N)] = learnability(ests[N], sch, prf, pc, pt, sch.delta, sch.delta_mid, sch.gamma)
                L = rec['learn'][str(N)]
                print(f"{name} N={N}: acc meas {L['acc_measured']:.3f} model {L['acc_model']:.3f} prec {L['precision']:.3f} "
                      f"({T():.0f}s)", flush=True)
            out[name] = rec; jdump2(out, 'steal2.json')
        if 'spoof' not in rec:
            rec['spoof'] = {}
            for N in (10000, 100000):
                est = ests.get(N) or Estimate(sch, wm_pairs[:N], prior_pairs)
                rng = np.random.default_rng(seed_from('steal2-spoof', name, N))
                spo = [recolor_trace(t, sch, est.scorer(), 'green', 1.0, rng) for t in real_test]
                spo = [t for t in spo if t]
                rec['spoof'][str(N)] = det(spo, sch, prf, sizes=(100,), n100=len(spo))
            rec['spoof']['real_unmodified'] = det(real_test, sch, prf, sizes=(100,), n100=len(real_test))
            print(f"{name} spoof: {[(k, v['m100']['frac_le_m3']) for k, v in rec['spoof'].items()]} ({T():.0f}s)", flush=True)
            out[name] = rec; jdump2(out, 'steal2.json')
        RA = trs[:10000]
        est = ests.get(10000) or Estimate(sch, wm_pairs[:10000], prior_pairs)
        if 'scrub' not in rec:
            rec['scrub'] = {}
            cf = os.path.join(D, f'{name}_scrub_r1.0.pkl')
            if os.path.exists(cf):
                scr = load_pickle(cf)
            else:
                rng = np.random.default_rng(seed_from('steal2-scrub', name))
                scr = []
                for t in RA:
                    x = recolor_trace(t, sch, est.scorer(), 'red', 1.0, rng)
                    scr.append(x if x is not None else dict(t, frac_changed=0.0))
                save_pickle(scr, cf)
            for vname, X in (('clean', RA), ('scrub_r1.0', scr)):
                r = dict(frac_changed=float(np.mean([t.get('frac_changed', 0.0) for t in X])) if vname != 'clean' else 0.0)
                r['none'] = det(X, sch, prf)
                cells = [tokenize(t['lat'], t['lon'], res) for t in X[:5000]]
                u = utility(cells, real_u, ds, res, train_set)
                u.update(fixed_metrics(cells, real_u, ds, None, None))
                r['utility'] = {k: v for k, v in u.items() if isinstance(v, (int, float))}
                mf = os.path.join(D, f'{name}_{vname}__mm.pkl')
                if os.path.exists(mf):
                    mm = load_pickle(mf)
                else:
                    mm = pool.map(_mm, [(t, i) for i, t in enumerate(X[:1000])], chunksize=4)
                    save_pickle(mm, mf)
                r['mm'] = det(mm, sch, prf, sizes=(100, 1000), n100=1000)
                rec['scrub'][vname] = r
                print(f"{name} {vname}: m100/m1000/m10000 {r['none']['m100']['median_mlp']}/{r['none']['m1000']['median_mlp']}/"
                      f"{r['none']['m10000']['median_mlp']}; mm m100 {r['mm']['m100']['median_mlp']}; bigram "
                      f"{r['utility'].get('jsd_bigram')} ({T():.0f}s)", flush=True)
            out[name] = rec; jdump2(out, 'steal2.json')
        if 'select' not in rec:
            rec['select'] = {}
            pl = wm_pairs[:10000]
            frac = []
            for f, _ in pl:
                g = [est.green_fine(s, c) for s, c in f]
                g = [x for x in g if not np.isnan(x)]
                frac.append(np.mean(g) if g else 0.5)
            order = np.argsort(np.array(frac), kind='stable')
            for keep in (0.5, 0.25, 0.1):
                idx = sorted(order[:int(keep * len(order))].tolist())
                X = [RA[i] for i in idx]
                rec['select'][str(keep)] = det(X, sch, prf, sizes=(100, len(X)), n100=len(X))
            print(f"{name} select: {[(k, v['m100']['median_mlp'], v['m100']['frac_le_m3']) for k, v in rec['select'].items()]} ({T():.0f}s)", flush=True)
            out[name] = rec; jdump2(out, 'steal2.json')
    pool.close()


if __name__ == '__main__':
    main()
