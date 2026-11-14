"""E6 baselines.
  --prep : build baseline releases from the unwatermarked release of each generator:
           posthoc__k<key>__s<seed>.pkl (post-hoc recolouring with GeoMark's identical coarse hashing, true key) and
           wtrace__k<key>__s<seed>.pkl (W-Trace, Fourier amplitude watermark; originals kept for non-blind detection)
  --eval : W-Trace NCC under the attack list (marked vs. unmarked-null), and the supervised real-vs-synthetic
           classifier (HistGradientBoosting on trace features). (Post-hoc releases are evaluated by evaluate.py.)
Outputs: <out_dir>/gen/<gen>/{posthoc,wtrace}__*.pkl, <out_dir>/baselines/*.parquet|json"""
import os, sys, argparse, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, save_pickle, load_pickle, dump_json, seed_from, Timer, LocalProj
from geomark.prf import PRF
from geomark.posthoc import recolor_trace, true_key_scorer
from geomark import wtrace, attacks
from geomark.data import release_dir, real_traces, load_ds
from geomark.hexgrid import tokenize
from generate import wm_scheme, release_jobs
from evaluate import load_cache


def prep(cfg):
    bc = cfg['baselines']
    sets = [x for x in ('baselines', 'baselines_big') if x in cfg['releases']]   # baselines_big: Stage 4 (10k post-hoc)
    for gen in bc['gens']:
        g = cfg['generators'][gen]
        for j in release_jobs(cfg, gen, sets):
            rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
            out = os.path.join(release_dir(cfg, gen), rid + '.pkl')
            src = os.path.join(release_dir(cfg, gen), f"none__k0__s{j['seed']}.pkl")
            if os.path.exists(out) or not os.path.exists(src):
                continue
            T = Timer()
            S = load_pickle(src)['traces'][:j['n']]
            meth = cfg['wm'][j['wm']]['method']
            sch = wm_scheme(cfg, j['wm'], g['res'])
            rng = np.random.default_rng(seed_from(gen, rid))
            trs = []
            if meth == 'posthoc':
                scorer = true_key_scorer(sch.with_(method='kgw'), PRF(j['key']))
                for tr in S:
                    x = recolor_trace(tr, sch, scorer, 'green', bc.get('posthoc_rate', 1.0), rng)
                    if x is None:
                        x = dict(tr, frac_changed=0.0)
                    trs.append(x)
            else:  # wtrace
                for tr in S:
                    m = wtrace.embed(tr, j['key'])
                    m['orig_lat'], m['orig_lon'] = np.asarray(tr['lat']), np.asarray(tr['lon'])
                    trs.append(m)
            meta = dict(gen=gen, wm=j['wm'], scheme=sch.to_dict(), key=j['key'], seed=j['seed'], n=len(trs), decode='uniform',
                        dataset=g['dataset'], res=g['res'], sec=T(), source=os.path.basename(src))
            save_pickle(dict(meta=meta, traces=trs), out)
            dump_json(meta, out[:-4] + '.meta.json')
            print(f'{gen}/{rid}: {len(trs)} traces ({T():.0f}s)', flush=True)


def wtrace_eval(cfg):
    bc = cfg['baselines']
    od = os.path.join(cfg['out_dir'], 'baselines')
    os.makedirs(od, exist_ok=True)
    for gen in bc['gens']:
        for j in release_jobs(cfg, gen, ['baselines']):
            if cfg['wm'][j['wm']]['method'] != 'wtrace':
                continue
            rid = f"{j['wm']}__k{j['key']}__s{j['seed']}"
            out = os.path.join(od, f'wtrace_{gen}_{rid}.parquet')
            f = os.path.join(release_dir(cfg, gen), rid + '.pkl')
            if os.path.exists(out) or not os.path.exists(f):
                continue
            R = load_pickle(f)['traces']
            pool_real = real_traces(cfg, cfg['generators'][gen]['dataset'], 'test', limit=2000, seed=2)
            rows = []
            for attack in bc['wtrace_attacks']:
                pre = attacks.cache_prefix(attack)
                cm = load_cache(cfg, gen, rid, pre) if pre else None
                cn = load_cache(cfg, gen, f"none__k0__s{j['seed']}", pre) if pre else None
                if pre and (cm is None or cn is None):
                    continue
                n = min(bc['wtrace_n'], len(R), *( [len(cm), len(cn)] if pre else []))
                for i in range(n):
                    tr = R[i]
                    orig = dict(lat=tr['orig_lat'], lon=tr['orig_lon'], t=tr['t'], ds=tr['ds'])
                    rng = np.random.default_rng(seed_from(rid, attack, i))
                    try:
                        xm = attacks.apply(tr, attack, rng, pool_real, {pre: cm[i]} if pre else None)
                        xn = attacks.apply(orig, attack, np.random.default_rng(seed_from(rid, attack, i, 'n')), pool_real, {pre: cn[i]} if pre else None)
                        a_, b_ = wtrace.ncc(xm, orig, j['key']), wtrace.ncc(xn, orig, j['key'])
                    except (TypeError, KeyError):   # failed map-matching output
                        a_, b_ = 0.0, 0.0
                    rows.append(dict(attack=attack, i=i, ncc=a_, ncc_null=b_))
                print(f'  wtrace {gen} {attack}: mean ncc {np.mean([r["ncc"] for r in rows if r["attack"] == attack]):.3f}', flush=True)
            pd.DataFrame(rows).to_parquet(out, index=False)


def features(tr, res, pop, bigram):
    pr = LocalProj.for_dataset(tr['ds'])
    x, y = pr.to_xy(tr['lat'], tr['lon'])
    d = np.hypot(np.diff(x), np.diff(y)) if len(x) > 1 else np.zeros(1)
    ang = np.diff(np.arctan2(np.diff(y), np.diff(x))) if len(x) > 2 else np.zeros(1)
    ang = np.abs((ang + np.pi) % (2 * np.pi) - np.pi)
    cells = tokenize(tr['lat'], tr['lon'], res)
    lp = [pop.get(int(c), 0) for c in cells]
    lb = [bigram.get((int(a), int(b)), 0) for a, b in zip(cells[:-1], cells[1:])]
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    tot = float(d.sum())
    return [len(x), tot, disp, disp / max(tot, 1), float(np.sqrt(((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean())),
            float(d.mean()), float(d.std()), float(np.median(d)), float(ang.mean()), float(ang.std()),
            len(cells), len(set(cells.tolist())) / max(1, len(cells)), float(np.mean(lp)) if lp else 0, float(np.min(lp)) if lp else 0,
            float(np.mean(lb)) if lb else 0, float(np.mean(np.array(lb) == 0)) if lb else 0]


def classifier(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    bc = cfg['baselines']
    od = os.path.join(cfg['out_dir'], 'baselines')
    os.makedirs(od, exist_ok=True)
    for gen in bc['gens']:
        out = os.path.join(od, f'classifier_{gen}.json')
        if os.path.exists(out) and json.load(open(out)):      # Stage 4: an empty result (no s1 release, LLMs) is redone
            continue
        g = cfg['generators'][gen]
        ds, res = g['dataset'], g['res']
        D = load_ds(cfg, ds)
        from collections import Counter
        tr_idx = [i for i, t in enumerate(D['traces']) if t['split'] == 'train']
        pop_c = Counter(int(c) for i in tr_idx for c in D['tok'][res][i])
        tot = sum(pop_c.values())
        pop = {k: np.log(v / tot) for k, v in pop_c.items()}
        bg_c = Counter((int(a), int(b)) for i in tr_idx for a, b in zip(D['tok'][res][i][:-1], D['tok'][res][i][1:]))
        bg = {k: np.log(v) for k, v in bg_c.items()}
        nt, ne = bc['clf_train'], bc['clf_test']
        rtrain = real_traces(cfg, ds, 'val', limit=nt, seed=5)
        rtest = real_traces(cfg, ds, 'test', limit=ne, seed=6)
        res_out = {}
        for wm in bc['clf_wm']:
            fa = os.path.join(release_dir(cfg, gen), f'{wm}__k0__s0.pkl'); fb = os.path.join(release_dir(cfg, gen), f'{wm}__k0__s1.pkl')
            if not os.path.exists(fa):
                continue
            if os.path.exists(fb):
                sa, sb = load_pickle(fa)['traces'][:nt], load_pickle(fb)['traces'][:ne]
            else:   # Stage 4: single sampling seed (LLM generators): disjoint train / test halves of the s0 release
                A_ = load_pickle(fa)['traces']
                sb = A_[-ne:]; sa = A_[:max(0, len(A_) - ne)][:nt]
                if len(sa) < 500:
                    continue
            X = np.array([features(t, res, pop, bg) for t in rtrain + sa]); y = np.r_[np.zeros(len(rtrain)), np.ones(len(sa))]
            Xt = np.array([features(t, res, pop, bg) for t in rtest + sb]); yt = np.r_[np.zeros(len(rtest)), np.ones(len(sb))]
            clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
            s = clf.predict_proba(Xt)[:, 1]
            neg = np.sort(s[yt == 0])[::-1]
            r = dict(auc=float(roc_auc_score(yt, s)), n_train=len(y), n_test=len(yt))
            for a in (1e-2, 1e-3):
                k = max(1, int(np.floor(a * len(neg))))
                thr = neg[k - 1] if k <= len(neg) else neg[-1]
                r[f'tpr_at_fpr{a:g}'] = float((s[yt == 1] > thr).mean())
            res_out[wm] = r
            print(f'classifier {gen} {wm}: {r}', flush=True)
        dump_json(res_out, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--eval', action='store_true')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    if a.prep:
        prep(cfg)
    if a.eval:
        wtrace_eval(cfg)
        if cfg['baselines'].get('classifier', True):
            classifier(cfg)


if __name__ == '__main__':
    main()
