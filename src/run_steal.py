"""E8 watermark stealing, spoofing and scrubbing (adversary S knows the scheme, not the key).
Estimation: from N watermarked traces, per-seed frequency of each (seed, cell) pair (de-duplicated per trace) vs a prior
from real training traces (same seeding); estimated green iff the smoothed frequency ratio exceeds 1. Same at the medium
level. Spoofing: real test traces recoloured towards estimated green cells (vs. oracle key and random controls), then
detected with the TRUE key. Scrubbing: held-out watermarked traces recoloured towards estimated red cells, vs. random
recolouring at the same rates. Corpora: k=1, k=2 and key rotation (4 keys mixed).
Output: <out_dir>/steal/results.jsonl. Usage: python src/run_steal.py --cfg CFG"""
import os, sys, argparse
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, load_pickle, append_jsonl, done_keys, seed_from, Timer
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells, detect_trace, pooled
from geomark.posthoc import recolor_trace, true_key_scorer
from geomark.hexgrid import parent
from geomark.data import release_dir, real_traces
from generate import wm_scheme


class Estimate:
    def __init__(self, sch, wm_pairs, prior_pairs, alpha=0.5):
        self.sch, self.alpha = sch, alpha
        self.w, self.p = Counter(), Counter()
        self.wm_tot, self.pr_tot = Counter(), Counter()
        self.wmid, self.pmid, self.wmid_tot, self.pmid_tot = Counter(), Counter(), Counter(), Counter()
        for f, m in wm_pairs:
            for s, c in f:
                self.w[(s, c)] += 1; self.wm_tot[s] += 1
            for s, P in m:
                self.wmid[(s, P)] += 1; self.wmid_tot[s] += 1
        for f, m in prior_pairs:
            for s, c in f:
                self.p[(s, c)] += 1; self.pr_tot[s] += 1
            for s, P in m:
                self.pmid[(s, P)] += 1; self.pmid_tot[s] += 1

    def _ratio(self, w, p, wt, pt, s, x):
        a = self.alpha
        if wt[s] == 0 or (w[(s, x)] == 0 and p[(s, x)] == 0):
            return np.nan
        return ((w[(s, x)] + a) / (wt[s] + a)) / ((p[(s, x)] + a) / (pt[s] + a))

    def green_fine(self, s, c):
        r = self._ratio(self.w, self.p, self.wm_tot, self.pr_tot, s, int(c))
        return np.nan if np.isnan(r) else float(r > 1)

    def green_mid(self, s, P):
        r = self._ratio(self.wmid, self.pmid, self.wmid_tot, self.pmid_tot, s, int(P))
        return np.nan if np.isnan(r) else float(r > 1)

    def scorer(self):
        sch = self.sch
        def f(seed, cands):
            out = []
            for c in cands:
                gf = self.green_fine(seed, c)
                gm = self.green_mid(seed, parent(c, sch.mid_res)) if sch.uses_mid else np.nan
                out.append((0.5 if np.isnan(gf) else gf) + 0.5 * (0.5 if np.isnan(gm) else gm))
            return np.array(out)
        return f

    def accuracy(self, prf):
        keys = list(self.w)
        if not keys:
            return {}
        s_, c_ = zip(*keys)
        true = prf.pairs('fine', s_, c_) < self.sch.gamma
        est = np.array([self.green_fine(s, c) for s, c in keys])
        wts = np.array([self.w[k] for k in keys], dtype=float)
        ok = ~np.isnan(est)
        est_g = est == 1
        return dict(n_pairs=len(keys), n_seeds=len(self.wm_tot),
                    acc=float((est_g[ok] == true[ok]).mean()), acc_weighted=float(((est_g == true) * wts)[ok].sum() / wts[ok].sum()),
                    precision=float(true[est_g].mean()) if est_g.any() else float('nan'),
                    recall=float(est_g[true].mean()) if true.any() else float('nan'))


def detect_many(trs, sch, prf, m_pool):
    lps = [detect_trace(t, sch, prf)['lp'] for t in trs]
    pl = [pair_sets(*trace_cells(t, sch), sch) for t in trs]
    pooled_lp = [pooled(pl[i:i + m_pool], sch, prf)['lp'] for i in range(0, len(pl) - m_pool + 1, m_pool)]
    return dict(median_lp=float(np.median(lps)), frac_lp_le_m3=float(np.mean(np.array(lps) <= -3)),
                frac_lp_le_m5=float(np.mean(np.array(lps) <= -5)), pooled_lps=pooled_lp, n=len(trs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--n_spoof', type=int, default=None, help='Stage 4: override n_spoof; writes steal/results_spoof<N>.jsonl')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    sc = dict(cfg['steal'])
    if a.n_spoof:
        sc['n_spoof'] = a.n_spoof
    gen = sc['gen']
    g = cfg['generators'][gen]
    out = os.path.join(cfg['out_dir'], 'steal', f'results_spoof{a.n_spoof}.jsonl' if a.n_spoof else 'results.jsonl')
    done = done_keys(out)
    real_train = real_traces(cfg, g['dataset'], 'train', limit=sc['n_prior'], seed=7)
    real_test = real_traces(cfg, g['dataset'], 'test', limit=sc['n_spoof'], seed=8)
    for corp in sc['corpora']:
        sch = wm_scheme(cfg, corp['wm'], g['res'])
        files = [os.path.join(release_dir(cfg, gen), r + '.pkl') for r in corp['releases']]
        if not all(os.path.exists(f) for f in files):
            print('missing corpus', corp['name']); continue
        rels = [load_pickle(f)['traces'] for f in files]
        keys = corp['keys']
        hold = rels[0][-sc['n_scrub']:]                         # held-out watermarked traces (key keys[0])
        pool_ = [rels[r][i] for i in range(max(len(x) for x in rels) - sc['n_scrub']) for r in range(len(rels)) if i < len(rels[r]) - sc['n_scrub']]
        T = Timer()
        prior_pairs = [pair_sets(*trace_cells(t, sch), sch) for t in real_train]
        wm_pairs_all = [pair_sets(t['cells'], t['slot'], sch) for t in pool_[:max(sc['N'])]]
        prfs = [PRF(k) for k in keys]
        rng = np.random.default_rng(seed_from(corp['name'], *([a.n_spoof] if a.n_spoof else [])))
        # references that do not depend on N
        ref_key = f"{corp['name']}|ref"
        if ref_key not in done:
            rec = dict(key=ref_key, corpus=corp['name'])
            rec['clean_heldout'] = detect_many(hold, sch, prfs[0], sc['m_pool'])
            rec['real_unmodified'] = detect_many(real_test, sch, prfs[0], sc['m_pool'])
            orc = [recolor_trace(t, sch, true_key_scorer(sch.with_(method='kgw'), prfs[0]), 'green', 1.0, rng) for t in real_test]
            rec['spoof_oracle'] = detect_many([t for t in orc if t], sch, prfs[0], sc['m_pool'])
            for rate in sc['scrub_rates']:
                rnd = [recolor_trace(t, sch, lambda s, c: rng.random(len(c)), 'green', rate, rng) for t in hold]
                rnd = [t for t in rnd if t]
                rec[f'scrub_random_r{rate}'] = dict(detect_many(rnd, sch, prfs[0], sc['m_pool']), frac_changed=float(np.mean([t['frac_changed'] for t in rnd])))
            append_jsonl(out, rec)
            print(f"{corp['name']} ref: clean {rec['clean_heldout']['median_lp']:.2f}, oracle spoof {rec['spoof_oracle']['frac_lp_le_m3']:.3f} ({T():.0f}s)", flush=True)
        for N in sc['N']:
            k = f"{corp['name']}|N{N}"
            if k in done or N > len(wm_pairs_all):
                continue
            est = Estimate(sch, wm_pairs_all[:N], prior_pairs)
            rec = dict(key=k, corpus=corp['name'], N=N, accuracy=est.accuracy(prfs[0]))
            spo = [recolor_trace(t, sch, est.scorer(), 'green', 1.0, rng) for t in real_test]
            spo = [t for t in spo if t]
            rec['spoof'] = {f'key{kk}': detect_many(spo, sch, prf, sc['m_pool']) for kk, prf in zip(keys, prfs)}
            rec['spoof_frac_changed'] = float(np.mean([t['frac_changed'] for t in spo]))
            for rate in sc['scrub_rates']:
                scr = [recolor_trace(t, sch, est.scorer(), 'red', rate, rng) for t in hold]
                scr = [t for t in scr if t]
                rec[f'scrub_r{rate}'] = dict(detect_many(scr, sch, prfs[0], sc['m_pool']), frac_changed=float(np.mean([t['frac_changed'] for t in scr])))
            append_jsonl(out, rec)
            sk = 'scrub_r%s' % sc['scrub_rates'][-1]
            print(f"{corp['name']} N={N}: acc {rec['accuracy'].get('acc_weighted', float('nan')):.3f}, spoof@1e-3 "
                  f"{rec['spoof']['key%d' % keys[0]]['frac_lp_le_m3']:.3f}, scrub median lp {rec[sk]['median_lp']:.2f} ({T():.0f}s)", flush=True)


if __name__ == '__main__':
    main()
