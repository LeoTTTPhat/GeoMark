"""Stage 7: re-timing-robust tests (reviewers: re-timestamping by 60 min removes all dataset-level evidence, and the
bin-shift search was evaluated only per trace with a Bonferroni factor).

Shift-union test. For every trace the detector takes the UNION of its (seed, cell) pairs over all candidate start bins
b + j, |j| <= J (J = 2 for 30-min bins, 1 for 120-min bins), and over all bins when timestamps are stripped. The union
is a function of the data only, so by Lemma 1 the per-trace and the pooled tests stay exact; the cost is dilution
(wrong-bin pairs behave as null pairs). Conditions: plain (no search), union_J, union_all; attacks: none, retime15,
retime60 (independent random sign per trace, as in geomark/attacks.py), retime120, strip (time information removed:
only union_all applies). Pools: disjoint 100- and 1000-trace pools of the first 2,000 traces of each seed-0 release.
Null: 3,000 real test traces (retime60) in 100-trace pools under the 3 experiment keys and 200 fresh keys.
Output: results/rev1/shiftpool.json. Model-free. Usage: python src/rev1/shiftpool.py"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import (cfg_rev1, OUT, load_release, jdump, PairIndex, uniforms_vec, log10_binom_sf, log10_gamma_sf,
                          fisher_log10, wilson)
from geomark.common import seed_from, Timer, slot_of
from geomark.scheme import Scheme, seeds_for_cells
from geomark.hexgrid import tokenize, parents
from geomark.detect import score_pairs, LN10
from geomark.prf import PRF
from geomark import attacks
from geomark.data import real_traces
from generate import wm_scheme

GENS = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']
WMS = ['gm_kgw', 'gm_gumbel', 'a_g25_d4', 'a_bin120', 'gm_kgw_eosn', 'exact_kgw_bin']
ATTACKS = ['none', 'retime15', 'retime60', 'retime120', 'strip']


def nbins(sch):
    return 48 * 30 // sch.bin_min if sch.bin_min else 1


def trace_pairs_by_bin(cells, slot, sch):
    """Per bin b: (fine pairs, mid pairs) of the trace if its start bin were b. Masking / history do not depend on b."""
    if len(cells) == 0:
        return {}, None
    seeds, masked = seeds_for_cells(cells, slot, sch)
    keep = [i for i in range(len(cells)) if not masked[i]]
    mids = parents(cells, sch.mid_res) if sch.uses_mid else None
    b0 = sch.bin_of_slot(slot)
    base = [(seeds[i][:-1], int(cells[i]), int(mids[i]) if mids is not None else None) for i in keep]
    return base, b0


def pairs_for_bins(base, bins, uses_mid):
    F, M = {}, {}
    for b in bins:
        for h, c, m in base:
            s = h + (b,)
            F[(s, c)] = None
            if uses_mid:
                M[(s, m)] = None
    return F, M


def cand_bins(b0, sch, cond):
    nb = nbins(sch)
    if cond == 'plain':
        return [b0]
    if cond == 'union_all':
        return list(range(nb))
    J = 2 if sch.bin_min == 30 else 1
    return sorted({(b0 + j) % nb for j in range(-J, J + 1)})


def score_union(F, M, sch, prf):
    r = score_pairs(list(F), list(M), sch, prf)
    return r['lp'], r['n_f']


def run_release(cfg, gen, rid, sch, key, n=2000):
    R = load_release(cfg, gen, rid)
    if R is None:
        return None
    trs = R['traces'][:n]
    prf = PRF(key)
    out = {}
    for attack in ATTACKS:
        bases = []
        for i, tr in enumerate(trs):
            if attack in ('none', 'strip'):
                x = tr
            else:
                x = attacks.apply(tr, attack, np.random.default_rng(seed_from(f'{gen}/{rid}', attack, i)))
            cells = tokenize(x['lat'], x['lon'], sch.res)
            slot = slot_of(x['t'][0], x['ds']) if len(x['t']) else 0
            bases.append(trace_pairs_by_bin(cells, slot, sch))
        conds = ['union_all'] if attack == 'strip' else ['plain', 'union_J', 'union_all']
        for cond in conds:
            if cond != 'plain' and not sch.bin_min:
                continue
            per = [pairs_for_bins(b, cand_bins(b0, sch, cond), sch.uses_mid) if b else ({}, {}) for b, b0 in bases]
            lp_tr = [score_union(F, M, sch, prf)[0] for F, M in per]
            rec = dict(tpr_m3=float(np.mean(np.array(lp_tr) <= -3)), n_traces=len(lp_tr))
            for m in (100, 1000):
                lps = []
                for gi in range(len(per) // m):
                    F, M = {}, {}
                    for f, mm in per[gi * m:(gi + 1) * m]:
                        F.update(f); M.update(mm)
                    lps.append(score_union(F, M, sch, prf)[0])
                rec[f'm{m}'] = dict(lps=[round(float(x), 3) for x in lps], median=float(np.median(lps)) if lps else None,
                                    frac_le_m3=float(np.mean(np.array(lps) <= -3)) if lps else None, n_pools=len(lps))
            out[f'{attack}|{cond}'] = rec
    return out


def null_block(cfg, ds, sch, keys):
    """Real test traces, retime60 (and strip), shift-union pooled tests in 100-trace pools under many keys."""
    trs = real_traces(cfg, ds, 'test', limit=3000, seed=1, min_pts=2)
    res = {}
    for attack in ('retime60', 'strip', 'none'):
        bases = []
        for i, tr in enumerate(trs):
            x = tr if attack in ('none', 'strip') else attacks.apply(tr, attack, np.random.default_rng(seed_from(f'real_{ds}', attack, i)))
            cells = tokenize(x['lat'], x['lon'], sch.res)
            slot = slot_of(x['t'][0], ds) if len(x['t']) else 0
            bases.append(trace_pairs_by_bin(cells, slot, sch))
        for cond in (['union_all'] if attack == 'strip' else ['union_J', 'union_all']):
            pools = []
            for gi in range(len(bases) // 100):
                F, M = {}, {}
                for b, b0 in bases[gi * 100:(gi + 1) * 100]:
                    if b:
                        f, m = pairs_for_bins(b, cand_bins(b0, sch, cond), sch.uses_mid)
                        F.update(f); M.update(m)
                pools.append((list(F), list(M)))
            # vectorised scoring under many keys
            idx = PairIndex()
            encF = [idx.encode(F) for F, _ in pools]
            encM = [idx.encode(M) for _, M in pools]
            lps = []
            cache_f, cache_m = {}, {}
            for key in keys:
                for (sf, cf), (sm, cm) in zip(encF, encM):
                    uf = uniforms_vec(key, 'fine', idx.seeds, sf, cf, cache_f)
                    if sch.score == 'kgw':
                        lf = float(log10_binom_sf(int((uf < sch.gamma).sum()), len(uf), sch.gamma))
                    else:
                        lf = float(log10_gamma_sf(float(-np.log1p(-uf).sum()), len(uf)))
                    if sch.uses_mid and len(cm):
                        um = uniforms_vec(key, 'mid', idx.seeds, sm, cm, cache_m)
                        if sch.score == 'kgw':
                            lm = float(log10_binom_sf(int((um < sch.gamma).sum()), len(um), sch.gamma))
                        else:
                            lm = float(log10_gamma_sf(float(-np.log1p(-um).sum()), len(um)))
                        lp = float(fisher_log10(lf, lm, len(uf), len(um)))
                    else:
                        lp = lf
                    lps.append(lp)
            lps = np.array(lps)
            k3 = int((lps <= -3).sum()); k2 = int((lps <= -2).sum())
            res[f'{attack}|{cond}'] = dict(n_tests=len(lps), n_pools=len(pools), n_keys=len(keys), k_le_m3=k3, k_le_m2=k2,
                                           fpr_m3=k3 / len(lps), fpr_m2=k2 / len(lps), wilson_m2=wilson(k2, len(lps)),
                                           ks_p=float(__import__('scipy.stats', fromlist=['kstest']).kstest(np.clip(10 ** lps, 0, 1), 'uniform').pvalue))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    cfg = cfg_rev1()
    f = os.path.join(OUT, 'shiftpool.json')
    import json
    out = json.load(open(f)) if os.path.exists(f) else {'releases': {}, 'null': {}}
    T = Timer()
    for gen in GENS:
        g = cfg['generators'][gen]
        for wm in WMS:
            for key in (0, 1, 2):
                rid = f'{wm}__k{key}__s0'
                k = f'{gen}|{rid}'
                if k in out['releases']:
                    continue
                sch = wm_scheme(cfg, wm, g['res'])
                r = run_release(cfg, gen, rid, sch, key, n=300 if a.quick else 2000)
                if r is None:
                    continue
                out['releases'][k] = r
                print(f"{k}: retime60 plain {r['retime60|plain']['m100']['median']} union_J {r.get('retime60|union_J', {}).get('m100', {}).get('median')} "
                      f"strip {r.get('strip|union_all', {}).get('m100', {}).get('median')} ({T():.0f}s)", flush=True)
                if not a.quick:
                    jdump(out, 'shiftpool.json')
                if a.quick:
                    return
    keys = [0, 1, 2] + list(range(2000, 2200))
    for ds, gen in (('porto', 'gpt_porto_r9_s0'), ('geolife', 'gpt_geolife_r9_s0')):
        for wm in ('gm_kgw', 'gm_gumbel', 'a_bin120'):
            k = f'{ds}_r9|{wm}'
            if k in out['null'] or (wm == 'a_bin120' and ds == 'geolife'):
                continue
            out['null'][k] = null_block(cfg, ds, wm_scheme(cfg, wm, 9), keys)
            print(f"null {k}: {[(c, v['fpr_m3'], v['fpr_m2']) for c, v in out['null'][k].items()]} ({T():.0f}s)", flush=True)
            jdump(out, 'shiftpool.json')


if __name__ == '__main__':
    main()
