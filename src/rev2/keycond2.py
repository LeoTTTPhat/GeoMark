"""Stage 8: key-CONDITIONAL calibration of the pooled test as a function of dataset size, and two key-conditional
procedures (reviewers: the key-induced offset grows with the pool size; the remedy of Remark 1 was not evaluated and an
empirical p-value from ~100 reference pools cannot reach 1e-3).

For one fixed key K a dataset of a city shares popular (seed, cell) pairs with every other dataset of that city, so its
pooled z-score has a key-dependent mean mu_K(m) and spread sigma_K(m) (m = pool size). We estimate them from the OWNER's
own reference corpus (the real TRAINING split, which the owner holds), with B random m-trace pools, and evaluate:
  raw : the parametric pooled test (exact over the key, Lemma 1);
  V   : variance-inflated / re-centred test per level: t = (z - mu_K)/(sigma_K sqrt(1 + 1/B)) with the finite-population
        correction sigma_K / sqrt(1 - m/N_ref); p = P[t_{B-1} >= t]; fine and medium levels Fisher-combined;
  E   : exclusion of every pair that occurs in at least `f` of the reference traces (document frequency; f = 1e-3, 1e-4),
        then the raw test (still exact over the key: the exclusion set depends on public data only);
  E+V : both.
Null suspects: disjoint m-trace pools of the real TEST split, of an unwatermarked small-GPT release and of the Qwen release,
and of 3,000 map-matched real test traces, all under 200 fixed keys (3000..3199). Power: pools of the watermarked
seed-0 releases (keys 0-2), without attack and after map-matching. Pool sizes m = 100, 1000, 10000. The
exclusion frequencies come from the first half of the training split and the reference pools from the second half.
Reported per cell: FPR at 1e-3 / 1e-2, worst-key FPR, key variance component tau^2(m) estimated from the reference pools
(Var_K of the estimated mu_K minus its sampling variance), and the median -log10 p / detection share of watermarked pools.
Output: results/rev2/keycond2.json. Model-free (hashing). Usage: python src/rev2/keycond2.py"""
import os, sys, math
import numpy as np
from scipy import stats, sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, real_cells, subkeys, ppath, load_pickle, LN10
from geomark.common import Timer, seed_from
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells
from generate import wm_scheme

KEYS = list(range(3000, 3200))
WKEYS = [0, 1, 2]
MS = (100, 1000, 10000)
BREF = {100: 200, 1000: 200, 10000: 100}
FEX = (1e-3, 1e-4)


class Enc:
    def __init__(self):
        self.idx = {}
        self.seeds = {}
        self.sid, self.cell = [], []

    def ids(self, pairs):
        out = np.empty(len(pairs), np.int64)
        for j, (s, c) in enumerate(pairs):
            v = self.idx.get((s, c))
            if v is None:
                v = self.idx[(s, c)] = len(self.cell)
                si = self.seeds.get(s)
                if si is None:
                    si = self.seeds[s] = len(self.seeds)
                self.sid.append(si); self.cell.append(int(c))
            out[j] = v
        return out


def greens(enc, key, domain, gamma):
    seeds = [None] * len(enc.seeds)
    for s, i in enc.seeds.items():
        seeds[i] = s
    if key >= 10 ** 7:
        raise ValueError('fast PRF only')
    a, b = subkeys(key, domain, seeds)
    sid = np.asarray(enc.sid, np.int64); cell = np.asarray(enc.cell, np.int64)
    return (PRF._u(a[sid], b[sid], cell) < gamma).astype(np.float32)


def incidence(pools, n_pairs):
    rows = np.concatenate([np.full(len(p), i) for i, p in enumerate(pools)]) if pools else np.zeros(0, int)
    cols = np.concatenate(pools) if pools else np.zeros(0, int)
    return sparse.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(pools), n_pairs))


def zscores(Minc, G, gamma, keep=None):
    """Minc: pools x pairs incidence; G: pairs x keys green indicators. Returns (z, T, n) arrays pools x keys."""
    if keep is not None:
        Minc = Minc @ sparse.diags(keep.astype(np.float32))
    n = np.asarray(Minc.sum(1)).ravel()[:, None]
    T = np.asarray(Minc @ G)
    z = (T - n * gamma) / np.sqrt(np.maximum(n, 1) * gamma * (1 - gamma))
    return z, T, n


def lp_raw(T, n, gamma):
    return np.where(n > 0, stats.binom.logsf(T - 1, n, gamma) / LN10, 0.0)


def fisher2(lf, lm):
    return stats.chi2.logsf(-2 * LN10 * (lf + lm), 4) / LN10


def lp_t(z, mu, sd, B, fpc):
    sd = np.maximum(sd, 1e-3)
    t = (z - mu) / (sd / math.sqrt(fpc) * math.sqrt(1 + 1 / B))
    return stats.t.logsf(t, B - 1) / LN10


def pools_from(trs_pairs, m, disjoint=True, B=None, rng=None):
    """trs_pairs: list of (fine ids, mid ids). Disjoint consecutive pools, or B random m-subsets."""
    n = len(trs_pairs)
    if disjoint:
        groups = [list(range(i * m, (i + 1) * m)) for i in range(n // m)]
    else:
        if m > n:
            return [], []
        groups = [rng.choice(n, size=m, replace=False) for _ in range(B)]
    F = [np.unique(np.concatenate([trs_pairs[j][0] for j in g])) for g in groups]
    M = [np.unique(np.concatenate([trs_pairs[j][1] for j in g])) for g in groups]
    return F, M


def summ(LP, alpha):
    k = (LP <= math.log10(alpha)).sum(1)
    J = LP.shape[1]
    return dict(fpr=float(k.sum() / LP.size), worst_key=float(k.max() / J), n_tests=int(LP.size), n_pools=int(J))


def main():
    cfg = cfg_rev2()
    out = jload2('keycond2.json')
    T = Timer()
    combos = [('porto', 'gm_kgw', ['gm_kgw', 'gm_kgw_eosn']), ('porto', 'gm_kgw_sn', ['gm_kgw_sn']), ('porto', 'a_k0', ['a_k0']),
              ('porto', 'exact_kgw_bin', ['exact_kgw_bin']), ('porto', 'exact_kgw', ['exact_kgw']),
              ('geolife', 'gm_kgw', ['gm_kgw', 'gm_kgw_eosn']), ('geolife', 'gm_kgw_sn', ['gm_kgw_sn'])]
    for ds, det, wms in combos:
        ck = f'{ds}|{det}'
        if ck in out:
            continue
        sch = wm_scheme(cfg, det, 9)
        gpt, qwen = f'gpt_{ds}_r9_s0', f'qwen15_{ds}_r9'
        enc_f, enc_m = Enc(), Enc()

        def enc_trace(cells, slot):
            f, m_ = pair_sets(cells, slot, sch)
            return enc_f.ids(f), enc_m.ids(m_)
        rc, rs = real_cells(cfg, ds, 9, 'train')
        ref = [enc_trace(c, s) for c, s in zip(rc, rs)]
        Nref = len(ref)
        tc, ts = real_cells(cfg, ds, 9, 'test')
        sus = {'real_test': [enc_trace(c, s) for c, s in zip(tc, ts)]}
        for name, gen in (('gpt_none', gpt), ('qwen_none', qwen)):
            R = load_release(cfg, gen, 'none__k0__s0')
            if R is not None:
                sus[name] = [enc_trace(t['cells'], t['slot']) for t in R['traces']]
        f = ppath('results', 'attacked', f'real_{ds}', 'test__mm.pkl')
        if os.path.exists(f):
            mm = [t for t in load_pickle(f)['traces'] if t is not None and len(t['lat'])]
            sus['real_test_mm'] = [enc_trace(*trace_cells(t, sch)) for t in mm]
        wmp = {}
        for wm in wms:
            for key in WKEYS:
                for seed in (0, 10):
                    R = load_release(cfg, gpt, f'{wm}__k{key}__s{seed}')
                    if R is None:
                        continue
                    trs = R['traces'][:10000]
                    wmp[(wm, key, 'none', seed)] = [enc_trace(t['cells'], t['slot']) for t in trs]
                fm = ppath('results', 'attacked', gpt, f'{wm}__k{key}__s0__mm.pkl')
                if os.path.exists(fm):
                    C = load_pickle(fm)['traces']
                    wmp[(wm, key, 'mm', 0)] = [enc_trace(*trace_cells(t, sch)) if t is not None and len(t['lat']) else
                                               (np.zeros(0, np.int64), np.zeros(0, np.int64)) for t in C]
        nF, nM = len(enc_f.cell), len(enc_m.cell)
        # reference document frequency of every pair (for exclusion); pairs absent from the reference have frequency 0
        # (computed on the first half of the reference corpus; reference pools are drawn from the second half, so that
        # the exclusion rule and the null estimates use disjoint data)
        hA, ref = ref[:Nref // 2], ref[Nref // 2:]
        Nref = len(ref)
        df_f = np.bincount(np.concatenate([r[0] for r in hA]), minlength=nF) / len(hA)
        df_m = np.bincount(np.concatenate([r[1] for r in hA]), minlength=nM) / len(hA)
        print(f'{ck}: {Nref} ref traces, {nF} fine / {nM} mid pairs ({T():.0f}s)', flush=True)
        rng = np.random.default_rng(seed_from('keycond2', ds, det))
        res = dict(n_ref_pools_corpus=Nref, n_ref_exclusion_corpus=len(hA), n_pairs_fine=nF, n_pairs_mid=nM, sizes={})
        for m in MS:
            B = BREF[m]
            if m > Nref / 3:
                continue
            fpc = 1 - m / Nref
            RF, RM = pools_from(ref, m, disjoint=False, B=B, rng=rng)
            blocks = {'ref': (RF, RM)}
            for sn, sp in sus.items():
                F_, M_ = pools_from(sp, m)
                if F_:
                    blocks[sn] = (F_, M_)
            wblocks = {}
            for (wm, key, att, seed), sp in wmp.items():
                if m == 10000 and seed == 0 and (wm, key, att, 10) in wmp:
                    continue
                if m < 10000 and seed == 10:
                    continue
                F_, M_ = pools_from(sp, m)
                if F_:
                    wblocks[(wm, key, att)] = (F_, M_)
            keepsets = {'raw': (None, None)}
            for fx in FEX:
                keepsets[f'E{fx:g}'] = (df_f < fx, df_m < fx)
            rec = dict(B=B, fpc=fpc, null={}, power={}, tau2={}, excluded_share={})
            for kname, (kf, km) in keepsets.items():
                if kf is not None:
                    rec['excluded_share'][kname] = dict(fine_pairs=float(1 - kf.mean()), mid_pairs=float(1 - km.mean()))
            # null: 200 fixed keys, in batches
            null_lp = {kn: {b: {'raw': [], 'V': []} for b in blocks if b != 'ref'} for kn in keepsets}
            ref_mu = {kn: [] for kn in keepsets}
            for k0 in range(0, len(KEYS), 50):
                ks = KEYS[k0:k0 + 50]
                Gf = np.stack([greens(enc_f, k, 'fine', sch.gamma) for k in ks], 1)
                Gm = np.stack([greens(enc_m, k, 'mid', sch.gamma) for k in ks], 1) if sch.uses_mid else None
                for kn, (kf, km) in keepsets.items():
                    zf_r, _, _ = zscores(incidence(blocks['ref'][0], nF), Gf, sch.gamma, kf)
                    muf, sdf = zf_r.mean(0), zf_r.std(0, ddof=1)
                    if sch.uses_mid:
                        zm_r, _, _ = zscores(incidence(blocks['ref'][1], nM), Gm, sch.gamma, km)
                        mum, sdm = zm_r.mean(0), zm_r.std(0, ddof=1)
                    ref_mu[kn].append(np.c_[muf, sdf ** 2] if not sch.uses_mid else np.c_[muf, sdf ** 2, mum, sdm ** 2])
                    for b, (F_, M_) in blocks.items():
                        if b == 'ref':
                            continue
                        zf, Tf, nf = zscores(incidence(F_, nF), Gf, sch.gamma, kf)
                        lf = lp_raw(Tf, nf, sch.gamma); vf = lp_t(zf, muf, sdf, B, fpc)
                        if sch.uses_mid:
                            zm, Tm, nm = zscores(incidence(M_, nM), Gm, sch.gamma, km)
                            lm = lp_raw(Tm, nm, sch.gamma); vm = lp_t(zm, mum, sdm, B, fpc)
                            lraw, lv = fisher2(lf, lm), fisher2(vf, vm)
                        else:
                            lraw, lv = lf, vf
                        null_lp[kn][b]['raw'].append(lraw.T); null_lp[kn][b]['V'].append(lv.T)
            for kn in keepsets:
                A = np.concatenate(ref_mu[kn], 0)                     # keys x (mu, var[, mu_m, var_m])
                rec['tau2'][kn] = dict(fine=float(A[:, 0].var(ddof=1) - A[:, 1].mean() / B), within_var_fine=float(A[:, 1].mean()),
                                       mid=float(A[:, 2].var(ddof=1) - A[:, 3].mean() / B) if A.shape[1] > 2 else None)
                for b in null_lp[kn]:
                    for proc in ('raw', 'V'):
                        LP = np.concatenate(null_lp[kn][b][proc], 0)          # keys x pools
                        rec['null'][f'{kn}|{proc}|{b}'] = dict(a1e3=summ(LP, 1e-3), a1e2=summ(LP, 1e-2))
            # power under the true keys 0..2
            for key in WKEYS:
                Gf = greens(enc_f, key, 'fine', sch.gamma)[:, None]
                Gm = greens(enc_m, key, 'mid', sch.gamma)[:, None] if sch.uses_mid else None
                for kn, (kf, km) in keepsets.items():
                    zf_r, _, _ = zscores(incidence(blocks['ref'][0], nF), Gf, sch.gamma, kf)
                    muf, sdf = zf_r.mean(0), zf_r.std(0, ddof=1)
                    if sch.uses_mid:
                        zm_r, _, _ = zscores(incidence(blocks['ref'][1], nM), Gm, sch.gamma, km)
                        mum, sdm = zm_r.mean(0), zm_r.std(0, ddof=1)
                    for (wm, k_, att), (F_, M_) in wblocks.items():
                        if k_ != key:
                            continue
                        zf, Tf, nf = zscores(incidence(F_, nF), Gf, sch.gamma, kf)
                        lf = lp_raw(Tf, nf, sch.gamma); vf = lp_t(zf, muf, sdf, B, fpc)
                        if sch.uses_mid:
                            zm, Tm, nm = zscores(incidence(M_, nM), Gm, sch.gamma, km)
                            lm = lp_raw(Tm, nm, sch.gamma); vm = lp_t(zm, mum, sdm, B, fpc)
                            lraw, lv = fisher2(lf, lm).ravel(), fisher2(vf, vm).ravel()
                        else:
                            lraw, lv = lf.ravel(), vf.ravel()
                        for proc, L in (('raw', lraw), ('V', lv)):
                            rec['power'].setdefault(f'{kn}|{proc}|{wm}|{att}', []).extend([round(float(x), 3) for x in L])
            for k_, v in list(rec['power'].items()):
                a = np.array(v)
                rec['power'][k_] = dict(n_pools=len(a), median_mlp=float(-np.median(a)), frac_le_m3=float((a <= -3).mean()), lps=v)
            res['sizes'][str(m)] = rec
            r0 = rec['null'].get('raw|raw|real_test', {}).get('a1e2', {})
            rv = rec['null'].get('raw|V|real_test', {}).get('a1e2', {})
            print(f"{ck} m={m}: tau2 {rec['tau2']['raw']['fine']:.3f}; real_test raw FPR@1e-2 {r0.get('fpr')} worst {r0.get('worst_key')}; "
                  f"V {rv.get('fpr')} worst {rv.get('worst_key')} ({T():.0f}s)", flush=True)
        out[ck] = res
        jdump2(out, 'keycond2.json')


if __name__ == '__main__':
    main()
