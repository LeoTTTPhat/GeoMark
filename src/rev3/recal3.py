"""Stage 10 (round-3 audit): reference-calibrated re-scoring of the release-size attack results, a calibrated all-bin
union for timestamp removal, a width-agnostic jitter detector, and a per-key validity check of the reference-calibrated
test (reviewers: every 'the attack fails at release size' claim used the parametric test, which is not controlled for one
key at large m; the calibrated test was called nominal although only its FPR averaged over keys was shown).

Parts (each checkpointed into results/rev3/recal3.json):
  perkey : m = 100, Porto GeoMark-KGW / step-neutral and GeoLife GeoMark-KGW: the 275 (GeoLife: 194) unwatermarked null
           pools of src/rev2/keycond2.py under each of 200 fixed keys, with the same reference pools (same RNG stream);
           per key: number of pools flagged at 0.05 / 0.01 vs Binomial(n_pools, alpha), dispersion index, and a one-sided KS
           test (too many small p-values) of the key's p-values; parametric and reference-calibrated.
  hop    : boundary hopping / adaptive hopping releases of results/rev2/breakatk (key 0): parametric and calibrated
           (upper, two-sided) at 100 / 1,000 / 10,000 traces (plain pair sets; the reverse-hop defence is not calibrated).
  dil    : dilution datasets of src/rev2/breakatk.py (identical random draws), parametric and calibrated.
  strip  : timestamp removal with the all-bin union (keys 0-2, GeoMark-KGW 10^4 and step-neutral 5,000-trace releases):
           reference pools of the owner's training data passed through the SAME union (key-conditional calibration of the
           union); plus the calibrated union's null on real test pools under keys 0-2 and 50 fixed null keys (FPR, worst key).
  jitter : width-agnostic union for start-time jitter: min over J in {2, 4, 6} with a Bonferroni factor 3 (parametric).
  hopkeys: boundary hopping and 5% / 10% dilution for the 10^4-trace releases of keys 1-4 (GeoMark-KGW, step-neutral),
           parametric and calibrated under each release's own key (run with --parts hopkeys after gen).
Output: results/rev3/recal3.json. Model-free. Usage: python src/rev3/recal3.py [--parts perkey,hop,dil,strip,jitter,hopkeys]"""
import os, sys, math, json, argparse
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r3common import (cfg_rev3, load_release, load_pickle, ppath, dump_json, OUT3, BREF, PairCodec, union, ref_pools,
                      ref_stats, score, summ, fisher, two, LN10, _tt)
from r2common import real_cells as real_cells_slots, subkeys
from geomark.common import seed_from, Timer
from geomark.prf import PRF
from geomark.detect import pair_sets, trace_cells
from geomark.data import real_traces
from generate import wm_scheme
import keycond2 as KC
import timeattack as TA

OUTF = os.path.join(OUT3, 'recal3.json')
GEN = 'gpt_porto_r9_s0'


def load_out():
    return json.load(open(OUTF)) if os.path.exists(OUTF) else {}


def save_out(o):
    dump_json(o, OUTF)


# ------------------------------------------------------------------ perkey
def part_perkey(out):
    cfg = cfg_rev3()
    res = out.get('perkey', {})
    T = Timer()
    for ds, det in (('porto', 'gm_kgw'), ('porto', 'gm_kgw_sn'), ('geolife', 'gm_kgw')):
        ck = f'{ds}|{det}'
        if ck in res:
            continue
        sch = wm_scheme(cfg, det, 9)
        enc_f, enc_m = KC.Enc(), KC.Enc()

        def enc_trace(cells, slot):
            f, m_ = pair_sets(cells, slot, sch)
            return enc_f.ids(f), enc_m.ids(m_)
        rc, rs = real_cells_slots(cfg, ds, 9, 'train')
        ref = [enc_trace(c, s) for c, s in zip(rc, rs)]
        tc, ts = real_cells_slots(cfg, ds, 9, 'test')
        sus = {'real_test': [enc_trace(c, s) for c, s in zip(tc, ts)]}
        for name, gen in (('gpt_none', f'gpt_{ds}_r9_s0'), ('qwen_none', f'qwen15_{ds}_r9')):
            R = load_release(cfg, gen, 'none__k0__s0')
            if R is not None:
                sus[name] = [enc_trace(t['cells'], t['slot']) for t in R['traces']]
        f = ppath('results', 'attacked', f'real_{ds}', 'test__mm.pkl')
        if os.path.exists(f):
            mm = [t for t in load_pickle(f)['traces'] if t is not None and len(t['lat'])]
            sus['real_test_mm'] = [enc_trace(*trace_cells(t, sch)) for t in mm]
        nF, nM = len(enc_f.cell), len(enc_m.cell)
        Nref = len(ref)
        ref = ref[Nref // 2:]
        Nref = len(ref)
        rng = np.random.default_rng(seed_from('keycond2', ds, det))       # same stream as keycond2 -> same reference pools
        m, B = 100, KC.BREF[100]
        fpc = 1 - m / Nref
        RF, RM = KC.pools_from(ref, m, disjoint=False, B=B, rng=rng)
        blocks = {}
        for sn, sp in sus.items():
            F_, M_ = KC.pools_from(sp, m)
            if F_:
                blocks[sn] = (F_, M_)
        LPr, LPv, src = [], [], []
        for k0 in range(0, len(KC.KEYS), 50):
            ks = KC.KEYS[k0:k0 + 50]
            Gf = np.stack([KC.greens(enc_f, k, 'fine', sch.gamma) for k in ks], 1)
            Gm = np.stack([KC.greens(enc_m, k, 'mid', sch.gamma) for k in ks], 1)
            zf_r, _, _ = KC.zscores(KC.incidence(RF, nF), Gf, sch.gamma)
            zm_r, _, _ = KC.zscores(KC.incidence(RM, nM), Gm, sch.gamma)
            muf, sdf, mum, sdm = zf_r.mean(0), zf_r.std(0, ddof=1), zm_r.mean(0), zm_r.std(0, ddof=1)
            r_, v_ = [], []
            for b, (F_, M_) in blocks.items():
                zf, Tf, nf = KC.zscores(KC.incidence(F_, nF), Gf, sch.gamma)
                zm, Tm, nm = KC.zscores(KC.incidence(M_, nM), Gm, sch.gamma)
                r_.append(KC.fisher2(KC.lp_raw(Tf, nf, sch.gamma), KC.lp_raw(Tm, nm, sch.gamma)).T)
                v_.append(KC.fisher2(KC.lp_t(zf, muf, sdf, B, fpc), KC.lp_t(zm, mum, sdm, B, fpc)).T)
                if k0 == 0:
                    src += [b] * len(F_)
            LPr.append(np.concatenate(r_, 1)); LPv.append(np.concatenate(v_, 1))
        LPr, LPv = np.concatenate(LPr, 0), np.concatenate(LPv, 0)          # keys x pools (log10 p)
        n_pools = LPr.shape[1]
        rec = dict(n_keys=LPr.shape[0], n_pools_per_key=n_pools, pools_by_source={b: src.count(b) for b in blocks})
        srcs = np.array(src)
        variants = [('parametric', LPr), ('calibrated', LPv)]
        for b in ('real_test', 'gpt_none'):                       # one population at a time (Stage 10b)
            if (srcs == b).sum() >= 50:
                variants += [(f'parametric|{b}', LPr[:, srcs == b]), (f'calibrated|{b}', LPv[:, srcs == b])]
        for proc, LP in variants:
            n_pools = LP.shape[1]
            P = 10.0 ** LP
            e = {'n_pools': n_pools}
            for a in (0.05, 0.01, 0.001):
                c = (P <= a).sum(1)
                q99 = stats.binom.ppf(0.99, n_pools, a)
                e[f'{a:g}'] = dict(mean_count=float(c.mean()), expected=n_pools * a, var_ratio=float(c.var(ddof=1) / (n_pools * a * (1 - a))),
                                   max_count=int(c.max()), binom_q99=float(q99), frac_keys_above_q99=float((c > q99).mean()),
                                   worst_key_share=float(c.max() / n_pools), fpr=float(c.sum() / P.size))
            ksp = np.array([stats.kstest(P[i], 'uniform', alternative='greater').pvalue for i in range(P.shape[0])])
            e['ks_greater'] = dict(frac_p_lt_0p05=float((ksp < 0.05).mean()), frac_p_lt_0p01=float((ksp < 0.01).mean()),
                                   median_p=float(np.median(ksp)))
            rec[proc] = e
        res[ck] = rec
        print(f"perkey {ck}: pools/key {n_pools}; " + '; '.join(
            f"{p}: 0.05 var ratio {rec[p]['0.05']['var_ratio']:.2f}, keys>q99 {rec[p]['0.05']['frac_keys_above_q99']:.3f}, KS<0.05 {rec[p]['ks_greater']['frac_p_lt_0p05']:.3f}"
            for p in ('parametric', 'calibrated')) + f' ({T():.0f}s)', flush=True)
        out['perkey'] = res
        save_out(out)


# ------------------------------------------------------------------ hop + dilution
def _calib_setup(cfg, sch, name, sizes):
    codec = PairCodec(sch)
    rc, rs = real_cells_slots(cfg, 'porto', 9, 'train')
    ref = [codec.enc(pair_sets(c, s, sch)) for c, s in zip(rc, rs)]
    ref = ref[len(ref) // 2:]
    rp = {m: ref_pools(ref, m, BREF[m], np.random.default_rng(seed_from('rev3-ref', name, m))) for m in sizes}
    return codec, rp, len(ref)


def part_hop_dil(out, parts):
    cfg = cfg_rev3()
    T = Timer()
    for name, wm in (('k1', 'gm_kgw'), ('eosn', 'gm_kgw_eosn'), ('sn', 'gm_kgw_sn')):
        need = [p for p in parts if p in ('hop', 'dil') and name not in out.get(p, {})]
        if not need:
            continue
        sch = wm_scheme(cfg, wm, 9)
        codec, rp, n_ref = _calib_setup(cfg, sch, name, (100, 1000, 10000))
        enc = {}
        if 'hop' in need:
            for var in ('clean', 'hopfm', 'hoprand'):
                X, disp, near = load_pickle(ppath('results', 'rev2', 'breakatk', f'{name}_{var}.pkl'))
                enc[var] = [codec.enc(pair_sets(*trace_cells(t, sch), sch)) for t in X]
        if 'dil' in need:
            R = load_release(cfg, GEN, f'{wm}__k0__s10')
            RA = R['traces'][:10000]
            F = load_release(cfg, GEN, 'none__k0__s20')['traces']
            plw = [codec.enc(pair_sets(t['cells'], t['slot'], sch)) for t in RA]
            plf = [codec.enc(pair_sets(t['cells'], t['slot'], sch)) for t in F]
        Gf, Gm = codec.greens(0)
        rs = {m: ref_stats(rp[m], Gf, Gm, sch.gamma, BREF[m], n_ref, m) for m in rp}
        if 'hop' in need:
            h = {}
            for var, E in enc.items():
                r = {}
                for m, nmax in ((100, 2000), (1000, None), (10000, None)):
                    src = E[:nmax] if nmax else E
                    pools = [union(src[i:i + m]) for i in range(0, len(src) - m + 1, m)]
                    sc = [score(p, Gf, Gm, sch.gamma, rs[m]) for p in pools]
                    r[f'm{m}'] = {f: summ(sc, f) for f in ('up', 'two', 'cal_up', 'cal_two')}
                    r[f'm{m}']['zf'] = float(np.median([s['zf'] for s in sc])); r[f'm{m}']['zm'] = float(np.median([s['zm'] for s in sc]))
                h[var] = r
            out.setdefault('hop', {})[name] = h
            print(f"hop {name}: " + '; '.join(f"{v} m10000 up {h[v]['m10000']['up']['median_mlp']:.1f} cal {h[v]['m10000']['cal_up']['median_mlp']:.1f}" for v in h) + f' ({T():.0f}s)', flush=True)
        if 'dil' in need:
            d = {}
            for size, reps in ((1000, 10), (10000, 3)):
                for dd in (0.01, 0.02, 0.05, 0.1, 0.2):
                    nw = int(round(dd * size))
                    sc = []
                    for j in range(reps):
                        rng = np.random.default_rng(seed_from('dilution', name, size, dd, j))
                        wi = rng.choice(len(plw), nw, replace=False); fi = rng.choice(len(plf), size - nw, replace=False)
                        sc.append(score(union([plw[i] for i in wi] + [plf[i] for i in fi]), Gf, Gm, sch.gamma, rs[size]))
                    d[f'{size}|{dd}'] = {f: summ(sc, f) for f in ('up', 'two', 'cal_up', 'cal_two')}
            out.setdefault('dil', {})[name] = d
            print(f"dil {name}: " + ', '.join(f"{k}: {v['up']['median_mlp']:.1f}/{v['cal_up']['median_mlp']:.1f}" for k, v in d.items()) + f' ({T():.0f}s)', flush=True)
        save_out(out)


# ------------------------------------------------------------------ hopping and dilution under keys 1-4 (Stage 10b)
def part_hopkeys(out):
    """Boundary hop (the plain pair set is the same for the adaptive hop, whose cells are identical) and dilution for the
    10^4-trace releases of keys 1-4 (src/configs/rev3.yaml, set rev3_scrub), GeoMark-KGW and step-neutral; parametric and
    reference-calibrated under each release's own key."""
    from multiprocessing import Pool
    import breakatk as BA
    from geomark.common import scaled
    cfg = cfg_rev3()
    res = out.get('hopkeys', {})
    T = Timer()
    for name, wm in (('k1', 'gm_kgw'), ('sn', 'gm_kgw_sn')):
        if name in res:
            continue
        sch = wm_scheme(cfg, wm, 9)
        codec, rp, n_ref = _calib_setup(cfg, sch, name, (100, 1000, 10000))
        F = load_release(cfg, GEN, 'none__k0__s20')['traces']
        plf = [codec.enc(pair_sets(t['cells'], t['slot'], sch)) for t in F]
        rec = {}
        with Pool(scaled(16, 1), initializer=BA._init, initargs=(sch,)) as pool:
            for key in (1, 2, 3, 4):
                R = load_release(cfg, GEN, f'{wm}__k{key}__s10')
                RA = R['traces'][:10000]
                rs_ = pool.map(BA._hop, [(t, 'hopfm', seed_from('rev3-hop', name, key, i)) for i, t in enumerate(RA)], chunksize=20)
                E = {'clean': [codec.enc(pair_sets(*trace_cells(t, sch), sch)) for t in RA],
                     'hop': [codec.enc(pair_sets(*trace_cells(r[0], sch), sch)) for r in rs_]}
                plw = [codec.enc(pair_sets(t['cells'], t['slot'], sch)) for t in RA]
                Gf, Gm = codec.greens(key)
                rs = {m: ref_stats(rp[m], Gf, Gm, sch.gamma, BREF[m], n_ref, m) for m in rp}
                r = dict(mean_disp_m=float(np.mean([x[1] for x in rs_])))
                for var, Ev in E.items():
                    for m, nmax in ((100, 2000), (1000, None), (10000, None)):
                        src = Ev[:nmax] if nmax else Ev
                        pools = [union(src[i:i + m]) for i in range(0, len(src) - m + 1, m)]
                        sc = [score(p, Gf, Gm, sch.gamma, rs[m]) for p in pools]
                        r[f'{var}|m{m}'] = {f: summ(sc, f) for f in ('up', 'cal_up')}
                for dd in (0.05, 0.1):
                    nw = int(round(dd * 10000)); sc = []
                    for j in range(3):
                        rng = np.random.default_rng(seed_from('rev3-dilution', name, key, dd, j))
                        wi = rng.choice(len(plw), nw, replace=False); fi = rng.choice(len(plf), 10000 - nw, replace=False)
                        sc.append(score(union([plw[i] for i in wi] + [plf[i] for i in fi]), Gf, Gm, sch.gamma, rs[10000]))
                    r[f'dil|{dd}'] = {f: summ(sc, f) for f in ('up', 'cal_up')}
                rec[str(key)] = r
                print(f"hopkeys {name} k{key}: hop m100 {r['hop|m100']['up']['median_mlp']:.1f}/{r['hop|m100']['cal_up']['median_mlp']:.1f} "
                      f"m10000 {r['hop|m10000']['up']['median_mlp']:.1f}/{r['hop|m10000']['cal_up']['median_mlp']:.1f}; clean m10000 cal "
                      f"{r['clean|m10000']['cal_up']['median_mlp']:.1f}; dil 5% {r['dil|0.05']['up']['median_mlp']:.1f}/{r['dil|0.05']['cal_up']['median_mlp']:.1f} "
                      f"10% {r['dil|0.1']['up']['median_mlp']:.1f}/{r['dil|0.1']['cal_up']['median_mlp']:.1f} ({T():.0f}s)", flush=True)
        res[name] = rec
        out['hopkeys'] = res
        save_out(out)


# ------------------------------------------------------------------ strip (all-bin union), calibrated
class UScorer:
    def __init__(self, base, key):
        cl = np.empty(len(base.cells), np.int64)
        for c, i in base.cells.items():
            cl[i] = c
        self.cells = cl
        seeds = [h + (b,) for h in base.H for b in range(48)]
        self.ab = {d: subkeys(key, d, seeds) for d in ('fine', 'mid')}

    def counts(self, F, M, gamma):
        r = []
        for d, codes in (('fine', F), ('mid', M)):
            if len(codes) == 0:
                r += [0.0, 0]; continue
            s = codes // 10 ** 7; c = self.cells[codes % 10 ** 7]
            a, b = self.ab[d]
            u = PRF._u(a[s], b[s], c)
            r += [float((u < gamma).sum()), len(u)]
        return r


def uscore(cnt, gamma, rs=None):
    Tf, nf, Tm, nm = cnt
    zf = (Tf - nf * gamma) / math.sqrt(nf * gamma * (1 - gamma)) if nf else 0.0
    zm = (Tm - nm * gamma) / math.sqrt(nm * gamma * (1 - gamma)) if nm else 0.0
    uf = float(stats.binom.logsf(Tf - 1, nf, gamma) / LN10) if nf else 0.0
    um = float(stats.binom.logsf(Tm - 1, nm, gamma) / LN10) if nm else 0.0
    r = dict(zf=zf, zm=zm, up=fisher(uf, um, nf, nm))
    if rs is not None:
        cuf, clf = _tt(zf, rs['muf'], rs['sdf'], rs['B'], rs['fpc'])
        cum, clm = _tt(zm, rs['mum'], rs['sdm'], rs['B'], rs['fpc'])
        r['cal_up'] = fisher(cuf, cum, nf, nm)
    return r


def part_strip(out):
    cfg = cfg_rev3()
    res = out.get('strip', {})
    T = Timer()
    ALL = list(range(48))
    for wm, n_rel in (('gm_kgw', 10000), ('gm_kgw_sn', 5000)):
        if wm in res:
            continue
        sch = wm_scheme(cfg, wm, 9)
        base = TA.Base(sch)
        rc, rs = real_cells_slots(cfg, 'porto', 9, 'train')
        ref = [base.enc(c, s) for c, s in zip(rc, rs)]
        ref = ref[len(ref) // 2:]
        n_ref = len(ref)
        real = real_traces(cfg, 'porto', 'test')
        null_items = [base.enc(*TA._cells_slot(t, sch)) for t in real]
        rel = {}
        for key in (0, 1, 2):
            R = load_release(cfg, GEN, f'{wm}__k{key}__s0')
            rel[key] = [base.enc(t['cells'], t['slot']) for t in R['traces']]
        keys_null = [0, 1, 2] + TA.NULL_KEYS
        scorers = {k: UScorer(base, k) for k in keys_null}
        print(f'strip {wm}: {n_ref} ref traces, {len(base.H)} histories ({T():.0f}s)', flush=True)
        g = sch.gamma
        sizes = {100: 200, 1000: 100, n_rel: 50}
        refcnt = {}                                                    # m -> key -> list of counts
        for m, B in sizes.items():
            rng = np.random.default_rng(seed_from('rev3-strip-ref', wm, m))
            kk = keys_null if m in (100, 1000) else [0, 1, 2]
            refcnt[m] = {k: [] for k in kk}
            for _ in range(B):
                gidx = rng.choice(n_ref, size=m, replace=False)
                F, M = TA.pool_pairs([ref[j] for j in gidx], [ALL] * m)
                for k in kk:
                    refcnt[m][k].append(scorers[k].counts(F, M, g))
            print(f'  ref m={m} B={B} done ({T():.0f}s)', flush=True)

        def rstat(m, k):
            c = np.array(refcnt[m][k])
            zf = (c[:, 0] - c[:, 1] * g) / np.sqrt(c[:, 1] * g * (1 - g))
            zm = (c[:, 2] - c[:, 3] * g) / np.sqrt(np.maximum(c[:, 3], 1) * g * (1 - g))
            return dict(muf=float(zf.mean()), sdf=float(zf.std(ddof=1)), mum=float(zm.mean()), sdm=float(zm.std(ddof=1)),
                        B=len(c), fpc=1 - m / n_ref)
        rec = dict(n_ref=n_ref, ref_B={str(k): v for k, v in sizes.items()}, releases={}, null={})
        for key in (0, 1, 2):
            items = rel[key]
            r = {}
            for lab, m, nmax in (('m100', 100, 2000), ('m1000', 1000, None), ('whole', n_rel, None)):
                src = items[:nmax] if nmax else items[:n_rel]
                st = rstat(m, key)
                sc = []
                for i in range(0, len(src) - m + 1, m):
                    F, M = TA.pool_pairs(src[i:i + m], [ALL] * m)
                    sc.append(uscore(scorers[key].counts(F, M, g), g, st))
                r[lab] = {f: summ(sc, f) for f in ('up', 'cal_up')}
                r[lab]['ref_sd_f'] = st['sdf']
            rec['releases'][str(key)] = r
        for m, nuse in ((100, 3000), (1000, len(null_items))):
            pools = [TA.pool_pairs(null_items[i:i + m], [ALL] * m) for i in range(0, nuse - m + 1, m)]
            LPu, LPc = [], []
            for k in keys_null:
                st = rstat(m, k)
                s_ = [uscore(scorers[k].counts(F, M, g), g, st) for F, M in pools]
                LPu.append([x['up'] for x in s_]); LPc.append([x['cal_up'] for x in s_])
            e = {}
            for proc, LP in (('parametric', np.array(LPu)), ('calibrated', np.array(LPc))):
                e[proc] = {}
                for a in (1e-2, 1e-3):
                    k_ = (LP <= math.log10(a)).sum(1)
                    e[proc][f'{a:g}'] = dict(fpr=float(k_.sum() / LP.size), worst_key=float(k_.max() / LP.shape[1]))
            rec['null'][f'm{m}'] = dict(n_keys=len(keys_null), n_pools=len(pools), **e)
        res[wm] = rec
        print(f"strip {wm}: " + '; '.join(f"k{k} " + '/'.join(f"{v[l]['up']['median_mlp']:.1f}->{v[l]['cal_up']['median_mlp']:.1f}" for l in ('m100', 'm1000', 'whole'))
                                         for k, v in rec['releases'].items())
              + '; null ' + '; '.join(f"{m}: param {v['parametric']['0.01']['worst_key']:.2f} cal {v['calibrated']['0.01']['worst_key']:.2f} (fpr {v['calibrated']['0.01']['fpr']:.4f})" for m, v in rec['null'].items())
              + f' ({T():.0f}s)', flush=True)
        out['strip'] = res
        save_out(out)


# ------------------------------------------------------------------ jitter, width-agnostic
def part_jitter(out):
    cfg = cfg_rev3()
    res = out.get('jitter', {})
    T = Timer()
    for wm in ('gm_kgw', 'gm_kgw_sn'):
        if wm in res:
            continue
        sch = wm_scheme(cfg, wm, 9)
        rec = {}
        for key in (0, 1, 2):
            rid = f'{wm}__k{key}__s0'
            R = load_release(cfg, GEN, rid)
            trs = R['traces']
            base = TA.Base(sch)
            items = [base.enc(t['cells'], t['slot']) for t in trs]
            m0 = [TA.minute_of_day(t['t'][0], 'porto') for t in trs]
            sc = TA.Scorer(base, key)
            r = {}
            for att in ('none', 'jitter120', 'jitter180'):
                rng = np.random.default_rng(seed_from('timeattack', GEN, rid, att))     # same draws as timeattack.py
                bins = [TA.attack_minute(x, att, rng) // 30 for x in m0]
                e = {}
                for lab, size, n_ in (('m100', 100, 2000), ('m1000', 1000, len(items)), ('whole', len(items), len(items))):
                    lps = []
                    for i in range(0, n_ - size + 1, size):
                        best = 0.0
                        for J in (2, 4, 6):
                            cb = [TA.cand_bins(b, f'J{J}') for b in bins[i:i + size]]
                            F, M = TA.pool_pairs(items[i:i + size], cb)
                            best = min(best, sc.lp(F, M)[0])
                        lps.append(min(0.0, best + math.log10(3)))
                    a = np.array(lps)
                    e[lab] = dict(n_pools=len(a), median_mlp=float(-np.median(a)), frac_le_m3=float((a <= -3).mean()))
                r[att] = e
            rec[str(key)] = r
        agg = {}
        for att in ('none', 'jitter120', 'jitter180'):
            agg[att] = {lab: dict(median_mlp=float(np.median([rec[k][att][lab]['median_mlp'] for k in rec])),
                                  frac_le_m3=float(np.mean([rec[k][att][lab]['frac_le_m3'] for k in rec])))
                        for lab in ('m100', 'm1000', 'whole')}
        res[wm] = dict(per_key=rec, over_keys=agg)
        print(f"jitter {wm}: " + '; '.join(f"{a}: " + '/'.join(f"{v[l]['median_mlp']:.1f}({v[l]['frac_le_m3']:.2f})" for l in ('m100', 'm1000', 'whole')) for a, v in agg.items()) + f' ({T():.0f}s)', flush=True)
        out['jitter'] = res
        save_out(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='perkey,hop,dil,strip,jitter')
    a = ap.parse_args()
    parts = a.parts.split(',')
    out = load_out()
    if 'perkey' in parts:
        part_perkey(out)
    if 'hop' in parts or 'dil' in parts:
        part_hop_dil(out, parts)
    if 'jitter' in parts:
        part_jitter(out)
    if 'strip' in parts:
        part_strip(out)
    if 'hopkeys' in parts:
        part_hopkeys(out)


if __name__ == '__main__':
    main()
