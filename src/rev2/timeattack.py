"""Stage 8: start-time attacks (reviewers: most dataset-level evidence rests on the start-time bin, an attribute the
adversary controls; temporal generalization, large jitter and timestamp removal were not evaluated at release size).

The watermark seed uses only the 30-minute start bin of a trace, so a time attack is fully described by what it does to
the start minute m0 of every trace (geometry is untouched):
  none; retime60 (+-60 min, random sign); round60 / round180 / daypart (floor to the hour, to 3 h, to 6 h = day part,
  i.e. temporal generalization as applied to released mobility data); jitter120 / jitter180 (independent uniform shift
  in [-120, 120] / [-180, 180] min); strip (no time information).
Detector conditions (all exact over the key, Lemma 1, because the candidate set is a function of the data):
  plain (observed bin); union_J (union of the trace's pairs over bins b+j, |j| <= J); fwd_W (bins b .. b+W-1, the
  one-sided union matching floor rounding); all (all 48 bins; the only option when timestamps are stripped).
Pools: 100 traces (first 2,000 traces of each seed-0 release), 1,000 traces, and the whole release (all traces of the
seed-0 release: 10,000 for GeoMark-KGW, 5,000 for the EOS-/step-neutral releases); keys 0-2. Null: real test traces with
the same time attack, disjoint 100- and 1,000-trace pools, under 50 fixed keys (3000..3049) plus keys 0-2 (GeoMark-KGW detector; 1,000-trace pools only for the
union conditions): FPR at
1e-3 / 1e-2, worst-key FPR and the between-key variance ratio of the pooled z. Time-of-day utility: JSD between the
48-slot start-time histogram of the attacked release and that of the real test traces.
Output: results/rev2/timeattack.json. Model-free. Usage: python src/rev2/timeattack.py"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release, subkeys, log10_binom_sf, fisher_log10, LN10
from geomark.common import Timer, seed_from, TZ_OFFSET_H
from geomark.scheme import seeds_for_cells
from geomark.hexgrid import parents
from geomark.prf import PRF
from geomark.data import real_traces
from geomark.utility import jsd
from generate import wm_scheme

ATT_CONDS = {'none': ['plain', 'J1', 'J2', 'J4', 'all'], 'retime60': ['plain', 'J2', 'J4'],
             'round60': ['plain', 'fwd2', 'J2'], 'round180': ['plain', 'fwd6', 'J4', 'all'],
             'daypart': ['plain', 'fwd12', 'all'], 'jitter120': ['plain', 'J4', 'J6', 'all'],
             'jitter180': ['plain', 'J6', 'all'], 'strip': ['all']}
NULL_KEYS = list(range(3000, 3050))
RELS = [('gpt_porto_r9_s0', 'gm_kgw'), ('gpt_porto_r9_s0', 'gm_kgw_eosn'), ('gpt_porto_r9_s0', 'gm_kgw_sn'),
        ('gpt_geolife_r9_s0', 'gm_kgw'), ('gpt_geolife_r9_s0', 'gm_kgw_eosn'), ('gpt_geolife_r9_s0', 'gm_kgw_sn')]


def minute_of_day(t0, ds):
    return int(((int(t0) + 3600 * TZ_OFFSET_H[ds]) % 86400) // 60)


def attack_minute(m0, att, rng):
    if att in ('none', 'strip'):
        return m0
    if att == 'retime60':
        return (m0 + int(rng.choice([-1, 1])) * 60) % 1440
    if att.startswith('round'):
        w = int(att[5:]); return (m0 // w) * w
    if att == 'daypart':
        return (m0 // 360) * 360
    if att.startswith('jitter'):
        w = int(att[6:]); return int(m0 + rng.uniform(-w, w)) % 1440
    raise ValueError(att)


def cand_bins(b, cond):
    if cond == 'plain':
        return [b]
    if cond == 'all':
        return list(range(48))
    if cond.startswith('fwd'):
        return [(b + j) % 48 for j in range(int(cond[3:]))]
    J = int(cond[1:])
    return sorted({(b + j) % 48 for j in range(-J, J + 1)})


class Base:
    """Per trace: arrays (history id, cell, mid) of its unmasked positions; seeds are (h, m, bin)."""

    def __init__(self, sch):
        self.sch, self.H, self.hid, self.cells = sch, [], {}, {}

    def enc(self, cells, slot):
        if len(cells) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        seeds, masked = seeds_for_cells(cells, slot, self.sch)
        mids = parents(cells, self.sch.mid_res)
        keep = [i for i in range(len(cells)) if not masked[i]]
        if getattr(self.sch, 'adj_neutral', False):
            from geomark.hexgrid import is_neighbor
            keep = [i for i in keep if i == 0 or is_neighbor(cells[i - 1], cells[i])]
        h = np.array([self._h(seeds[i][:-1]) for i in keep], np.int64)
        c = np.array([self._c(int(cells[i])) for i in keep], np.int64)
        m = np.array([self._c(int(mids[i])) for i in keep], np.int64)
        return h, c, m

    def _h(self, x):
        v = self.hid.get(x)
        if v is None:
            v = self.hid[x] = len(self.H); self.H.append(x)
        return v

    def _c(self, x):
        v = self.cells.get(x)
        if v is None:
            v = self.cells[x] = len(self.cells)
        return v


def pool_pairs(items, conds_bins):
    """items: list of (h, c, m) arrays; conds_bins: list of candidate-bin lists (one per trace). Returns unique
    (seed index, cell index) codes for fine and mid, with seed index = h * 48 + bin."""
    F, M = [], []
    for (h, c, m), bins in zip(items, conds_bins):
        if len(h) == 0:
            continue
        for b in bins:
            s = h * 48 + b
            F.append(s * 10 ** 7 + c); M.append(s * 10 ** 7 + m)
    if not F:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.unique(np.concatenate(F)), np.unique(np.concatenate(M))


class Scorer:
    def __init__(self, base, key):
        cells_list = np.empty(len(base.cells), np.int64)
        for c, i in base.cells.items():
            cells_list[i] = c
        self.cells = cells_list
        seeds = [h + (b,) for h in base.H for b in range(48)]
        self.ab = {d: subkeys(key, d, seeds) for d in ('fine', 'mid')}
        self.gamma = base.sch.gamma

    def lp(self, F, M):
        out = []
        zs = []
        for d, codes in (('fine', F), ('mid', M)):
            if len(codes) == 0:
                out.append(0.0); zs.append(0.0); continue
            s = codes // 10 ** 7; c = self.cells[codes % 10 ** 7]
            a, b = self.ab[d]
            u = PRF._u(a[s], b[s], c)
            T, n = int((u < self.gamma).sum()), len(u)
            out.append(float(log10_binom_sf(T, n, self.gamma)))
            zs.append((T - n * self.gamma) / math.sqrt(n * self.gamma * (1 - self.gamma)))
        return float(fisher_log10(out[0], out[1], len(F), len(M))), zs[0]


def time_hist(minutes):
    return np.bincount(np.asarray(minutes) // 30, minlength=48).astype(float)


def main():
    cfg = cfg_rev2()
    out = jload2('timeattack.json')
    out.setdefault('releases', {}); out.setdefault('null', {}); out.setdefault('tod_utility', {})
    T = Timer()
    for gen, wm in RELS:
        g = cfg['generators'][gen]; ds = g['dataset']
        sch = wm_scheme(cfg, wm, 9)
        real = real_traces(cfg, ds, 'test')
        real_m = [minute_of_day(t['t'][0], ds) for t in real]
        href = time_hist(real_m)
        for key in (0, 1, 2):
            rid = f'{wm}__k{key}__s0'
            k = f'{gen}|{rid}'
            if k in out['releases']:
                continue
            R = load_release(cfg, gen, rid)
            if R is None:
                continue
            trs = R['traces']
            base = Base(sch)
            items = [base.enc(t['cells'], t['slot']) for t in trs]
            m0 = [minute_of_day(t['t'][0], ds) for t in trs]
            sc = Scorer(base, key)
            rec = {}
            for att, conds in ATT_CONDS.items():
                rng = np.random.default_rng(seed_from('timeattack', gen, rid, att))
                mm = [attack_minute(x, att, rng) for x in m0]
                if key == 0 and att != 'strip':
                    out['tod_utility'][f'{gen}|{wm}|{att}'] = float(jsd(time_hist(mm), href))
                bins = [x // 30 for x in mm]
                for cond in conds:
                    cb = [cand_bins(b, cond) for b in bins]
                    r = {}
                    for m, n_ in (('m100', 2000), ('m1000', len(items)), ('whole', len(items))):
                        size = 100 if m == 'm100' else (1000 if m == 'm1000' else n_)
                        lps = []
                        for i in range(0, n_ - size + 1, size):
                            F, M = pool_pairs(items[i:i + size], cb[i:i + size])
                            lps.append(sc.lp(F, M)[0])
                        r[m] = dict(n_pools=len(lps), size=size, median_mlp=float(-np.median(lps)) if lps else None,
                                    frac_le_m3=float(np.mean(np.array(lps) <= -3)) if lps else None)
                    rec[f'{att}|{cond}'] = r
            out['releases'][k] = rec
            print(f"{k}: " + ', '.join(f"{a}: {v['m100']['median_mlp']:.1f}/{v['m1000']['median_mlp']:.1f}/{v['whole']['median_mlp']:.1f}"
                                         for a, v in rec.items() if a.split('|')[1] in ('plain', 'all', 'fwd2', 'fwd6', 'fwd12', 'J4', 'J6'))
                  + f' ({T():.0f}s)', flush=True)
            jdump2(out, 'timeattack.json')
        # null: real test traces with the same time attack, many fixed keys (KGW detector of this scheme)
        nk = f'{ds}|{wm}'
        if nk in out['null'] or wm != 'gm_kgw':          # EOS-neutral uses the GeoMark-KGW detector; step-neutral null not run
            continue
        base = Base(sch)
        items = [base.enc(*_cells_slot(t, sch)) for t in real]
        res = {}
        scorers = [Scorer(base, kk) for kk in [0, 1, 2] + NULL_KEYS]
        for att, conds in ATT_CONDS.items():
            rng = np.random.default_rng(seed_from('timeattack-null', ds, att))
            bins = [attack_minute(x, att, rng) // 30 for x in real_m]
            for cond in conds:
                if cond == 'plain' and att != 'none':
                    continue
                cb = [cand_bins(b, cond) for b in bins]
                for size in ((100, 1000) if cond != 'plain' and not cond.startswith('J1') and not cond.startswith('J2') else (100,)):
                    n_use = 3000 if size == 100 else len(items)
                    pools = [pool_pairs(items[i:i + size], cb[i:i + size]) for i in range(0, n_use - size + 1, size)]
                    LP = np.zeros((len(scorers), len(pools))); Z = np.zeros_like(LP)
                    for a_, s_ in enumerate(scorers):
                        for j, (F, M) in enumerate(pools):
                            LP[a_, j], Z[a_, j] = s_.lp(F, M)
                    J_ = LP.shape[1]
                    k3 = (LP <= -3).sum(1); k2 = (LP <= -2).sum(1)
                    res[f'{att}|{cond}|m{size}'] = dict(n_keys=len(scorers), n_pools=J_, fpr_m3=float(k3.sum() / LP.size),
                                                       fpr_m2=float(k2.sum() / LP.size), worst_key_m3=float(k3.max() / J_),
                                                       worst_key_m2=float(k2.max() / J_),
                                                       z_var_ratio=float(J_ * Z.mean(1).var(ddof=1)) if J_ > 1 else None)
            print(f"null {nk} {att}: " + ', '.join(f"{c}: {res[f'{att}|{c}|m100']['fpr_m2']:.4f}" for c in conds if f'{att}|{c}|m100' in res)
                  + f' ({T():.0f}s)', flush=True)
        out['null'][nk] = res
        jdump2(out, 'timeattack.json')


def _cells_slot(t, sch):
    from geomark.hexgrid import tokenize
    from geomark.common import slot_of
    return tokenize(t['lat'], t['lon'], sch.res), slot_of(t['t'][0], t['ds'])


if __name__ == '__main__':
    main()
