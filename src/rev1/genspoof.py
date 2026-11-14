"""Stage 7: generative spoofer (reviewer: the recolouring forger is restricted to same-coarse neighbours; a forger can
instead run its OWN generator with a KGW bias towards the stolen green lists).

The forger's generator is the independently trained small GPT gpt_porto_r9_s1 (same data, different training seed;
never used to produce O's releases). Its sampler adds delta (fine) and delta/2 (medium) to the cells that the stealer's
estimate (run_steal.Estimate, from N watermarked traces of the k = 1 corpus vs 80,000 real traces) labels green under
the seed of the current context; stop probability is preserved (EOS-neutral). Forged traces are decoded like a release
and tested with the owner's TRUE key 0: per-trace forgery rate at p <= 1e-3 and share of 100-trace pools at p <= 1e-3.
Controls: the same generator without bias (null) and with the owner's true key (positive control).
Output: results/rev1/genspoof.json. Loads one small model (CPU). Usage: python src/rev1/genspoof.py"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump
from geomark.common import seed_from, Timer, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark import sampler as S
from geomark.detect import pair_sets, trace_cells, detect_trace, pooled
from geomark.data import real_traces, slot_distribution, t0_for_slot, load_ds
from run_steal import Estimate
from generate import wm_scheme, get_adapter, make_decoder

N_TRACES = 2000


class StolenSampler(S.WMSampler):
    EST = None

    def __init__(self, sch, key_id, vocab, gen_seed, cache_max=20000):
        super().__init__(sch, key_id, vocab, gen_seed, cache_max)

    def vectors(self, seed):
        v = self.cache.get(seed)
        if v is None:
            est = StolenSampler.EST
            u = np.full(len(self.vocab.cells_eos), 0.99)
            for i, c in enumerate(self.vocab.cells):
                if est.green_fine(seed, int(c)) == 1.0:
                    u[i] = 0.0
            m = np.full(len(self.vocab.mids_eos), 0.99)
            for i, P in enumerate(self.vocab.mids):
                if est.green_mid(seed, int(P)) == 1.0:
                    m[i] = 0.0
            if len(self.cache) >= self.cache_max:
                self.cache.clear()
            v = self.cache[seed] = (torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(m, dtype=torch.float64))
        return v


def run(cfg, adapter, vi, g, sch, key, stolen, tag):
    dec, D = make_decoder(cfg, g, 'uniform')
    ps = slot_distribution(cfg, g['dataset'])
    rng = np.random.default_rng(seed_from('genspoof', tag))
    slots = [int(x) for x in rng.choice(48, size=N_TRACES, p=ps)]
    bs = scaled(256, 8)
    orig = S.WMSampler
    if stolen:
        S.WMSampler = StolenSampler
    try:
        seqs = []
        for a in range(0, N_TRACES, bs):
            seqs += S.generate(adapter, sch, key, vi, slots[a:a + bs], gen_seed=seed_from('genspoof', tag, a), max_len=g['max_len'])
    finally:
        S.WMSampler = orig
    trs = []
    for s, sl in zip(seqs, slots):
        cells = vi.cells[np.asarray(s, dtype=np.int64)] if s else np.zeros(0, dtype=np.int64)
        lat, lon, t = dec.decode(cells, t0_for_slot(sl, g['dataset'], rng, D['day0']), rng)
        trs.append(dict(cells=cells, slot=sl, lat=lat, lon=lon, t=t, ds=g['dataset']))
    return trs


def evaluate(trs, sch_true, prf):
    lps = np.array([detect_trace(t, sch_true, prf)['lp'] if len(t['lat']) else 0.0 for t in trs])
    pl = [pair_sets(*trace_cells(t, sch_true), sch_true) if len(t['lat']) else ([], []) for t in trs]
    pools = [pooled(pl[i:i + 100], sch_true, prf)['lp'] for i in range(0, len(pl) - 99, 100)]
    return dict(n=len(trs), tpr_m3=float((lps <= -3).mean()), pooled_lps=[round(float(x), 3) for x in pools],
                pooled_median=float(np.median(pools)), pooled_frac_le_m3=float(np.mean(np.array(pools) <= -3)),
                mean_len_tok=float(np.mean([len(t['cells']) for t in trs])))


def main():
    cfg = cfg_rev1()
    fn = os.path.join(OUT, 'genspoof.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    os.environ['GEOMARK_DEVICE'] = 'cpu'
    torch.set_num_threads(scaled(8, 1))
    T = Timer()
    gen_f = 'gpt_porto_r9_s1'
    g = cfg['generators'][gen_f]
    adapter, vocab = get_adapter(cfg, gen_f)
    vi = S.VocabInfo(vocab, g['res'])
    sch_true = wm_scheme(cfg, 'gm_kgw', 9)
    prf = PRF(0)
    if 'null' not in out:
        out['null'] = evaluate(run(cfg, adapter, vi, g, Scheme(method='none', res=9), 0, False, 'null'), sch_true, prf)
        print('null', out['null']['tpr_m3'], out['null']['pooled_median'], f'({T():.0f}s)', flush=True); jdump(out, 'genspoof.json')
    for delta in (2.0, 4.0):
        k = f'truekey_d{delta:g}'
        if k not in out:
            sch = sch_true.with_(delta=delta, delta_mid=delta / 2, eos_neutral=True)
            out[k] = evaluate(run(cfg, adapter, vi, g, sch, 0, False, k), sch_true, prf)
            print(k, out[k]['tpr_m3'], out[k]['pooled_median'], f'({T():.0f}s)', flush=True); jdump(out, 'genspoof.json')
    sc = cfg['steal']
    real_train = real_traces(cfg, 'porto', 'train', limit=sc['n_prior'], seed=7)
    prior_pairs = [pair_sets(*trace_cells(t, sch_true), sch_true) for t in real_train]
    corpus = load_release(cfg, 'gpt_porto_r9_s0', 'gm_kgw__k0__s10')['traces']
    wm_pairs = [pair_sets(t['cells'], t['slot'], sch_true) for t in corpus[:100000]]
    for N in (1000, 10000, 100000):
        est = None
        for delta in (2.0, 4.0):
            k = f'stolen_N{N}_d{delta:g}'
            if k in out:
                continue
            if est is None:
                est = Estimate(sch_true, wm_pairs[:N], prior_pairs)
                StolenSampler.EST = est
            sch = sch_true.with_(delta=delta, delta_mid=delta / 2, eos_neutral=True)
            out[k] = evaluate(run(cfg, adapter, vi, g, sch, 0, True, k), sch_true, prf)
            print(k, out[k]['tpr_m3'], out[k]['pooled_median'], out[k]['pooled_frac_le_m3'], f'({T():.0f}s)', flush=True)
            jdump(out, 'genspoof.json')
    del adapter; free_torch()


if __name__ == '__main__':
    main()
