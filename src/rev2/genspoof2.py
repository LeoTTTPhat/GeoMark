"""Stage 8: generative forger against every seeding configuration (reviewers: forging was run only for k = 1; the seed-space
argument needs forging success as a function of the seed space). Same design as src/rev1/genspoof.py: the forger runs the
independently trained small GPT gpt_porto_r9_s1 with an EOS-neutral KGW bias (delta = 2, medium delta/2 where the scheme
has a medium level) towards the green lists estimated (run_steal.Estimate) from N watermarked traces of the corpus
(N = 10^3, 10^4, 10^5), against 80,000 real training traces; 2,000 forged traces are tested with the owner's TRUE key 0:
per-trace forgery rate and share of 100-trace pools at p <= 1e-3. Null: the same generator without bias, tested under every
scheme. Corpora: k0 (bin only), exb (exact + bin), eosn (EOS-neutral GeoMark-KGW), sn (step-neutral), k2.
Output: results/rev2/genspoof2.json. Loads one small model (CPU). Usage: python src/rev2/genspoof2.py"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2common import cfg_rev2, jdump2, jload2, load_release
from geomark.common import Timer, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark import sampler as S
from geomark.detect import pair_sets, trace_cells
from geomark.data import real_traces
from run_steal import Estimate
from generate import wm_scheme, get_adapter
from rev1.genspoof import StolenSampler, run, evaluate

CORPORA = [('k0', 'a_k0'), ('exb', 'exact_kgw_bin'), ('eosn', 'gm_kgw_eosn'), ('sn', 'gm_kgw_sn'), ('k2', 'gm_kgw_k2')]


def main():
    cfg = cfg_rev2()
    out = jload2('genspoof2.json')
    os.environ['GEOMARK_DEVICE'] = 'cpu'
    torch.set_num_threads(scaled(8, 1))
    T = Timer()
    gen_f = 'gpt_porto_r9_s1'
    g = cfg['generators'][gen_f]
    adapter, vocab = get_adapter(cfg, gen_f)
    vi = S.VocabInfo(vocab, g['res'])
    prf = PRF(0)
    null_trs = None
    sc = cfg['steal']
    real_train = real_traces(cfg, 'porto', 'train', limit=sc['n_prior'], seed=7)
    for name, wm in CORPORA:
        sch_true = wm_scheme(cfg, wm, 9)
        if f'{name}|null' not in out:
            if null_trs is None:
                null_trs = run(cfg, adapter, vi, g, Scheme(method='none', res=9), 0, False, 'null')
            out[f'{name}|null'] = evaluate(null_trs, sch_true, prf)
            jdump2(out, 'genspoof2.json')
        todo = [N for N in (1000, 10000, 100000) if f'{name}|N{N}' not in out]
        if not todo:
            continue
        R = load_release(cfg, 'gpt_porto_r9_s0', f'{wm}__k0__s10')
        if R is None:
            print('missing corpus', name, flush=True); continue
        prior_pairs = [pair_sets(*trace_cells(t, sch_true), sch_true) for t in real_train]
        wm_pairs = [pair_sets(t['cells'], t['slot'], sch_true) for t in R['traces'][:100000]]
        for N in todo:
            StolenSampler.EST = Estimate(sch_true, wm_pairs[:N], prior_pairs)
            dmid = 1.0 if sch_true.uses_mid else 0.0
            sch = sch_true.with_(delta=2.0, delta_mid=dmid, eos_neutral=True)
            out[f'{name}|N{N}'] = evaluate(run(cfg, adapter, vi, g, sch, 0, True, f'{name}_N{N}'), sch_true, prf)
            r = out[f'{name}|N{N}']
            print(f"{name} N={N}: per-trace {r['tpr_m3']:.4f} pooled median {r['pooled_median']:.2f} detect {r['pooled_frac_le_m3']:.2f} "
                  f"({T():.0f}s)", flush=True)
            jdump2(out, 'genspoof2.json')
    del adapter; free_torch()


if __name__ == '__main__':
    main()
