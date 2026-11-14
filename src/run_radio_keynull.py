"""E9 follow-up: key-randomised null for the radioactivity tests (added in Stage 3 after the full run).

Why: in results/radio/results.jsonl the WRONG key (key 1) gave reading-mode log10 p of about -2.2 to -3.4 for every
KGW student, including the rho = 0 students trained on real data only. The reading-mode pair set (seed of the real
prefix, student's top-1 cell) does not depend on the key, and every student is read on the same 5,000 contexts, so all
students share nearly the same pairs and therefore nearly the same draw of the wrong key. Exactness of the test holds
over a RANDOM key, so one fixed wrong key is one draw. This script measures the key-randomised null directly: for every
saved student it scores the same reading-mode (and sampling-mode) pair sets under N fresh keys and reports
  * the null distribution of log10 p (KS test of 10^lp against U(0,1) -- conservative tests give KS stat >= 0 one-sided),
  * the empirical rank p-value of the teacher key: (1 + #{null lp <= teacher lp}) / (N + 1).
Students are loaded from models/radio/<name>/ (no retraining). Sampling mode regenerates the student's unwatermarked
samples with the same seeds as run_radio.py. Output: <out_dir>/radio/keynull.jsonl (one record per student x method;
resumable). Usage: python src/run_radio_keynull.py --cfg CFG [--n_keys 500]"""
import os, sys, argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, append_jsonl, done_keys, seed_from, Timer, device, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark.detect import score_pairs, pair_sets, pooled
from geomark.sampler import VocabInfo, generate
from geomark.gpt import GPTAdapter
from geomark.data import get_vocab, real_sequences, slot_distribution
from generate import wm_scheme
from train_gpt import train_model
from run_radio import reading_pairs

NULL_KEY0 = 1000          # null keys 1000 .. 1000+N-1 (disjoint from the experiment keys 0..3)


def summarise(true_lp, null_lps):
    null = np.asarray(null_lps, dtype=float)
    pv = np.clip(10.0 ** null, 0, 1)
    ks = stats.kstest(pv, 'uniform')
    return dict(lp_true=float(true_lp), n_null=len(null), rank_p=float((1 + (null <= true_lp).sum()) / (len(null) + 1)),
                null_median_lp=float(np.median(null)), null_frac_le_m2=float((null <= -2).mean()),
                null_frac_le_m3=float((null <= -3).mean()), null_min_lp=float(null.min()),
                null_ks_stat=float(ks.statistic), null_ks_p=float(ks.pvalue),
                null_lps=[round(float(x), 4) for x in null])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--n_keys', type=int, default=500)
    ap.add_argument('--power', action='store_true', help='Stage 4: cfg[radio_power] (more contexts / samples, subset of rho) '
                                                          '-> <out_dir>/radio/keynull_power.jsonl')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    rc = dict(cfg['radio'])
    if a.power:
        pw = cfg['radio_power']
        rc.update(fracs=pw['fracs'], n_contexts=pw['n_contexts'], n_sample=pw['n_sample'])
        a.n_keys = pw.get('n_keys', a.n_keys)
    g = cfg['generators'][rc['teacher']]
    vocab = VocabInfo(get_vocab(cfg, g['dataset'], g['res']), g['res'])
    index = vocab.index
    out = os.path.join(cfg['out_dir'], 'radio', 'keynull_power.jsonl' if a.power else 'keynull.jsonl')
    done = done_keys(out)
    ctx, ctx_sl = real_sequences(cfg, g['dataset'], g['res'], 'test', index, g['max_len'], limit=rc['n_contexts'], seed=13)
    dev = device()
    ps = slot_distribution(cfg, g['dataset'])
    keys = list(range(NULL_KEY0, NULL_KEY0 + a.n_keys))
    mc = dict(rc['student'], max_len=g['max_len'])
    for wm in rc['methods']:
        sch = wm_scheme(cfg, wm, g['res'])
        for rho in rc['fracs']:
            for seed in rc['seeds']:
                name = f'{wm}_rho{rho}_s{seed}' if rho > 0 else f'real_rho0_s{seed}'
                k = f'{name}|{wm}'
                if k in done:
                    continue
                mdir = os.path.join(cfg['model_dir'], 'radio', name)
                if not os.path.exists(os.path.join(mdir, 'done.json')):
                    print('missing student', mdir, flush=True)
                    continue
                T = Timer()
                model = train_model(mdir, vocab.V, [], [], [], [], mc, seed=100 + seed)   # loads the finished student
                ad = GPTAdapter(model, dev)
                F, M = reading_pairs(ad, ctx, ctx_sl, vocab, sch)
                rec = dict(key=k, student=name, wm=wm, rho=rho, seed=seed, n_f=len(F), n_m=len(M), null_keys=[keys[0], keys[-1]],
                           n_contexts=len(ctx), n_sample=rc['n_sample'])
                rec['read'] = summarise(score_pairs(F, M, sch, PRF(rc['key']))['lp'],
                                        [score_pairs(F, M, sch, PRF(kk))['lp'] for kk in keys])
                srng = np.random.default_rng(seed_from('radio-sample', name, *(['power'] if a.power else [])))
                slots = [int(x) for x in srng.choice(48, size=rc['n_sample'], p=ps)]
                bs = scaled(g.get('gen_batch', 256), 8)
                seqs = []
                for b in range(0, len(slots), bs):
                    seqs += generate(GPTAdapter(model, dev), Scheme(method='none', res=g['res']), 0, vocab, slots[b:b + bs],
                                     gen_seed=seed_from(name, b, *(['power'] if a.power else [])), max_len=g['max_len'])
                pl = [pair_sets(vocab.cells[np.asarray(s, dtype=np.int64)], sl_, sch) for s, sl_ in zip(seqs, slots) if s]
                rec['sample'] = summarise(pooled(pl, sch, PRF(rc['key']))['lp'], [pooled(pl, sch, PRF(kk))['lp'] for kk in keys])
                rec['sample']['n_f'] = int(pooled(pl, sch, PRF(rc['key']))['n_f'])
                rec['sec'] = T()
                append_jsonl(out, rec)
                print(f"keynull {k}: read lp {rec['read']['lp_true']:.2f} rank-p {rec['read']['rank_p']:.4f} "
                      f"(null KS p {rec['read']['null_ks_p']:.3f}, null<=-2 {rec['read']['null_frac_le_m2']:.3f}); "
                      f"sample lp {rec['sample']['lp_true']:.2f} rank-p {rec['sample']['rank_p']:.4f} ({T():.0f}s)", flush=True)
                del model, ad; free_torch()
                done.add(k)


if __name__ == '__main__':
    main()
