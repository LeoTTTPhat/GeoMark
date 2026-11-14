"""Stage 7: radioactivity of POST-HOC-recoloured data (reviewers: generation-time embedding was justified by
radioactivity, but students were never trained on post-hoc-marked data, whose green bias is equally learnable).

Teacher corpus: 50,000 unwatermarked traces of the Porto small GPT (release none__k0__s20, same seed as the radio
teachers) recoloured post hoc with the owner's key 0 and GeoMark's coarse hashing (rate 1.0, as the post-hoc baseline).
Students: small GPTs as in run_radio.py, trained on 50,000 traces of which a fraction rho in {0.5, 1.0} is post-hoc
data (rest real), 3 student seeds. Tests: reading mode (5,000 real test contexts) and sampling mode (2,000 samples),
the teacher key's log10 p and its rank among 500 fresh null keys (1000..1499, as run_radio_keynull.py), and the joint
rank test over the three students. Output: results/rev1/radio_posthoc.jsonl (+ joint summary in radio_posthoc.json).
Loads one small model at a time (MPS). Usage: python src/rev1/radio_posthoc.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump, read_jsonl, append_jsonl, save_pickle, load_pickle
from geomark.common import seed_from, Timer, device, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark.detect import score_pairs, pair_sets, pooled
from geomark.sampler import VocabInfo, generate
from geomark.gpt import GPTAdapter
from geomark.posthoc import recolor_trace, true_key_scorer
from geomark.data import get_vocab, real_sequences, to_task, slot_distribution
from geomark.hexgrid import tokenize
from generate import wm_scheme
from train_gpt import train_model
from run_radio import reading_pairs
from run_radio_keynull import summarise
from rev1.radio_joint import joint


def main():
    cfg = cfg_rev1()
    rc = cfg['radio']
    teacher = rc['teacher']
    g = cfg['generators'][teacher]
    vocab = VocabInfo(get_vocab(cfg, g['dataset'], g['res']), g['res'])
    index = vocab.index
    sch_ph = wm_scheme(cfg, 'posthoc', g['res'])        # detection with GeoMark-KGW-2R hashing (same as the post-hoc baseline)
    tf = os.path.join(OUT, 'radio_posthoc_teacher.pkl')
    T = Timer()
    if os.path.exists(tf):
        wm_seqs = load_pickle(tf)
    else:
        R = load_release(cfg, teacher, 'none__k0__s20')['traces']
        rng = np.random.default_rng(seed_from('rev1-radio-posthoc'))
        scorer = true_key_scorer(sch_ph.with_(method='kgw'), PRF(rc['key']))
        wm_seqs = []
        for tr in R:
            x = recolor_trace(tr, sch_ph, scorer, 'green', 1.0, rng)
            cells = x['cells'] if x is not None else tr['cells']
            s = to_task(cells, index)
            if s and len(s) >= 2:
                wm_seqs.append((s, int(tr['slot'])))
        save_pickle(wm_seqs, tf)
        print(f'post-hoc teacher corpus: {len(wm_seqs)} sequences ({T():.0f}s)', flush=True)
    out = os.path.join(OUT, 'radio_posthoc.jsonl')
    done = {r['key'] for r in read_jsonl(out)}
    real_tr, real_sl = real_sequences(cfg, g['dataset'], g['res'], 'train', index, g['max_len'], limit=rc['n_train'], seed=11)
    val, val_sl = real_sequences(cfg, g['dataset'], g['res'], 'val', index, g['max_len'], limit=1000, seed=12)
    ctx, ctx_sl = real_sequences(cfg, g['dataset'], g['res'], 'test', index, g['max_len'], limit=rc['n_contexts'], seed=13)
    dev = device()
    ps = slot_distribution(cfg, g['dataset'])
    keys = list(range(1000, 1500))
    for rho in (0.5, 1.0):
        for seed in (0, 1, 2):
            name = f'posthoc_rho{rho}_s{seed}'
            if name in done:
                continue
            n = rc['n_train']; nw = int(round(rho * n))
            rng = np.random.default_rng(seed_from('radio', name))
            wi = rng.choice(len(wm_seqs), size=min(nw, len(wm_seqs)), replace=False)
            ri = rng.choice(len(real_tr), size=min(n - nw, len(real_tr)), replace=False) if n - nw > 0 else []
            tr = [wm_seqs[i][0] for i in wi] + [real_tr[i] for i in ri]
            sl = [wm_seqs[i][1] for i in wi] + [real_sl[i] for i in ri]
            mc = dict(rc['student'], max_len=g['max_len'])
            model = train_model(os.path.join(cfg['model_dir'], 'radio', name), vocab.V, tr, sl, val, val_sl, mc, seed=100 + seed)
            ad = GPTAdapter(model, dev)
            F, M = reading_pairs(ad, ctx, ctx_sl, vocab, sch_ph)
            rec = dict(key=name, rho=rho, seed=seed, n_train=len(tr), n_wm=len(wi), n_f=len(F), null_keys=[keys[0], keys[-1]])
            rec['read'] = summarise(score_pairs(F, M, sch_ph, PRF(rc['key']))['lp'], [score_pairs(F, M, sch_ph, PRF(k))['lp'] for k in keys])
            srng = np.random.default_rng(seed_from('radio-sample', name))
            slots = [int(x) for x in srng.choice(48, size=rc['n_sample'], p=ps)]
            bs = scaled(g.get('gen_batch', 256), 8)
            seqs = []
            for b in range(0, len(slots), bs):
                seqs += generate(GPTAdapter(model, dev), Scheme(method='none', res=g['res']), 0, vocab, slots[b:b + bs],
                                 gen_seed=seed_from(name, b), max_len=g['max_len'])
            pl = [pair_sets(vocab.cells[np.asarray(s, dtype=np.int64)], sl_, sch_ph) for s, sl_ in zip(seqs, slots) if s]
            rec['sample'] = summarise(pooled(pl, sch_ph, PRF(rc['key']))['lp'], [pooled(pl, sch_ph, PRF(k))['lp'] for k in keys])
            rec['sec'] = T()
            append_jsonl(out, rec)
            print(f"{name}: read lp {rec['read']['lp_true']:.2f} rank {rec['read']['rank_p']:.4f}; sample lp {rec['sample']['lp_true']:.2f} "
                  f"rank {rec['sample']['rank_p']:.4f} ({T():.0f}s)", flush=True)
            del model, ad; free_torch()
    recs = read_jsonl(out)
    summ = {}
    for rho in (0.5, 1.0):
        rs = sorted([r for r in recs if r['rho'] == rho], key=lambda r: r['seed'])
        if len(rs) >= 2:
            summ[f'rho{rho}'] = {m: joint(rs, m) for m in ('read', 'sample')}
    jdump(summ, 'radio_posthoc.json')


if __name__ == '__main__':
    main()
