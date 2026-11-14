"""E9 radioactivity: students (small GPT) trained on a mix of teacher-watermarked synthetic traces (fraction rho) and
real traces; detection of the TEACHER's key in the student.
  reading mode : teacher-forced top-1 predictions of the student on held-out real contexts; score the de-duplicated
                 (seed of the real prefix, predicted cell) pairs (exact under H0: the student is independent of K).
  sampling mode: the student samples traces without watermark; pooled de-duplicated test over its outputs.
Controls: rho = 0 students, and a wrong key. Output: <out_dir>/radio/results.jsonl.
Usage: python src/run_radio.py --cfg CFG"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, load_pickle, append_jsonl, done_keys, seed_from, Timer, device, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme, seeds_for_cells
from geomark.detect import score_pairs, pair_sets, pooled
from geomark.sampler import VocabInfo, generate
from geomark.gpt import GPTAdapter
from geomark.hexgrid import parents
from geomark.data import release_dir, get_vocab, real_sequences, to_task, slot_distribution
from generate import wm_scheme
from train_gpt import train_model


def reading_pairs(adapter, seqs, slots, vocab, sch):
    preds = adapter.teacher_forced_top1(seqs, slots)
    F, M = {}, {}
    for s, sl, pr in zip(seqs, slots, preds):
        cells = vocab.cells[np.asarray(s)]
        seeds, _ = seeds_for_cells(cells, sl, sch)
        for t in range(len(s)):
            p = int(pr[t])
            if p >= vocab.V:
                continue
            c = int(vocab.cells[p])
            F[(seeds[t], c)] = None
            if sch.uses_mid:
                M[(seeds[t], int(parents([c], sch.mid_res)[0]))] = None
    return list(F), list(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    rc = cfg['radio']
    teacher = rc['teacher']
    g = cfg['generators'][teacher]
    vocab_cells = get_vocab(cfg, g['dataset'], g['res'])
    vocab = VocabInfo(vocab_cells, g['res'])
    index = vocab.index
    out = os.path.join(cfg['out_dir'], 'radio', 'results.jsonl')
    done = done_keys(out)
    real_tr, real_sl = real_sequences(cfg, g['dataset'], g['res'], 'train', index, g['max_len'], limit=rc['n_train'], seed=11)
    val, val_sl = real_sequences(cfg, g['dataset'], g['res'], 'val', index, g['max_len'], limit=1000, seed=12)
    ctx, ctx_sl = real_sequences(cfg, g['dataset'], g['res'], 'test', index, g['max_len'], limit=rc['n_contexts'], seed=13)
    dev = device()
    ps = slot_distribution(cfg, g['dataset'])
    for wm in rc['methods']:
        sch = wm_scheme(cfg, wm, g['res'])
        tf = os.path.join(release_dir(cfg, teacher), f"{wm}__k{rc['key']}__s{rc['teacher_seed']}.pkl")
        if not os.path.exists(tf):
            print('missing teacher release', tf); continue
        T_ = load_pickle(tf)['traces']
        wm_seqs = [(to_task(t['cells'], index), t['slot']) for t in T_]
        wm_seqs = [(s, sl) for s, sl in wm_seqs if s and len(s) >= 2]
        for rho in rc['fracs']:
            for seed in rc['seeds']:
                name = f'{wm}_rho{rho}_s{seed}' if rho > 0 else f'real_rho0_s{seed}'
                k = f'{name}|{wm}'
                if k in done:
                    continue
                T = Timer()
                n = rc['n_train']
                nw = int(round(rho * n))
                rng = np.random.default_rng(seed_from('radio', name))
                wi = rng.choice(len(wm_seqs), size=min(nw, len(wm_seqs)), replace=False) if nw else []
                ri = rng.choice(len(real_tr), size=min(n - nw, len(real_tr)), replace=False) if n - nw > 0 else []
                tr = [wm_seqs[i][0] for i in wi] + [real_tr[i] for i in ri]
                sl = [wm_seqs[i][1] for i in wi] + [real_sl[i] for i in ri]
                mc = dict(rc['student'], max_len=g['max_len'])
                model = train_model(os.path.join(cfg['model_dir'], 'radio', name), vocab.V, tr, sl, val, val_sl, mc, seed=100 + seed)
                ad = GPTAdapter(model, dev)
                rec = dict(key=k, student=name, wm=wm, rho=rho, seed=seed, n_train=len(tr), n_wm=len(wi))
                F, M = reading_pairs(ad, ctx, ctx_sl, vocab, sch)
                for kk in (rc['key'], rc['wrong_key']):
                    r = score_pairs(F, M, sch, PRF(kk))
                    rec[f'read_k{kk}'] = dict(lp=r['lp'], lp_f=r['lp_f'], lp_m=r['lp_m'], n_f=r['n_f'], n_m=r['n_m'], stat_f=r['stat_f'])
                # sampling mode
                srng = np.random.default_rng(seed_from('radio-sample', name))
                slots = [int(x) for x in srng.choice(48, size=rc['n_sample'], p=ps)]
                bs = scaled(g.get('gen_batch', 256), 8)
                seqs = []
                for b in range(0, len(slots), bs):
                    seqs += generate(GPTAdapter(model, dev), Scheme(method='none', res=g['res']), 0, vocab, slots[b:b + bs],
                                     gen_seed=seed_from(name, b), max_len=g['max_len'])
                pl = [pair_sets(vocab.cells[np.asarray(s, dtype=np.int64)], sl_, sch) for s, sl_ in zip(seqs, slots) if s]
                for kk in (rc['key'], rc['wrong_key']):
                    r = pooled(pl, sch, PRF(kk))
                    rec[f'sample_k{kk}'] = dict(lp=r['lp'], lp_f=r['lp_f'], n_f=r['n_f'], n_m=r['n_m'], stat_f=r['stat_f'])
                rec['sec'] = T()
                append_jsonl(out, rec)
                k1, k2 = 'read_k%d' % rc['key'], 'read_k%d' % rc['wrong_key']
                print(f"radio {k}: read lp {rec[k1]['lp']:.2f} (wrong key {rec[k2]['lp']:.2f}), "
                      f"sample lp {rec['sample_k%d' % rc['key']]['lp']:.2f} ({T():.0f}s)", flush=True)
                del model, ad; free_torch()
                done.add(k)


if __name__ == '__main__':
    main()
