"""Stage 7: re-generation attack with an INDEPENDENT generator, and its null (reviewers: the original re-generation
attack used the owner's own generator, which the launderer does not have, and had no real-trace null).

The launderer keeps the first half of each trace's cells and regenerates the rest with its own unwatermarked model,
here the independently trained small GPT gpt_porto_r9_s1. Applied to the first 2,000 traces of the seed-0 releases of
GeoMark-KGW, GeoMark-Gumbel-H and the delta = 4 point (3 keys each), and, as the null, to 3,000 real test traces
(first half real, second half generated), tested with the 3 experiment keys. Reports per-trace TPR / FPR at 1e-3 and
pooled 100-trace results. Output: results/rev1/regen2.json. Loads one small model (CPU). Usage: python src/rev1/regen2.py"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import cfg_rev1, OUT, load_release, jdump, wilson
from geomark.common import seed_from, Timer, free_torch, scaled
from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark.sampler import VocabInfo, generate
from geomark.detect import pair_sets, trace_cells, detect_trace, pooled
from geomark.data import real_traces, load_ds
from generate import wm_scheme, get_adapter, make_decoder


def regen(adapter, vi, g, dec, trs, tag):
    rng = np.random.default_rng(seed_from('regen2', tag))
    out = []
    bs = scaled(256, 8)
    for a in range(0, len(trs), bs):
        chunk = trs[a:a + bs]
        pref = []
        for tr in chunk:
            idx = [vi.index.get(int(c)) for c in tr['cells'][:max(1, len(tr['cells']) // 2)]]
            pref.append([i for i in idx if i is not None] or [0])
        seqs = generate(adapter, Scheme(method='none', res=g['res']), 0, vi, [int(tr['slot']) for tr in chunk],
                        gen_seed=seed_from('regen2', tag, a), max_len=g['max_len'], prefixes=pref)
        for s, tr in zip(seqs, chunk):
            cells = vi.cells[np.asarray(s, dtype=np.int64)]
            lat, lon, t = dec.decode(cells, tr['t'][0], rng)
            out.append(dict(cells=cells, slot=tr['slot'], lat=lat, lon=lon, t=t, ds=tr['ds']))
    return out


def evaluate(trs, sch, prf):
    lps = np.array([detect_trace(t, sch, prf)['lp'] for t in trs])
    pl = [pair_sets(*trace_cells(t, sch), sch) for t in trs]
    pools = [pooled(pl[i:i + 100], sch, prf)['lp'] for i in range(0, len(pl) - 99, 100)]
    return dict(n=len(trs), k_m3=int((lps <= -3).sum()), rate_m3=float((lps <= -3).mean()), pooled_lps=[round(float(x), 3) for x in pools],
                pooled_median=float(np.median(pools)), pooled_frac_le_m3=float(np.mean(np.array(pools) <= -3)))


def main():
    cfg = cfg_rev1()
    fn = os.path.join(OUT, 'regen2.json')
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    os.environ['GEOMARK_DEVICE'] = 'cpu'
    torch.set_num_threads(scaled(8, 1))
    T = Timer()
    gen_l = 'gpt_porto_r9_s1'
    g = cfg['generators'][gen_l]
    adapter, vocab = get_adapter(cfg, gen_l)
    vi = VocabInfo(vocab, g['res'])
    dec, D = make_decoder(cfg, g, 'uniform')
    for wm in ('gm_kgw', 'gm_gumbel', 'a_g25_d4'):
        sch = wm_scheme(cfg, wm, 9)
        for key in (0, 1, 2):
            k = f'gpt_porto_r9_s0|{wm}__k{key}__s0'
            if k in out:
                continue
            R = load_release(cfg, 'gpt_porto_r9_s0', f'{wm}__k{key}__s0')['traces'][:2000]
            out[k] = evaluate(regen(adapter, vi, g, dec, R, k), sch, PRF(key))
            print(k, out[k]['rate_m3'], out[k]['pooled_median'], f'({T():.0f}s)', flush=True); jdump(out, 'regen2.json')
    if 'null' not in out:
        real = real_traces(cfg, 'porto', 'test', limit=3000, seed=1, min_pts=2)
        from geomark.hexgrid import tokenize
        rg = regen(adapter, vi, g, dec, [dict(tr, cells=tokenize(tr['lat'], tr['lon'], 9)) for tr in real], 'null')
        res = {}
        for wm in ('gm_kgw', 'gm_gumbel', 'a_g25_d4'):
            sch = wm_scheme(cfg, wm, 9)
            for key in (0, 1, 2):
                res[f'{wm}|k{key}'] = evaluate(rg, sch, PRF(key))
        tot = sum(v['k_m3'] for v in res.values()); n = sum(v['n'] for v in res.values())
        out['null'] = dict(per=res, fpr_m3=tot / n, n=n, wilson=wilson(tot, n),
                           pooled_frac_le_m3=float(np.mean([x <= -3 for v in res.values() for x in v['pooled_lps']])))
        print('null', out['null']['fpr_m3'], out['null']['pooled_frac_le_m3'], f'({T():.0f}s)', flush=True)
        jdump(out, 'regen2.json')
    del adapter; free_torch()


if __name__ == '__main__':
    main()
