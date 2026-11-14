"""E2/E3/E8/E9 generation of (watermarked) synthetic releases, and the re-generation attack.
Each release = generator x watermark config x key x sampling seed; written in parts (resume) then merged:
  <out_dir>/gen/<gen>/<wm>__k<key>__s<seed>.pkl   (dict(meta=..., traces=[{cells, slot, lat, lon, t, ds}, ...]))
Usage:
  python src/generate.py --cfg CFG --gen GEN --set main            # all releases of a release set for one generator
  python src/generate.py --cfg CFG --gen GEN --regen                # re-generation attack on that generator's releases
"""
import os, sys, argparse, glob, re
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, scaled, seed_from, save_pickle, load_pickle, free_torch, Timer, device, append_jsonl, dump_json
from geomark.scheme import Scheme
from geomark.sampler import VocabInfo, generate
from geomark.hexgrid import Decoder
from geomark.data import load_ds, get_vocab, slot_distribution, t0_for_slot, release_dir


def wm_scheme(cfg, wm_name, res):
    d = dict(cfg['wm'][wm_name]); d['res'] = res
    return Scheme.from_dict(d)


def get_adapter(cfg, gen):
    g = cfg['generators'][gen]
    vocab = get_vocab(cfg, g['dataset'], g['res'])
    if g['type'] == 'gpt':
        from train_gpt import load_gpt
        from geomark.gpt import GPTAdapter
        m, _ = load_gpt(cfg, gen)
        return GPTAdapter(m, device()), vocab
    from train_qwen_lora import QwenAdapter
    return QwenAdapter(cfg, gen), vocab


def release_jobs(cfg, gen, set_names):
    jobs = []
    for sn in set_names:
        rs = cfg['releases'][sn]
        if gen not in rs['gens']:
            continue
        for wm in rs['wm']:
            keys = [0] if cfg['wm'][wm]['method'] == 'none' else rs['keys']
            for key in keys:
                for seed in rs['seeds']:
                    jobs.append(dict(set=sn, wm=wm, key=key, seed=seed, n=rs['n'], decode=rs.get('decode', 'uniform')))
    return jobs


def make_decoder(cfg, g, mode):
    D = load_ds(cfg, g['dataset'])
    road = None
    if mode == 'road':
        import networkx as nx
        from geomark.common import OSM_GRAPH, p
        G = nx.read_graphml(p(OSM_GRAPH[g['dataset']]))
        road = np.array([(float(d['y']), float(d['x'])) for _, d in G.nodes(data=True)])
    return Decoder(g['dataset'], D['ppc'][g['res']], D['dt'], mode=mode, road_nodes=road), D


def release_id(job):
    rid = f"{job['wm']}__k{job['key']}__s{job['seed']}"
    if job['decode'] != 'uniform':
        rid += f"__{job['decode']}"
    return rid


def n_parts(cfg, job):
    return int(np.ceil(job['n'] / cfg['gen'].get('part_size', 1000)))


def part_file(cfg, gen, job, pi):
    return os.path.join(release_dir(cfg, gen), f'{release_id(job)}.part{pi}.pkl')


def gen_part(cfg, gen, job, pi, adapter, vocab_info, log=print):
    """Generate one part of a release (deterministic given gen, sampling seed and part index) and checkpoint it."""
    pf = part_file(cfg, gen, job, pi)
    if os.path.exists(pf):
        return pf
    g = cfg['generators'][gen]
    sch = wm_scheme(cfg, job['wm'], g['res'])
    dec, D = make_decoder(cfg, g, job['decode'])
    ps = slot_distribution(cfg, g['dataset'])
    part = cfg['gen'].get('part_size', 1000)
    bs = scaled(g.get('gen_batch', 256), 8)
    T = Timer()
    n = min(part, job['n'] - pi * part)
    rng = np.random.default_rng(seed_from(gen, job['seed'], pi, 'slots'))
    slots = rng.choice(48, size=n, p=ps)
    traces = []
    for a in range(0, n, bs):
        sl = [int(x) for x in slots[a:a + bs]]
        seqs = generate(adapter, sch, job['key'], vocab_info, sl, gen_seed=seed_from(gen, job['seed'], pi, a, 'sample'),
                        max_len=g['max_len'], top_k=cfg['gen'].get('top_k', 0))
        for s, slot in zip(seqs, sl):
            cells = vocab_info.cells[np.asarray(s, dtype=np.int64)] if s else np.zeros(0, dtype=np.int64)
            lat, lon, t = dec.decode(cells, t0_for_slot(slot, g['dataset'], rng, D['day0']), rng)
            traces.append(dict(cells=cells, slot=slot, lat=lat, lon=lon, t=t, ds=g['dataset']))
    save_pickle(dict(traces=traces, sec=T()), pf)
    log(f'  {gen}/{release_id(job)}: part {pi} ({n} traces) done ({T():.0f}s)')
    return pf


def finalize_release(cfg, gen, job, log=print):
    """Merge the parts of a release into <rid>.pkl (+ meta.json) once all parts exist. Returns True when merged."""
    g = cfg['generators'][gen]
    rid = release_id(job)
    out = os.path.join(release_dir(cfg, gen), rid + '.pkl')
    if os.path.exists(out):
        return True
    pfs = [part_file(cfg, gen, job, pi) for pi in range(n_parts(cfg, job))]
    if not all(os.path.exists(pf) for pf in pfs):
        return False
    traces, sec = [], 0.0
    for pf in pfs:
        x = load_pickle(pf)
        if isinstance(x, list):      # legacy part format
            traces += x
        else:
            traces += x['traces']; sec += x.get('sec', 0.0)
    sch = wm_scheme(cfg, job['wm'], g['res'])
    meta = dict(gen=gen, wm=job['wm'], scheme=sch.to_dict(), key=job['key'], seed=job['seed'], n=len(traces),
                decode=job['decode'], dataset=g['dataset'], res=g['res'], sec=sec)
    save_pickle(dict(meta=meta, traces=traces), out)
    dump_json(meta, out[:-4] + '.meta.json')
    for pf in pfs:
        os.remove(pf)
    append_jsonl(os.path.join(cfg['out_dir'], 'gen', 'timing.jsonl'), dict(gen=gen, rid=rid, n=len(traces), sec=sec))
    log(f'{gen}/{rid}: {len(traces)} traces, {sec:.0f}s compute')
    return True


def run_release(cfg, gen, job, adapter, vocab_info, log=print):
    """Sequential generation of one release (all parts), then merge."""
    out = os.path.join(release_dir(cfg, gen), release_id(job) + '.pkl')
    if os.path.exists(out):
        return out
    for pi in range(n_parts(cfg, job)):
        gen_part(cfg, gen, job, pi, adapter, vocab_info, log)
    finalize_release(cfg, gen, job, log)
    return out


# ---------------- parallel generation (worker processes, each with its own model copy) ----------------
_W = {}


def _winit(cfg_path, gen, dev, threads):
    import torch as _t
    if dev:
        os.environ['GEOMARK_DEVICE'] = dev
    _t.set_num_threads(max(1, threads))
    cfg = load_cfg(cfg_path)
    adapter, vocab = get_adapter(cfg, gen)
    _W.update(cfg=cfg, gen=gen, adapter=adapter, vi=VocabInfo(vocab, cfg['generators'][gen]['res']))


def _wpart(args):
    job, pi = args
    T = Timer()
    gen_part(_W['cfg'], _W['gen'], job, pi, _W['adapter'], _W['vi'], log=lambda *a: None)
    return f"  {_W['gen']}/{release_id(job)} part {pi}: {T():.0f}s"


def run_parallel(cfg, cfg_path, gen, jobs, workers, dev, threads, log=print):
    todo = [j for j in jobs if not os.path.exists(os.path.join(release_dir(cfg, gen), release_id(j) + '.pkl'))]
    tasks = [(j, pi) for j in todo for pi in range(n_parts(cfg, j)) if not os.path.exists(part_file(cfg, gen, j, pi))]
    log(f'{gen}: {len(todo)} releases to finish, {len(tasks)} parts to generate on {workers} x {dev or "default"} workers ({threads} threads)')
    if tasks:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=_winit, initargs=(cfg_path, gen, dev, threads)) as pool:
            for i, msg in enumerate(pool.imap_unordered(_wpart, tasks), 1):
                if i % 10 == 0 or i == len(tasks):
                    log(f'{msg} [{i}/{len(tasks)}]')
                    for j in todo:             # merge finished releases as we go
                        finalize_release(cfg, gen, j, log)
    for j in todo:
        finalize_release(cfg, gen, j, log)


def run_regen(cfg, gen, adapter, vocab_info):
    """Re-generation attack: keep the first half of each watermarked trace's cells, regenerate the rest with the
    unwatermarked generator `gen`. Output cached under <out_dir>/attacked/<src_gen>/<rid>__regen.pkl."""
    from geomark.data import release_files, load_release
    g = cfg['generators'][gen]
    rc = cfg['eval']
    srcs = [gen] + [s for s, tgt in cfg.get('regen_map', {}).items() if tgt == gen and s != gen]
    sch0 = Scheme(method='none', res=g['res'])
    for src in srcs:
        for f in release_files(cfg, src):
            rid = os.path.basename(f)[:-4]
            if not any(rid.startswith(w + '__') for w in rc.get('regen_wm', [])) or '__s0' not in rid:
                continue
            mk = re.match(r'.+?__k(\d+)__s', rid)
            if mk and int(mk.group(1)) >= 100:      # Stage 4: skip the 50-key 'keymix' releases (utility-only)
                continue
            out = os.path.join(cfg['out_dir'], 'attacked', src, f'{rid}__regen.pkl')
            if os.path.exists(out):
                continue
            R = load_release(f)
            n = min(rc['n_eval'], len(R['traces']))
            dec, D = make_decoder(cfg, g, 'uniform')
            rng = np.random.default_rng(seed_from(src, rid, 'regen'))
            bs = scaled(g.get('gen_batch', 256), 8)
            res_tr = []
            for a in range(0, n, bs):
                chunk = R['traces'][a:a + bs]
                pref = []
                for tr in chunk:
                    idx = [vocab_info.index.get(int(c)) for c in tr['cells'][:max(1, len(tr['cells']) // 2)]]
                    pref.append([i for i in idx if i is not None] or [0])
                seqs = generate(adapter, sch0, 0, vocab_info, [tr['slot'] for tr in chunk], gen_seed=seed_from(src, rid, a),
                                max_len=g['max_len'], prefixes=pref)
                for s, tr in zip(seqs, chunk):
                    cells = vocab_info.cells[np.asarray(s, dtype=np.int64)]
                    lat, lon, t = dec.decode(cells, tr['t'][0], rng)
                    res_tr.append(dict(cells=cells, slot=tr['slot'], lat=lat, lon=lon, t=t, ds=tr['ds']))
            save_pickle(dict(meta=dict(R['meta'], attack='regen', regen_gen=gen), traces=res_tr), out)
            print(f'regen {src}/{rid}: {len(res_tr)} traces', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--gen', required=True)
    ap.add_argument('--set', default='main', help='comma-separated release sets')
    ap.add_argument('--regen', action='store_true')
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    for gen in a.gen.split(','):          # several generators in one guarded step (fewer memguard slot waits)
        run_gen(cfg, a, gen)


def run_gen(cfg, a, gen):
    g = cfg['generators'][gen]
    dev = g.get('gen_device', cfg['gen'].get('device'))
    if not a.regen:
        jobs = release_jobs(cfg, gen, a.set.split(','))
        jobs = [j for j in jobs if cfg['wm'][j['wm']]['method'] not in ('posthoc', 'wtrace')]  # made by run_baselines.py
        print(f'{gen}: {len(jobs)} release jobs', flush=True)
        if not jobs:
            return
        # worker parallelism (small GPT on CPU workers; LLMs single process on MPS); shrinks with MEMGUARD_SCALE
        workers = scaled(g.get('gen_workers', cfg['gen'].get('workers', 1)), 1)
        if workers > 1:
            run_parallel(cfg, a.cfg, gen, jobs, workers, dev, g.get('gen_threads', cfg['gen'].get('threads', 4)))
            return
    if dev:
        os.environ['GEOMARK_DEVICE'] = dev
        if dev == 'cpu':
            torch.set_num_threads(scaled(8, 1))
    adapter, vocab = get_adapter(cfg, gen)
    vi = VocabInfo(vocab, g['res'])
    if a.regen:
        run_regen(cfg, gen, adapter, vi)
    else:
        for j in jobs:
            run_release(cfg, gen, j, adapter, vi)
    del adapter; free_torch()
    os.environ.pop('GEOMARK_DEVICE', None)


if __name__ == '__main__':
    main()
