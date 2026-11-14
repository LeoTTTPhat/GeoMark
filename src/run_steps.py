"""Run several pipeline steps sequentially inside ONE memguard-guarded job (one slot acquisition per group).

The step table below is the single definition of what every pipeline step runs. pipeline.sh wraps a group of steps as
  memguard --name p1-geomark-trajectory-watermark-<TAG>-<group> -- python src/run_steps.py --cfg CFG --steps a,b,c
Each step runs as its own python subprocess (src/_timed.py records its compute time in <out>/compute_times.tsv), so its
memory is released before the next one starts (one model at a time). Steps are idempotent (checkpointed outputs);
a done-marker <out>/.done/<step> is written after success unless an earlier step of the round failed
(env PIPELINE_FAILED=1 or a failure earlier in this group). Aliases: train, gen, regen expand to their per-generator steps.
Usage: python src/run_steps.py --cfg CFG --steps prep,tests [--force] [--list]"""
import os, sys, argparse, subprocess, datetime, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, ROOT

PY = sys.executable


def table(cfg, cfg_path):
    gens = cfg['generators']
    gpt = [g for g, v in gens.items() if v['type'] == 'gpt']
    llm = [g for g, v in gens.items() if v['type'] == 'qwen']
    rel = cfg['releases']
    gen_sets = ','.join(s for s in rel if s != 'baselines')
    # Stage 4b: the many-key 200-trace releases (keymix, keyvar_llm) are evaluated by run_keyvar.py / run_keymix.py only, never
    # by the generic eval/utility steps (they would otherwise be pooled into the 3-key main detection cells)
    all_sets = ','.join(s for s in rel if not s.startswith('steal') and s not in ('radio', 'keymix', 'keyvar_llm'))
    sets_for = lambda g: ','.join(s for s, r in rel.items() if g in r['gens'] and s != 'baselines')
    C = ['--cfg', cfg_path]
    T = {'prep': ['src/preprocess.py'] + C, 'tests': ['-m', 'pytest', '-q', 'src/tests']}
    if gpt:
        T['train-gpt'] = ['src/train_gpt.py'] + C + ['--gen', ','.join(gpt)]
    for g in llm:
        T[f'train-{g}'] = ['src/train_qwen_lora.py'] + C + ['--gen', g]
    if gpt:
        T['gen-gpt'] = ['src/generate.py'] + C + ['--gen', ','.join(gpt), '--set', gen_sets]
    for g in llm:
        if sets_for(g):
            T[f'gen-{g}'] = ['src/generate.py'] + C + ['--gen', g, '--set', sets_for(g)]
    T['baselines-prep'] = ['src/run_baselines.py'] + C + ['--prep']
    T['mapmatch'] = ['src/mapmatch_run.py'] + C + ['--real']
    if gpt:
        T['regen-gpt'] = ['src/generate.py'] + C + ['--gen', ','.join(gpt), '--regen']
    for g in llm:
        T[f'regen-{g}'] = ['src/generate.py'] + C + ['--gen', g, '--regen']
    T['eval'] = ['src/evaluate.py'] + C + ['--sets', all_sets, '--nulls']
    T['calib'] = ['src/run_calibration.py'] + C
    T['baselines-eval'] = ['src/run_baselines.py'] + C + ['--eval']
    T['utility'] = ['src/run_utility.py'] + C + ['--sets', all_sets, '--nextloc']
    T['steal'] = ['src/run_steal.py'] + C
    T['radio'] = ['src/run_radio.py'] + C
    T['radio-keynull'] = ['src/run_radio_keynull.py'] + C   # key-randomised null for E9 (added in Stage 3)
    T['capacity'] = ['src/run_capacity.py'] + C
    # ---- Stage 4 (completeness check) steps; all idempotent / resumable, see results/GAPS.md ----
    s4_gpt = ','.join(g for g in ('gpt_porto_r9_s0', 'gpt_geolife_r9_s0') if g in gens)
    T['s4-gen-gpt'] = ['src/generate.py'] + C + ['--gen', s4_gpt, '--set', 'strong,strong_geolife,decode_centroid,keymix']
    if 'qwen15_porto_r9' in gens:
        T['s4-gen-qwen'] = ['src/generate.py'] + C + ['--gen', 'qwen15_porto_r9', '--set', 'strong_llm']
        T['s4-regen-qwen'] = ['src/generate.py'] + C + ['--gen', 'qwen15_porto_r9', '--regen']
    T['s4-baselines-prep'] = ['src/run_baselines.py'] + C + ['--prep', '--eval']   # --eval: Qwen classifier (wtrace outputs exist)
    T['s4-mapmatch'] = ['src/mapmatch_run.py'] + C + ['--sets', 'strong,strong_geolife,strong_llm,resolution', '--gens',
                                                      'gpt_porto_r9_s0,gpt_geolife_r9_s0,qwen15_porto_r9,gpt_porto_r8_s0,gpt_geolife_r8_s0']
    T['s4-regen-gpt'] = ['src/generate.py'] + C + ['--gen', s4_gpt, '--regen']
    T['s4-eval'] = ['src/evaluate.py'] + C + ['--sets', 'strong,strong_geolife,strong_llm,ablation,ablation_geolife,resolution,decode_centroid', '--nulls']
    T['s4-shift'] = ['src/run_shift.py'] + C + ['--gens', 'gpt_porto_r9_s0,gpt_geolife_r9_s0,qwen15_porto_r9', '--wms', 'gm_kgw,gm_gumbel,a_g25_d4']
    T['s4-calib'] = ['src/run_calibration.py'] + C + ['--ext']
    T['s4-utility'] = ['src/run_utility.py'] + C + ['--sets', 'strong,strong_geolife,strong_llm,decode_centroid,baselines_big', '--nextloc']
    T['s4-keymix'] = ['src/run_keymix.py'] + C
    T['s4-steal'] = ['src/run_steal.py'] + C + ['--n_spoof', '5000']
    T['s4-radio'] = ['src/run_radio.py'] + C                     # third student seed (radio.seeds now [0, 1, 2])
    T['s4-radio-keynull'] = ['src/run_radio_keynull.py'] + C     # key-randomised null for the new students
    T['s4-radio-power'] = ['src/run_radio_keynull.py'] + C + ['--power']
    # ---- Stage 4b (2026-10-17, completeness critic, second attempt): key variability + r10 map-matching ----
    if 'gpt_geolife_r9_s0' in gens:
        T['s4b-gen-gpt'] = ['src/generate.py'] + C + ['--gen', 'gpt_geolife_r9_s0', '--set', 'keymix']
    if 'qwen15_porto_r9' in gens:
        T['s4b-gen-qwen'] = ['src/generate.py'] + C + ['--gen', 'qwen15_porto_r9', '--set', 'keyvar_llm']
    T['s4b-keymix'] = ['src/run_keymix.py'] + C                      # GeoLife key-averaged utility (Porto records are skipped)
    T['s4b-keyvar'] = ['src/run_keyvar.py'] + C
    if 'gpt_porto_r10_s0' in gens:
        T['s4b-mapmatch-r10'] = ['src/mapmatch_run.py'] + C + ['--sets', 'resolution', '--gens', 'gpt_porto_r10_s0']
        T['s4b-eval-r10'] = ['src/evaluate.py'] + C + ['--sets', 'resolution', '--gens', 'gpt_porto_r10_s0', '--nulls']
    T['aggregate'] = ['src/aggregate.py'] + C
    T['plots'] = ['src/plots.py'] + C
    alias = {'train': [k for k in T if k.startswith('train-')], 'gen': [k for k in T if k.startswith('gen-')],
             'regen': [k for k in T if k.startswith('regen-')]}
    return T, alias


def expand(names, T, alias):
    out = []
    for n in names:
        for x in alias.get(n, [n]):
            if x not in T:
                raise SystemExit(f'unknown step {x}; known: {list(T)}')
            if x not in out:
                out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--steps', default='all')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the expanded step names and exit')
    ap.add_argument('--retries', type=int, default=2)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    T, alias = table(cfg, a.cfg)
    names = list(T) if a.steps == 'all' else expand(a.steps.split(','), T, alias)
    if a.list:
        print(' '.join(names)); return
    out = cfg['out_dir']
    done_dir = os.path.join(out, '.done')
    os.makedirs(done_dir, exist_ok=True)
    env = dict(os.environ, STEP_TIMES=os.path.join(out, 'compute_times.tsv'), HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    failed = os.environ.get('PIPELINE_FAILED', '0') == '1'
    any_failed = False
    for n in names:
        marker = os.path.join(done_dir, n)
        if os.path.exists(marker) and not a.force:
            print(f'[{datetime.datetime.now():%F %T}] === {n} already done', flush=True)
            continue
        rc = 1
        for attempt in range(1, a.retries + 1):
            t0 = time.time()
            print(f'[{datetime.datetime.now():%F %T}] >>> {n} (attempt {attempt})', flush=True)
            rc = subprocess.call([PY, 'src/_timed.py', n] + T[n], cwd=ROOT, env=env)
            dt = time.time() - t0
            with open(os.path.join(out, 'step_times.tsv'), 'a') as f:
                f.write(f'{datetime.datetime.now():%F %T}\t{n}\t{dt:.0f}\t{rc}\n')
            print(f'[{datetime.datetime.now():%F %T}] <<< {n} rc={rc} {dt:.0f}s', flush=True)
            if rc == 0 or rc in (-2, 130):     # success, or interrupted by SIGINT: do not retry
                break
            time.sleep(20)
        if rc in (-2, 130):
            sys.exit(130)
        if rc == 0 and not failed:
            with open(marker, 'w') as f:
                f.write(f'{datetime.datetime.now():%F %T}\n')
        if rc != 0:
            failed = any_failed = True
    sys.exit(1 if any_failed else 0)


if __name__ == '__main__':
    main()
