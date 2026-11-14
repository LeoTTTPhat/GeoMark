"""Stage 7 (revision 1) step runner: runs the named steps sequentially inside ONE memguard job; each step is a python
subprocess (memory released between steps); done-markers results/rev1/.done/<step>; failed steps are retried by the
supervisor src/run_rev1.sh. Usage: python src/rev1_steps.py --steps a,b,c [--list]"""
import os, sys, argparse, subprocess, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
R1 = ['--cfg', 'src/configs/rev1.yaml']
MK = ['--cfg', 'src/configs/rev1_mm200.yaml']
GPT2 = 'gpt_porto_r9_s0,gpt_geolife_r9_s0'
T = {
    'tests': ['-m', 'pytest', '-q', 'src/tests'],
    'gen-gpt': ['src/generate.py'] + R1 + ['--gen', GPT2, '--set', 'rev1_fact,rev1_eosn,rev1_manykey,rev1_gumkeymix,rev1_crypto'],
    'posthoc-prep': ['src/rev1_posthoc_prep.py'] + R1,
    'mapmatch': ['src/mapmatch_run.py'] + R1 + ['--sets', 'rev1_fact,rev1_eosn,rev1_posthoc', '--gens', GPT2],
    'mapmatch-mk': ['src/mapmatch_run.py'] + MK + ['--sets', 'rev1_manykey', '--gens', 'gpt_porto_r9_s0'],
    'eval': ['src/evaluate.py'] + R1 + ['--sets', 'rev1_fact,rev1_eosn,rev1_posthoc,rev1_crypto', '--gens', GPT2, '--nulls'],
    'eval-mk': ['src/evaluate.py'] + MK + ['--sets', 'rev1_manykey', '--gens', 'gpt_porto_r9_s0'],
    'gen-qwen': ['src/generate.py'] + R1 + ['--gen', 'qwen15_porto_r9', '--set', 'rev1_llm'],
    'mapmatch-qwen': ['src/mapmatch_run.py'] + R1 + ['--sets', 'rev1_llm', '--gens', 'qwen15_porto_r9'],
    'eval-qwen': ['src/evaluate.py'] + R1 + ['--sets', 'rev1_llm', '--gens', 'qwen15_porto_r9'],
    'gen-radio': ['src/generate.py'] + R1 + ['--gen', 'gpt_porto_r9_s0', '--set', 'rev1_radio_none'],
    'figs': ['src/make_figures.py'],
    'figs-rev1': ['src/rev1/make_figs_rev1.py'],
}
# analysis / extra-experiment steps (src/rev1/*.py), all idempotent; added as they are written
for name in ('utility', 'shiftpool', 'keycond', 'geomattack', 'linkage', 'wtpool', 'scrub', 'genspoof', 'regen2', 'radio_joint',
             'radio_posthoc', 'crypto_calib', 'lemma3', 'stealbound', 'stats', 'keymix2'):
    T['a-' + name] = [f'src/rev1/{name}.py'] + R1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, v in T.items():
            print(k, ' '.join(v))
        return
    os.chdir(ROOT)
    dd = os.path.join(ROOT, 'results', 'rev1', '.done')
    os.makedirs(dd, exist_ok=True)
    failed = []
    for st in [s for s in a.steps.split(',') if s]:
        mk = os.path.join(dd, st)
        if os.path.exists(mk) and not a.force:
            print(f'[{datetime.datetime.now():%F %T}] skip {st} (done)', flush=True)
            continue
        print(f'[{datetime.datetime.now():%F %T}] start {st}: {" ".join(T[st])}', flush=True)
        t0 = datetime.datetime.now()
        rc = subprocess.call([PY] + T[st], env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src')))
        dt = (datetime.datetime.now() - t0).total_seconds()
        with open(os.path.join(ROOT, 'results', 'rev1', 'step_times.tsv'), 'a') as f:
            f.write(f'{st}\t{rc}\t{dt:.0f}\t{t0:%F %T}\n')
        if rc == 0:
            open(mk, 'w').write(f'{datetime.datetime.now():%F %T}\n')
            print(f'[{datetime.datetime.now():%F %T}] done {st} ({dt:.0f}s)', flush=True)
        else:
            failed.append(st)
            print(f'[{datetime.datetime.now():%F %T}] FAILED {st} rc={rc} ({dt:.0f}s)', flush=True)
    if failed:
        print('failed steps:', ','.join(failed), flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
