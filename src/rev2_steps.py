"""Stage 8 (revision 2) step runner: runs the named steps sequentially inside ONE memguard job; each step is a python
subprocess (memory released between steps); done-markers results/rev2/.done/<step>; failed steps are retried by the
supervisor src/run_rev1.sh. Usage: python src/rev1_steps.py --steps a,b,c [--list]"""
import os, sys, argparse, subprocess, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
R2 = ['--cfg', 'src/configs/rev2.yaml']
MK = ['--cfg', 'src/configs/rev2_mm200.yaml']
GPT2 = 'gpt_porto_r9_s0,gpt_geolife_r9_s0'
QW = 'qwen15_geolife_r9,qwen15_porto_r9'
T = {
    'tests': ['-m', 'pytest', '-q', 'src/tests'],
    'gen-gpt': ['src/generate.py'] + R2 + ['--gen', GPT2, '--set', 'rev2_fact,rev2_sn,rev2_manykey,rev2_steal'],
    'mapmatch': ['src/mapmatch_run.py'] + R2 + ['--sets', 'rev2_fact,rev2_sn', '--gens', GPT2],
    'mapmatch-mk': ['src/mapmatch_run.py'] + MK + ['--sets', 'rev2_manykey', '--gens', 'gpt_porto_r9_s0'],
    'eval': ['src/evaluate.py'] + R2 + ['--sets', 'rev2_fact,rev2_sn', '--gens', GPT2, '--nulls'],
    'eval-mk': ['src/evaluate.py'] + MK + ['--sets', 'rev2_manykey', '--gens', 'gpt_porto_r9_s0'],
    'gen-qwen': ['src/generate.py'] + R2 + ['--gen', QW, '--set', 'rev2_llm_gl,rev2_llm_po'],
    'mapmatch-qwen': ['src/mapmatch_run.py'] + R2 + ['--sets', 'rev2_llm_gl,rev2_llm_po', '--gens', QW],
    'eval-qwen': ['src/evaluate.py'] + R2 + ['--sets', 'rev2_llm_gl,rev2_llm_po', '--gens', QW],
    'utility-rev2': ['src/rev1/utility.py', '--cfg', 'src/configs/rev2.yaml'],
}
# analysis steps (src/rev2/*.py), all idempotent
import glob as _glob
for _f in sorted(_glob.glob(os.path.join(ROOT, 'src', 'rev2', '*.py'))):
    _n = os.path.basename(_f)[:-3]
    if _n not in ('__init__', 'r2common'):
        T['a-' + _n] = [f'src/rev2/{_n}.py']


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
    dd = os.path.join(ROOT, 'results', 'rev2', '.done')
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
        with open(os.path.join(ROOT, 'results', 'rev2', 'step_times.tsv'), 'a') as f:
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
