"""Stage 10 (round-3 audit fix) step runner: runs the named steps sequentially inside ONE memguard job; each step is a
python subprocess; done-markers results/rev3/.done/<step>; timings appended to results/rev3/step_times.tsv.
Usage: python src/rev3_steps.py --steps a,b,c [--list]"""
import os, sys, argparse, subprocess, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
R3 = ['--cfg', 'src/configs/rev3.yaml']
MK = ['--cfg', 'src/configs/rev3_mm200.yaml']
T = {
    'tests': ['-m', 'pytest', '-q', 'src/tests'],
    'gen-gpt': ['src/generate.py'] + R3 + ['--gen', 'gpt_porto_r9_s0', '--set', 'rev3_scrub0,rev3_scrub,rev3_manykey'],
    'mapmatch-mk': ['src/mapmatch_run.py'] + MK + ['--sets', 'rev3_manykey', '--gens', 'gpt_porto_r9_s0'],
    'eval-mk': ['src/evaluate.py'] + MK + ['--sets', 'rev3_manykey', '--gens', 'gpt_porto_r9_s0'],
    'a-scrub3': ['src/rev3/scrub3.py'],
    'a-scrub3b': ['src/rev3/scrub3b.py'],
    'a-recal3': ['src/rev3/recal3.py'],
    'a-manykey3': ['src/rev3/manykey3.py'],
    'a-numbers3': ['src/rev3/numbers3.py'],
    'a-perkey3': ['src/rev3/recal3.py', '--parts', 'perkey'],
    'a-hopkeys3': ['src/rev3/recal3.py', '--parts', 'hopkeys'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, v in T.items():
            print(k, ' '.join(v))
        return
    os.chdir(ROOT)
    dd = os.path.join(ROOT, 'results', 'rev3', '.done')
    os.makedirs(dd, exist_ok=True)
    failed = []
    for st in [s for s in a.steps.split(',') if s]:
        mk = os.path.join(dd, st)
        if os.path.exists(mk):
            print(f'[{datetime.datetime.now():%F %T}] skip {st} (done)', flush=True)
            continue
        print(f'[{datetime.datetime.now():%F %T}] start {st}: {" ".join(T[st])}', flush=True)
        t0 = datetime.datetime.now()
        rc = subprocess.call([PY] + T[st], env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src')))
        dt = (datetime.datetime.now() - t0).total_seconds()
        with open(os.path.join(ROOT, 'results', 'rev3', 'step_times.tsv'), 'a') as f:
            f.write(f'{st}\t{rc}\t{dt:.0f}\t{t0:%F %T}\n')
        if rc == 0:
            open(mk, 'w').write(f'{datetime.datetime.now():%F %T}\n')
            print(f'[{datetime.datetime.now():%F %T}] done {st} ({dt:.0f}s)', flush=True)
        else:
            failed.append(st)
            print(f'[{datetime.datetime.now():%F %T}] FAILED {st} rc={rc} ({dt:.0f}s)', flush=True)
            break                                   # later steps may depend on this one
    if failed:
        print('failed steps:', ','.join(failed), flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
