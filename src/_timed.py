"""Run a python script/module and append its pure compute time (excluding memguard slot waiting) to
$STEP_TIMES as: date, step, seconds, exit code, MEMGUARD_ATTEMPT. Usage: _timed.py <step> <script.py|-m module> args..."""
import os, sys, time, runpy, datetime
step = sys.argv[1]
t0 = time.time()
rc = 0
try:
    if sys.argv[2] == '-m':
        sys.argv = [sys.argv[3]] + sys.argv[4:]
        runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
    else:
        sys.argv = sys.argv[2:]
        sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
        runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:
    import traceback; traceback.print_exc(); rc = 1
f = os.environ.get('STEP_TIMES')
if f:
    with open(f, 'a') as fh:
        fh.write(f"{datetime.datetime.now():%F %T}\t{step}\t{time.time() - t0:.1f}\t{rc}\t{os.environ.get('MEMGUARD_ATTEMPT', '0')}\n")
sys.exit(rc)
