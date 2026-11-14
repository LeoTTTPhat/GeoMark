"""Shared helpers for the Stage-8 (revision 2) analyses in src/rev2/ (re-exports src/rev1/rcommon.py). No model is loaded."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for p_ in (SRC, os.path.join(SRC, 'rev1')):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from rev1.rcommon import *  # noqa: F401,F403
from rev1.rcommon import ppath, dump_json, load_cfg  # noqa: F401

OUT2 = ppath('results', 'rev2')
os.makedirs(OUT2, exist_ok=True)


def cfg_rev2():
    return load_cfg(ppath('src', 'configs', 'rev2.yaml'))


def jdump2(obj, name):
    dump_json(obj, os.path.join(OUT2, name))
    print('wrote', os.path.join(OUT2, name), flush=True)


def jload2(name, default=None):
    f = os.path.join(OUT2, name)
    return json.load(open(f)) if os.path.exists(f) else ({} if default is None else default)
