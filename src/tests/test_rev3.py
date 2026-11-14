"""Stage 10 tests: the two-sided pooled test and the reference-calibrated test of src/rev3/r3common.py."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rev3'))
from r3common import score, ref_stats  # noqa: E402


def _null_scores(n_rep=3000, nf=400, nm=150, gamma=0.25, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        Gf = (rng.random(nf) < gamma).astype(np.float32)
        Gm = (rng.random(nm) < gamma).astype(np.float32)
        out.append(score((np.arange(nf), np.arange(nm)), Gf, Gm, gamma))
    return out


def test_two_sided_is_valid_under_h0():
    sc = _null_scores()
    for f in ('up', 'low', 'two'):
        lp = np.array([s[f] for s in sc])
        for a in (0.05, 0.01):
            assert (lp <= math.log10(a)).mean() <= a + 3 * math.sqrt(a * (1 - a) / len(lp)), (f, a)


def test_two_sided_is_bonferroni_of_tails():
    for s in _null_scores(n_rep=200, seed=1):
        assert abs(s['two'] - min(0.0, min(s['up'], s['low']) + math.log10(2))) < 1e-9


def test_lower_tail_detects_anti_green_data():
    nf, gamma = 2000, 0.25
    Gf = np.zeros(nf, np.float32); Gf[: int(0.15 * nf)] = 1          # 15% green instead of 25%
    s = score((np.arange(nf), np.zeros(0, np.int64)), Gf, np.zeros(0, np.float32), gamma)
    assert s['up'] > -0.1 and s['low'] < -10 and s['two'] < -10


def test_reference_calibration_recentres():
    rng = np.random.default_rng(2)
    gamma, n = 0.25, 1000
    # reference pools whose z-scores have mean ~ +2 (a 'hot' key): calibrated p of a similar pool must not be small
    pools = []
    G = (rng.random(50000) < gamma + 0.03).astype(np.float32)
    for _ in range(100):
        pools.append((rng.choice(50000, n, replace=False), np.zeros(0, np.int64)))
    rs = ref_stats(pools, G, np.zeros(0, np.float32), gamma, 100, 50000, n)
    s = score((rng.choice(50000, n, replace=False), np.zeros(0, np.int64)), G, np.zeros(0, np.float32), gamma, rs)
    assert s['up'] < -1 and s['cal_up'] > -1.5
