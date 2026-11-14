"""Unit tests for GeoMark's statistical core. Run: shared/.venv/bin/python -m pytest -q src/tests"""
import math
import numpy as np
import pytest
import torch
from scipy import stats

from geomark.prf import PRF
from geomark.scheme import Scheme, seeds_for_cells
from geomark.detect import pair_sets, score_pairs, detect_cells, pooled, detect_trace, uniforms
from geomark.sequential import BernoulliEProcess, ExpEProcess, SPRT, scores_from_uniforms
from geomark.sampler import VocabInfo, WMSampler, generate
from geomark.hexgrid import parents, tokenize, cells_of_points, collapse, sample_in_cell, neighbors
from geomark.common import LocalProj
from geomark import wtrace, attacks
from synth import walk, points_for, local_vocab, ToyAdapter, CENTER

KGW = Scheme(method='kgw', res=9, gamma=0.25, k=1, bin_min=30)
GUM = Scheme(method='gumbel', res=9, k=1, bin_min=30)
EXACT = Scheme(method='exact_kgw', res=9, gamma=0.25, two_res=False)


@pytest.fixture(scope='module')
def null_walks():
    rng = np.random.default_rng(1)
    return [(walk(rng, int(rng.integers(10, 60))), int(rng.integers(0, 48))) for _ in range(300)]


def null_pvalues(walks, sch, n_tests, randomized=True, dedup=True, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [pair_sets(c, s, sch, dedup=dedup) for c, s in walks]
    out = []
    for i in range(n_tests):
        f, m = pairs[i % len(pairs)]
        prf = PRF(10_000 + i)  # a fresh random key per test: H0 = trace independent of the key
        out.append(score_pairs(f, m, sch, prf, rng if randomized else None))
    return out


# ---------------- PRF ----------------
def test_prf_uniform_and_domain_separated():
    prf = PRF(3)
    cells = np.array(sorted(local_vocab(8)))
    u = np.concatenate([prf.vector('fine', ((i,), 0, 0), cells) for i in range(30)])
    v = np.concatenate([prf.vector('mid', ((i,), 0, 0), cells) for i in range(30)])
    assert stats.kstest(u, 'uniform').pvalue > 1e-3
    assert abs(np.corrcoef(u, v)[0, 1]) < 0.03
    assert np.array_equal(u[:len(cells)], PRF(3).vector('fine', ((0,), 0, 0), cells))     # deterministic
    assert not np.array_equal(u[:len(cells)], PRF(4).vector('fine', ((0,), 0, 0), cells))  # key-dependent
    assert np.allclose(prf.pairs('fine', [((0,), 0, 0)] * 5, cells[:5]), u[:5])           # pairwise == vector


# ---------------- Lemma 1: exact Type-I control ----------------
@pytest.mark.parametrize('sch', [KGW, GUM, EXACT, KGW.with_(k=2, ctx='mid'), GUM.with_(mask='M1'), KGW.with_(two_res=False)])
def test_null_pvalues_exactly_uniform(null_walks, sch):
    res = null_pvalues(null_walks, sch, 4000, randomized=True)
    pv = 10 ** np.array([r['lp'] for r in res])
    assert stats.kstest(pv, 'uniform').pvalue > 1e-3, sch
    for a in (0.01, 0.05, 0.1):
        assert (pv <= a).mean() <= a + 3 * math.sqrt(a * (1 - a) / len(pv))


def test_conservative_kgw_is_superuniform(null_walks):
    res = null_pvalues(null_walks, KGW, 4000, randomized=False)
    pv = 10 ** np.array([r['lp'] for r in res])
    for a in (0.01, 0.05, 0.1, 0.3):
        assert (pv <= a).mean() <= a + 3 * math.sqrt(a * (1 - a) / len(pv))


def test_without_dedup_the_null_is_violated(null_walks):
    """Oscillating walks repeat (seed, cell) pairs: scoring duplicates breaks the i.i.d. null (motivates dedup)."""
    res = null_pvalues(null_walks, GUM.with_(two_res=False), 4000, dedup=False)
    pv = 10 ** np.array([r['lp'] for r in res])
    assert (pv <= 0.01).mean() > 0.03


def test_dataset_level_pooling_is_exact(null_walks):
    rng = np.random.default_rng(5)
    pv = []
    for i in range(1500):
        idx = rng.choice(len(null_walks), size=20, replace=False)
        pl = [pair_sets(*null_walks[j], KGW) for j in idx]
        pv.append(10 ** pooled(pl, KGW, PRF(50_000 + i), rng)['lp'])
    assert stats.kstest(pv, 'uniform').pvalue > 1e-3


def test_bin_shift_search_bonferroni_valid(null_walks):
    lp = [detect_cells(c, s, KGW, PRF(70_000 + i), J=1)['lp_shift'] for i, (c, s) in enumerate(null_walks * 7)]
    pv = 10 ** np.array(lp)
    for a in (0.01, 0.05, 0.1):
        assert (pv <= a).mean() <= a + 3 * math.sqrt(a * (1 - a) / len(pv))


# ---------------- sequential tests ----------------
def test_eprocess_ville_bernoulli():
    rng = np.random.default_rng(0)
    alpha, g, R = 0.05, 0.25, 2000
    cross = 0
    for _ in range(R):
        ep = BernoulliEProcess(g)
        x = (rng.random(400) < g).astype(float)
        cross += any(ep.update(x[i:i + 10]) >= math.log(1 / alpha) for i in range(0, 400, 10))
    assert cross / R <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / R)
    # power under the alternative
    det = 0
    for _ in range(200):
        ep = BernoulliEProcess(g)
        det += ep.update((rng.random(300) < 0.4).astype(float)) >= math.log(1 / alpha)
    assert det / 200 > 0.9


def test_eprocess_ville_exponential():
    rng = np.random.default_rng(1)
    alpha, R = 0.05, 2000
    cross = 0
    for _ in range(R):
        ep = ExpEProcess()
        x = rng.exponential(1.0, 400)
        cross += any(ep.update(x[i:i + 10]) >= math.log(1 / alpha) for i in range(0, 400, 10))
    assert cross / R <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / R)
    ep = ExpEProcess()
    assert ep.update(rng.exponential(1.6, 300)) >= math.log(1 / alpha)


def test_sprt_false_positive_rate():
    rng = np.random.default_rng(2)
    alpha, beta, R = 0.05, 0.1, 3000
    rej = 0
    for _ in range(R):
        t = SPRT(0.25, 0.4, alpha, beta)
        t.update(rng.random(2000) < 0.25)
        rej += t.decision == 1
    bound = alpha / (1 - beta)
    assert rej / R <= bound + 3 * math.sqrt(bound * (1 - bound) / R)
    acc = sum(SPRT(0.25, 0.4, alpha, beta).update(rng.random(2000) < 0.4) == 1 for _ in range(300))
    assert acc / 300 >= 1 - beta - 0.05


def test_eprocess_on_real_null_scores(null_walks):
    """Streaming dataset-level detection: pooled de-duplicated scores of null traces under one random key."""
    alpha, R = 0.05, 400
    rng = np.random.default_rng(3)
    cross = 0
    for r in range(R):
        prf, seen, ep = PRF(90_000 + r), set(), BernoulliEProcess(0.25)
        hit = False
        for j in rng.choice(len(null_walks), size=40, replace=False):
            f, _ = pair_sets(*null_walks[j], KGW)
            new = [x for x in f if x not in seen]; seen.update(new)
            if new and ep.update(scores_from_uniforms(uniforms(prf, new, 'fine'), KGW)) >= math.log(1 / alpha):
                hit = True; break
        cross += hit
    assert cross / R <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / R)


# ---------------- Prop. 4: distortion-freeness of hierarchical Gumbel ----------------
def test_gumbel_h_distortion_free():
    vocab = VocabInfo(local_vocab(2), 9)   # 19 cells, several r8 parents
    rng = np.random.default_rng(0)
    logits = torch.as_tensor(rng.normal(0, 1.5, vocab.V + 1), dtype=torch.float32)
    p = torch.softmax(logits.double(), 0).numpy()
    N = 30000
    smp = WMSampler(GUM, 7, vocab, gen_seed=0)
    seeds = [((i,), 0, i % 48) for i in range(N)]
    out = []
    for a in range(0, N, 3000):
        out.append(smp.sample(logits.repeat(3000, 1), seeds[a:a + 3000], [False] * 3000).numpy())
    cnt = np.bincount(np.concatenate(out), minlength=vocab.V + 1)
    exp = p * N
    ok = exp >= 5
    chi = ((cnt[ok] - exp[ok]) ** 2 / exp[ok]).sum()
    assert stats.chi2.sf(chi, ok.sum() - 1) > 1e-3


def test_kgw_sampler_biases_green():
    vocab = VocabInfo(local_vocab(3), 9)
    logits = torch.zeros(4000, vocab.V + 1); logits[:, -1] = -1e9
    smp = WMSampler(KGW.with_(delta=2.0, delta_mid=0.0), 1, vocab, 0)
    seeds = [((i,), 0, 0) for i in range(4000)]
    out = smp.sample(logits, seeds, [False] * 4000).numpy()
    u = PRF(1).pairs('fine', seeds, vocab.cells[out])
    g = (u < 0.25).mean()
    expect = 0.25 * math.e ** 2 / (0.25 * math.e ** 2 + 0.75)
    assert abs(g - expect) < 0.03


# ---------------- generator <-> detector consistency (end to end on a toy model) ----------------
@pytest.mark.parametrize('sch', [KGW.with_(delta=3.0), GUM, GUM.with_(mask='M1', ctx='mid', k=2), EXACT.with_(delta=3.0)])
def test_generate_then_detect(sch):
    vocab = VocabInfo(local_vocab(8), 9)
    ad = ToyAdapter(vocab.cells, length=40, rng=np.random.default_rng(1))
    slots = list(range(0, 48, 2))
    seqs = generate(ad, sch, 11, vocab, slots, gen_seed=3, max_len=60)
    right = [pair_sets(vocab.cells[np.array(s)], sl, sch) for s, sl in zip(seqs, slots)]
    lp_right = pooled(right, sch, PRF(11))['lp']
    lp_wrong = pooled(right, sch, PRF(12))['lp']
    per_trace = [detect_cells(vocab.cells[np.array(s)], sl, sch, PRF(11))['lp'] for s, sl in zip(seqs, slots)]
    assert lp_right < -8                 # strong dataset-level evidence with the right key
    assert lp_wrong > -3                 # none with a wrong key
    assert np.median(per_trace) < -0.5   # per-trace evidence (low-entropy toy model: 6 candidate cells per step)


# ---------------- Lemma 2: coarse invariance ----------------
def _seed_by_pred_run(cells, slot, sch):
    """Map: coarse-run index of the predecessor -> seed (positions 1..n-1)."""
    seeds, _ = seeds_for_cells(cells, slot, sch)
    co = parents(cells, sch.coarse_res)
    run = np.concatenate([[0], np.cumsum(co[1:] != co[:-1])])
    return {int(run[t - 1]): seeds[t] for t in range(1, len(cells))}, collapse(co)[0]


@pytest.mark.parametrize('sch', [KGW, KGW.with_(k=3), GUM.with_(k=2)])
def test_coarse_invariance(sch):
    rng = np.random.default_rng(4)
    pr = LocalProj.for_dataset('porto')
    for _ in range(40):
        cells = walk(rng, 50, p_back=0.1)
        tr = points_for(cells, rng)
        fine0 = cells_of_points(tr['lat'], tr['lon'], 9)
        co0 = parents(fine0, sch.coarse_res)
        # perturbation that keeps every point inside the coarse parent of its fine cell (premise of Lemma 2)
        x, y = pr.to_xy(tr['lat'], tr['lon'])
        x2, y2 = x + rng.normal(0, 60, len(x)), y + rng.normal(0, 60, len(y))
        la, lo = pr.to_ll(x2, y2)
        co1 = parents(cells_of_points(la, lo, 9), sch.coarse_res)
        bad = co1 != co0
        la[bad], lo[bad] = tr['lat'][bad], tr['lon'][bad]
        c0, c1 = tokenize(tr['lat'], tr['lon'], 9), tokenize(la, lo, 9)
        m0, cs0 = _seed_by_pred_run(c0, 10, sch)
        m1, cs1 = _seed_by_pred_run(c1, 10, sch)
        assert np.array_equal(cs0, cs1)                      # collapsed coarse sequence preserved
        for j in set(m0) & set(m1):
            assert m0[j] == m1[j]                            # seeds indexed by predecessor run are identical
        assert set(m0.values()) - {m0.get(max(m0))} <= set(m1.values())


def test_lemma3_crossing_rate():
    """P(fine cell changes) under isotropic Gaussian noise ~ 0.921 sigma / a (closed form, r9 average edge)."""
    import h3.api.basic_int as h3i
    rng = np.random.default_rng(6)
    pr = LocalProj.for_dataset('porto')
    cells = walk(rng, 400, p_back=0.0)
    pts = [sample_in_cell(int(c), 10, rng, pr) for c in cells]
    lat, lon = np.concatenate([a for a, _ in pts]), np.concatenate([b for _, b in pts])
    x, y = pr.to_xy(lat, lon)
    sigma = 30.0
    la, lo = pr.to_ll(x + rng.normal(0, sigma, len(x)), y + rng.normal(0, sigma, len(y)))
    q = (cells_of_points(lat, lon, 9) != cells_of_points(la, lo, 9)).mean()
    a = h3i.average_hexagon_edge_length(9, 'm')
    assert abs(q - 0.921 * sigma / a) < 0.25 * 0.921 * sigma / a


# ---------------- end-to-end detection from points, attacks, W-Trace ----------------
def test_detect_trace_from_points_matches_cells():
    vocab = VocabInfo(local_vocab(8), 9)
    ad = ToyAdapter(vocab.cells, length=50, rng=np.random.default_rng(2))
    seqs = generate(ad, KGW.with_(delta=3.0), 21, vocab, [20] * 4, gen_seed=1, max_len=60)
    rng = np.random.default_rng(0)
    for s in seqs:
        cells = vocab.cells[np.array(s)]
        tr = points_for(cells, rng)
        tr['t'] = tr['t'] - tr['t'][0] + 1372636800 + 20 * 1800   # slot 20 (Porto local = UTC)
        a = detect_trace(tr, KGW, PRF(21))
        b = detect_cells(cells, 20, KGW, PRF(21))
        assert abs(a['lp'] - b['lp']) < 1e-9
        n = attacks.apply(tr, 'noise10+down2', np.random.default_rng(1))
        assert len(n['lat']) <= len(tr['lat'])


def test_wtrace_roundtrip():
    rng = np.random.default_rng(0)
    tr = points_for(walk(rng, 30), rng)
    m = wtrace.embed(tr, key=5)
    assert wtrace.ncc(m, tr, key=5) > 0.95
    assert abs(wtrace.ncc(m, tr, key=6)) < 0.6


# ---------------- batch compaction bookkeeping in generate() ----------------
class _DetAdapter:
    """Deterministic per-row logits (function of global row id and previous cell) with variable lengths; optional select()."""

    def __init__(self, vocab, with_select):
        self.vocab, self.V = vocab, len(vocab)
        self.index = {int(c): i for i, c in enumerate(vocab)}
        if with_select:
            self.select = self._select

    def _logits(self):
        lg = torch.full((len(self.ids), self.V + 1), -float('inf'))
        for j, (gid, c, n) in enumerate(zip(self.ids, self.last, self.n)):
            if n >= 5 + 7 * (gid % 5):          # row-specific length -> rows finish at different steps
                lg[j, self.V] = 0.0; continue
            c = c if c is not None else int(CENTER)
            nb = [self.index[x] for x in neighbors(c) if x in self.index]
            r = np.random.default_rng([gid, int(c) % 100003, n])
            lg[j, nb] = torch.as_tensor(r.normal(0, 1.0, len(nb)), dtype=torch.float32)
        return lg

    def start(self, slots, prefixes=None):
        self.ids = list(range(len(slots))); self.last = [None] * len(slots); self.n = [0] * len(slots)
        return self._logits()

    def _select(self, idx):
        idx = idx.tolist()
        self.ids = [self.ids[i] for i in idx]; self.last = [self.last[i] for i in idx]; self.n = [self.n[i] for i in idx]

    def step(self, tok):
        for j, t in enumerate(tok.tolist()):
            if t < self.V:
                self.last[j] = int(self.vocab[t]); self.n[j] += 1
        return self._logits()


def test_generate_batch_compaction_matches_full_batch():
    """Gumbel-H sampling is a deterministic function of logits and key, so dropping finished rows must not change outputs."""
    vocab = VocabInfo(local_vocab(8), 9)
    slots = [int(s) for s in np.arange(40) % 48]
    a = generate(_DetAdapter(vocab.cells, True), GUM, 5, vocab, slots, gen_seed=0, max_len=60)
    b = generate(_DetAdapter(vocab.cells, False), GUM, 5, vocab, slots, gen_seed=0, max_len=60)
    assert a == b
    assert sorted({len(s) for s in a}) == [5, 12, 19, 26, 33]


# ---------------- post-hoc recolouring (baseline / spoofing / scrubbing machinery) ----------------
def test_posthoc_recolor_keeps_coarse_seeds_and_marks():
    from geomark.posthoc import recolor, true_key_scorer
    rng = np.random.default_rng(9)
    sch = KGW.with_(method='posthoc')
    prf = PRF(31)
    g_before, g_after, g_wrong, lp_after = [], [], [], []
    for _ in range(30):
        cells = walk(rng, 80, p_back=0.0)
        new, ch = recolor(cells, 12, sch, true_key_scorer(sch.with_(method='kgw'), prf), 'green', 1.0, rng)
        # every changed cell keeps its coarse parent -> collapsed coarse sequence and all seeds unchanged (Lemma 2)
        assert np.array_equal(parents(new, sch.coarse_res), parents(cells, sch.coarse_res))
        assert seeds_for_cells(new, 12, sch)[0] == seeds_for_cells(cells, 12, sch)[0]
        assert all(new[t] != new[t + 1] for t in range(len(new) - 1))
        f0, _ = pair_sets(cells, 12, sch)
        f1, _ = pair_sets(new, 12, sch)
        g_before.append((uniforms(prf, f0, 'fine') < 0.25).mean())
        g_after.append((uniforms(prf, f1, 'fine') < 0.25).mean())
        g_wrong.append((uniforms(PRF(32), f1, 'fine') < 0.25).mean())
        lp_after.append(detect_cells(new, 12, sch, prf)['lp'])
    # few same-coarse alternatives adjacent to the previous cell exist (~1.3 per step), so the lift is moderate
    assert np.mean(g_after) > np.mean(g_before) + 0.08
    assert abs(np.mean(g_before) - 0.25) < 0.05 and abs(np.mean(g_wrong) - 0.25) < 0.05
    assert np.median(lp_after) < -1.0
