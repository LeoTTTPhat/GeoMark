"""Stage-7 (revision) unit tests: exact-hash seeding with a time bin, seed-level masking (M2) for Gumbel-H, EOS-neutral
KGW sampling, and the per-pair cryptographic PRF. Run: shared/.venv/bin/python -m pytest -q src/tests"""
import math
import numpy as np
import pytest
import torch
from scipy import stats

from geomark.prf import PRF, CryptoPRF, CRYPTO_KEY0
from geomark.scheme import Scheme, seeds_for_cells
from geomark.detect import pair_sets, score_pairs, detect_cells, pooled
from geomark.sampler import VocabInfo, WMSampler, generate
from synth import walk, local_vocab, ToyAdapter, CENTER

KGW = Scheme(method='kgw', res=9, gamma=0.25, k=1, bin_min=30)
GUM = Scheme(method='gumbel', res=9, k=1, bin_min=30)
EXB = Scheme(method='exact_kgw_bin', res=9, gamma=0.25, two_res=False, bin_min=30)


@pytest.fixture(scope='module')
def null_walks():
    rng = np.random.default_rng(1)
    return [(walk(rng, int(rng.integers(10, 60))), int(rng.integers(0, 48))) for _ in range(300)]


@pytest.mark.parametrize('sch,key0', [(EXB, 20_000), (GUM.with_(mask='M2'), 20_000), (KGW.with_(eos_neutral=True), 20_000),
                                      (KGW, CRYPTO_KEY0 + 1), (GUM, CRYPTO_KEY0 + 1)])
def test_null_exact_new_variants(null_walks, sch, key0):
    rng = np.random.default_rng(0)
    pairs = [pair_sets(c, s, sch) for c, s in null_walks]
    res = [score_pairs(*pairs[i % len(pairs)], sch, PRF(key0 + i), rng)['lp'] for i in range(3000)]
    pv = 10 ** np.array(res)
    assert stats.kstest(pv, 'uniform').pvalue > 1e-3
    assert (pv <= 0.05).mean() <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / len(pv))


def test_crypto_prf_selected_and_consistent():
    p = PRF(CRYPTO_KEY0 + 5)
    assert isinstance(p, CryptoPRF)
    cells = local_vocab(3)
    u = p.vector('fine', ((1,), 0, 2), cells)
    assert np.allclose(p.pairs('fine', [((1,), 0, 2)] * len(cells), cells), u)
    assert not isinstance(PRF(5), CryptoPRF)


def test_exact_bin_seed_contains_bin():
    c = walk(np.random.default_rng(3), 20)
    s1, _ = seeds_for_cells(c, 4, EXB)
    s2, _ = seeds_for_cells(c, 5, EXB)
    assert s1[3][:2] == s2[3][:2] and s1[3] != s2[3]                   # same exact cell, different bin
    s0, _ = seeds_for_cells(c, 4, EXB.with_(method='exact_kgw'))
    assert s0[3] == s1[3][:2]


def test_m2_masks_first_step_and_repeated_seeds():
    c = walk(np.random.default_rng(5), 40)
    seeds, masked = seeds_for_cells(c, 10, GUM.with_(mask='M2'))
    assert masked[0]
    seen = set()
    for s, m in zip(seeds[1:], masked[1:]):
        assert m == (s in seen)
        seen.add(s)


class MarkovAdapter:
    """Fixed first-order Markov model over a small vocabulary (logits are a deterministic function of the prefix)."""

    def __init__(self, V, length, seed=0):
        r = np.random.default_rng(seed)
        self.V, self.length = V, length
        self.W = torch.as_tensor(r.normal(0, 1.2, (V + 1, V + 1)), dtype=torch.float32)
        self.W[:, V] = -1e9

    def _lg(self):
        lg = self.W[self.last].clone()
        for i, n in enumerate(self.n):
            if n >= self.length:
                lg[i] = -float('inf'); lg[i, self.V] = 0.0
        return lg

    def start(self, slots, prefixes=None):
        self.last = torch.full((len(slots),), self.V, dtype=torch.long)
        self.n = [0] * len(slots)
        return self._lg()

    def step(self, tok):
        for i, t in enumerate(tok.tolist()):
            if t < self.V:
                self.last[i] = t; self.n[i] += 1
        return self._lg()


def _trace_dist(sch, n_keys, vocab, ad_seed=0, key0=50_000):
    out = []
    for i in range(n_keys):
        ad = MarkovAdapter(vocab.V, 3, ad_seed)
        s = generate(ad, sch, key0 + i, vocab, [7], gen_seed=i, max_len=3)[0]
        out.append(tuple(s))
    return out


def test_m2_gumbel_trace_level_distortion_free():
    """Over a random key, a whole Gumbel-H(M2) trace has the generator's distribution (seeds repeat inside the
    coarse run, so M1 does not guarantee this)."""
    vocab = VocabInfo(local_vocab(1), 9)            # 7 cells, one coarse run: every step after the first shares a seed
    wm = _trace_dist(GUM.with_(mask='M2'), 3000, vocab)
    ref = _trace_dist(Scheme(method='none', res=9), 3000, vocab, key0=90_000)
    keys = sorted(set(wm) | set(ref))
    a = np.array([wm.count(k) for k in keys]); b = np.array([ref.count(k) for k in keys])
    ok = (a + b) >= 10
    tab = np.vstack([a[ok], b[ok]])
    p = stats.chi2_contingency(tab)[1]
    assert p > 1e-3


def test_eos_neutral_preserves_stop_probability():
    vocab = VocabInfo(local_vocab(3), 9)
    N = 20000
    logits = torch.zeros(N, vocab.V + 1)
    logits[:, vocab.V] = math.log(0.3 / 0.7 * vocab.V)        # p(EOS) = 0.3 without watermark
    seeds = [((i,), 0, 0) for i in range(N)]
    neu = WMSampler(KGW.with_(delta=4.0, eos_neutral=True), 1, vocab, 0).sample(logits, seeds, [False] * N).numpy()
    raw = WMSampler(KGW.with_(delta=4.0), 1, vocab, 0).sample(logits, seeds, [False] * N).numpy()
    f_neu, f_raw = (neu == vocab.V).mean(), (raw == vocab.V).mean()
    assert abs(f_neu - 0.3) < 0.015
    assert f_raw < 0.2                                        # plain KGW suppresses termination
    u = PRF(1).pairs('fine', [s for s, x in zip(seeds, neu) if x < vocab.V], vocab.cells[neu[neu < vocab.V]])
    assert (u < 0.25).mean() > 0.6                            # the cell choice is still watermarked


@pytest.mark.parametrize('sch', [EXB.with_(delta=3.0), GUM.with_(mask='M2'), KGW.with_(delta=3.0, eos_neutral=True)])
def test_generate_then_detect_new_variants(sch):
    vocab = VocabInfo(local_vocab(8), 9)
    ad = ToyAdapter(vocab.cells, length=40, rng=np.random.default_rng(1))
    slots = list(range(0, 48, 2))
    seqs = generate(ad, sch, 11, vocab, slots, gen_seed=3, max_len=60)
    right = [pair_sets(vocab.cells[np.array(s)], sl, sch) for s, sl in zip(seqs, slots)]
    assert pooled(right, sch, PRF(11))['lp'] < -6
    assert pooled(right, sch, PRF(12))['lp'] > -3
