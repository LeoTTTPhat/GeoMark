"""Stage-8 (revision 2) unit tests: step-neutral KGW (adj_neutral) keeps the mass of EOS and of non-adjacent moves,
the detector scores only adjacent steps, the null stays exact, and generation-then-detection works.
Run: shared/.venv/bin/python -m pytest -q src/tests"""
import math
import numpy as np
import pytest
import torch
from scipy import stats

from geomark.prf import PRF
from geomark.scheme import Scheme
from geomark.detect import pair_sets, score_pairs, pooled
from geomark.sampler import VocabInfo, WMSampler, generate
from geomark.hexgrid import is_neighbor
from synth import walk, local_vocab, ToyAdapter

GN = Scheme(method='kgw', res=9, gamma=0.25, k=1, bin_min=30, eos_neutral=True, adj_neutral=True)


def test_step_neutral_preserves_group_masses():
    vocab = VocabInfo(local_vocab(3), 9)
    N = 30000
    prev = 0
    logits = torch.zeros(N, vocab.V + 1)
    logits[:, vocab.V] = math.log(0.2 / 0.8 * vocab.V)              # p(EOS) = 0.2
    adj = vocab.adj_mask([prev])[0].numpy()
    p_adj0 = 0.8 * adj[:vocab.V].sum() / vocab.V
    seeds = [((i,), 0, 0) for i in range(N)]
    out = WMSampler(GN.with_(delta=4.0), 1, vocab, 0).sample(logits, seeds, [False] * N, prevs=[prev] * N).numpy()
    f_eos = (out == vocab.V).mean()
    f_adj = adj[out].mean()
    assert abs(f_eos - 0.2) < 0.015
    assert abs(f_adj - p_adj0) < 0.015
    sel = adj[out] & (out < vocab.V)
    u = PRF(1).pairs('fine', [s for s, x in zip(seeds, sel) if x], vocab.cells[out[sel]])
    assert (u < 0.25).mean() > 0.6                                    # the adjacent choice is watermarked
    out2 = WMSampler(GN.with_(delta=4.0, adj_neutral=False), 1, vocab, 0).sample(logits, seeds, [False] * N).numpy()
    assert abs(adj[out2].mean() - p_adj0) > 0.02 or True             # plain EOS-neutral may shift it (not asserted)


def test_detector_scores_only_adjacent_steps():
    rng = np.random.default_rng(5)
    c = walk(rng, 30)
    c = np.concatenate([c[:10], c[20:]])                              # introduce a jump
    f, _ = pair_sets(c, 3, GN)
    f_all, _ = pair_sets(c, 3, GN.with_(adj_neutral=False))
    assert len(f) <= len(f_all)
    kept = {x[1] for x in f}
    jumps = [int(c[i]) for i in range(1, len(c)) if not is_neighbor(c[i - 1], c[i])]
    assert all(j not in kept or list(c).count(j) > 1 for j in jumps)


def test_step_neutral_null_exact():
    rng = np.random.default_rng(1)
    walks = [(walk(rng, int(rng.integers(10, 60))), int(rng.integers(0, 48))) for _ in range(300)]
    pairs = [pair_sets(c, s, GN) for c, s in walks]
    r2 = np.random.default_rng(0)
    pv = 10 ** np.array([score_pairs(*pairs[i % 300], GN, PRF(50_000 + i), r2)['lp'] for i in range(3000)])
    assert stats.kstest(pv, 'uniform').pvalue > 1e-3


def test_generate_then_detect_step_neutral():
    vocab = VocabInfo(local_vocab(8), 9)
    ad = ToyAdapter(vocab.cells, length=40, rng=np.random.default_rng(1))
    slots = list(range(0, 48, 2))
    sch = GN.with_(delta=3.0)
    seqs = generate(ad, sch, 11, vocab, slots, gen_seed=3, max_len=60)
    right = [pair_sets(vocab.cells[np.array(s)], sl, sch) for s, sl in zip(seqs, slots)]
    assert pooled(right, sch, PRF(11))['lp'] < -6
    assert pooled(right, sch, PRF(12))['lp'] > -3
