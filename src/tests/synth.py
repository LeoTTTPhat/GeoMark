"""Synthetic traces for unit tests: autocorrelated random walks on the H3 r9 grid around central Porto, with
back-and-forth oscillations so that (seed, cell) pairs repeat (the case where de-duplication matters)."""
import numpy as np
import torch
import h3.api.basic_int as h3i
from geomark.hexgrid import neighbors, sample_in_cell
from geomark.common import LocalProj

CENTER = h3i.latlng_to_cell(41.15, -8.61, 9)


def walk(rng, n, p_back=0.4, res=9, start=None):
    c = start or CENTER
    for _ in range(int(rng.integers(0, 30))):
        c = rng.choice(neighbors(c))
    seq = [int(c)]
    while len(seq) < n:
        if len(seq) > 1 and rng.random() < p_back:
            seq.append(seq[-2])
        else:
            nb = [x for x in neighbors(seq[-1]) if x != seq[-1]]
            seq.append(int(rng.choice(nb)))
    return np.array(seq, dtype=np.int64)


def points_for(cells, rng, ds='porto', ppc=3):
    pr = LocalProj.for_dataset(ds)
    la, lo = [], []
    for c in cells:
        a, b = sample_in_cell(int(c), ppc, rng, pr)
        la.append(a); lo.append(b)
    lat, lon = np.concatenate(la), np.concatenate(lo)
    return dict(lat=lat, lon=lon, t=1372636800 + 15 * np.arange(len(lat)), ds=ds)


def local_vocab(radius=6, res=9):
    return np.array(sorted(h3i.grid_disk(CENTER, radius)), dtype=np.int64)


class ToyAdapter:
    """Uniform next-cell distribution over grid neighbours of the previous cell; EOS after `length` cells."""

    def __init__(self, vocab, length=40, rng=None):
        self.vocab = vocab
        self.index = {int(c): i for i, c in enumerate(vocab)}
        self.V = len(vocab)
        self.length = length
        self.rng = rng or np.random.default_rng(0)

    def _logits(self):
        lg = torch.full((len(self.last), self.V + 1), -float('inf'))
        for i, c in enumerate(self.last):
            if self.n[i] >= self.length:
                lg[i, self.V] = 0.0
                continue
            if c is None:
                lg[i, self.index[int(CENTER)]] = 0.0
                continue
            nb = [self.index[x] for x in neighbors(c) if x in self.index]
            lg[i, nb] = torch.as_tensor(self.rng.normal(0, 1.0, len(nb)), dtype=torch.float32)  # random preferences
        return lg

    def start(self, slots, prefixes=None):
        self.last = [None] * len(slots)
        self.n = [0] * len(slots)
        return self._logits()

    def step(self, tok):
        for i, t in enumerate(tok.tolist()):
            if t < self.V:
                self.last[i] = int(self.vocab[t]); self.n[i] += 1
        return self._logits()
