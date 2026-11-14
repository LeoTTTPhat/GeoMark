"""Keyed pseudorandom function u = F_K(domain, seed, cell) in [0, 1).

Instantiation: a 128-bit per-(key, domain, seed) subkey from keyed BLAKE2b (cached; the seed space is small), then a
vectorised keyed SplitMix64-style mixing of the 64-bit H3 cell index under that subkey (three rounds, two subkey halves).
Domain separation ('fine', 'mid') makes fine- and medium-level uniforms independent in the random-oracle model.
The mapping depends only on the H3 integer of the cell, never on a model vocabulary, so detection is model-free.
Stage 7: key ids >= CRYPTO_KEY0 select a cryptographic instantiation instead: every (domain, seed, cell) value is a keyed
BLAKE2b-64 of the full input (no per-seed subkey, no invertible mixing), used for the PRF-replacement check.
"""
import hashlib
import numpy as np

_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
EOS_ID = 1   # pseudo cell id used for the end-of-trace token (never a valid H3 index)
CRYPTO_KEY0 = 10_000_000   # key ids >= this use the per-pair keyed-BLAKE2b PRF (Stage 7)


def _mix(x):
    x = (x ^ (x >> _S30)) * _C1
    x = (x ^ (x >> _S27)) * _C2
    return x ^ (x >> _S31)


def master_key(key_id):
    """Deterministic 32-byte master key for an integer key id (key ids >= 1000 are reserved for null keys)."""
    return hashlib.sha256(f'geomark-master-key-{int(key_id)}'.encode()).digest()


class PRF:
    def __new__(cls, key_id):
        if cls is PRF and int(key_id) >= CRYPTO_KEY0:
            return super().__new__(CryptoPRF)
        return super().__new__(cls)

    def __init__(self, key_id):
        self.key_id = key_id
        self.key = master_key(key_id)
        self._sub = {}

    def subkey(self, domain, seed):
        k = (domain, seed)
        v = self._sub.get(k)
        if v is None:
            d = hashlib.blake2b(f'{domain}|{seed!r}'.encode(), key=self.key, digest_size=16).digest()
            v = (np.uint64(int.from_bytes(d[:8], 'little')), np.uint64(int.from_bytes(d[8:], 'little')))
            if len(self._sub) > 2_000_000:
                self._sub.clear()
            self._sub[k] = v
        return v

    @staticmethod
    def _u(a, b, cells):
        with np.errstate(over='ignore'):
            x = _mix(cells.astype(np.uint64) ^ a)
            x = _mix(x + b * _GOLD)
            x = _mix(x ^ a ^ (b >> _S11))
        return ((x >> _S11).astype(np.float64) + 0.5) * (2.0 ** -53)  # in (0, 1), never exactly 0 or 1

    def vector(self, domain, seed, cells):
        """Uniforms for one seed and an array of cell ids."""
        a, b = self.subkey(domain, seed)
        return self._u(a, b, np.asarray(cells, dtype=np.int64))

    def pairs(self, domain, seeds, cells):
        """Uniforms for parallel lists of seeds and cells (one value per (seed, cell) pair)."""
        if len(cells) == 0:
            return np.zeros(0)
        ab = [self.subkey(domain, s) for s in seeds]
        a = np.fromiter((x[0] for x in ab), dtype=np.uint64, count=len(ab))
        b = np.fromiter((x[1] for x in ab), dtype=np.uint64, count=len(ab))
        return self._u(a, b, np.asarray(cells, dtype=np.int64))


def _blake_u(key, msg):
    d = hashlib.blake2b(msg, key=key, digest_size=8).digest()
    return ((int.from_bytes(d, 'little') >> 11) + 0.5) * (2.0 ** -53)


class CryptoPRF(PRF):
    """Per-pair keyed BLAKE2b (a standard keyed hash used as a PRF): u = BLAKE2b_K(domain | seed | cell)."""

    def vector(self, domain, seed, cells):
        pre = f'{domain}|{seed!r}|'.encode()
        k = self.key
        return np.fromiter((_blake_u(k, pre + str(int(c)).encode()) for c in np.asarray(cells, dtype=np.int64)),
                           dtype=np.float64, count=len(cells))

    def pairs(self, domain, seeds, cells):
        if len(cells) == 0:
            return np.zeros(0)
        k = self.key
        return np.fromiter((_blake_u(k, f'{domain}|{s!r}|{int(c)}'.encode()) for s, c in zip(seeds, cells)),
                           dtype=np.float64, count=len(cells))
