"""Re-implementation of W-Trace (Dadwal et al., SIGSPATIAL 2022, arXiv 2211.08116), following Sec. 3 of the paper:
points as complex numbers c_j = lat_j + i lon_j; split into sub-trajectories of size 16; FFT; amplitudes
alpha_e = alpha + s * w with w in {-1, 0, 1}^16 (10 non-zeros, key-derived); IFFT with the original phases.
Verification is non-blind: size alignment to the original by nearest points (haversine ~ local metric),
w' = (alpha' - alpha) / s per sub-trajectory, NCC(w, w') averaged over sub-trajectories, accept if > 0.85."""
import numpy as np
from .common import LocalProj

L, NNZ, S, TAU = 16, 10, 0.0003, 0.85


def wm_vector(key, j, L=L, nnz=NNZ):
    rng = np.random.default_rng([int(key), int(j), 7177])
    w = np.zeros(L)
    idx = rng.choice(L, size=nnz, replace=False)
    w[idx] = rng.choice([-1.0, 1.0], size=nnz)
    return w


def embed(tr, key, s=S):
    c = np.asarray(tr['lat'], dtype=float) + 1j * np.asarray(tr['lon'], dtype=float)
    out = c.copy()
    for j in range(len(c) // L):
        seg = c[j * L:(j + 1) * L]
        F = np.fft.fft(seg)
        a, ph = np.abs(F), np.angle(F)
        ae = np.maximum(a + s * wm_vector(key, j), 0.0)
        out[j * L:(j + 1) * L] = np.fft.ifft(ae * np.exp(1j * ph))
    return dict(tr, lat=out.real.copy(), lon=out.imag.copy())


def align(suspect, original):
    """Nearest suspect point for each original point (W-Trace size alignment)."""
    pr = LocalProj.for_dataset(original['ds'])
    ox, oy = pr.to_xy(original['lat'], original['lon'])
    sx, sy = pr.to_xy(suspect['lat'], suspect['lon'])
    if len(sx) == 0:
        return None
    from scipy.spatial import cKDTree
    _, idx = cKDTree(np.c_[sx, sy]).query(np.c_[ox, oy])
    return np.asarray(suspect['lat'])[idx] + 1j * np.asarray(suspect['lon'])[idx]


def ncc(suspect, original, key, s=S):
    c = align(suspect, original)
    if c is None:
        return 0.0
    o = np.asarray(original['lat']) + 1j * np.asarray(original['lon'])
    vals = []
    for j in range(len(o) // L):
        a = np.abs(np.fft.fft(o[j * L:(j + 1) * L]))
        a2 = np.abs(np.fft.fft(c[j * L:(j + 1) * L]))
        w2 = (a2 - a) / s
        w = wm_vector(key, j)
        den = np.sqrt((w ** 2).sum() * (w2 ** 2).sum())
        vals.append(float((w * w2).sum() / den) if den > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0
