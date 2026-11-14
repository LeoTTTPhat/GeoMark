"""Anytime-valid sequential detection for streaming releases (traces arrive one by one; each new distinct
(seed, cell) pair contributes one score). Under H0 the de-duplicated scores are i.i.d. Bernoulli(gamma) (KGW) or
Exp(1) (Gumbel, x = -log(1-u)), so the mixture likelihood ratios below are nonnegative test martingales and Ville's
inequality gives P_H0(sup_n E_n >= 1/alpha) <= alpha. A Wald SPRT with simple alternative is provided for comparison.
"""
import math
import numpy as np


class BernoulliEProcess:
    """Mixture over a grid of alternatives q > gamma (uniform weights) of prod (q/g)^x ((1-q)/(1-g))^(1-x)."""

    def __init__(self, gamma, grid=None):
        self.g = gamma
        self.q = np.asarray(grid if grid is not None else np.linspace(gamma + (1 - gamma) / 20, 1 - (1 - gamma) / 20, 12))
        self.logw = np.full(len(self.q), -math.log(len(self.q)))
        self.logL = np.zeros(len(self.q))

    def update(self, x):
        x = np.asarray(x, dtype=float)
        k, n = x.sum(), len(x)
        self.logL += k * np.log(self.q / self.g) + (n - k) * np.log((1 - self.q) / (1 - self.g))
        return self.log_e()

    def log_e(self):
        a = self.logw + self.logL
        m = a.max()
        return float(m + math.log(np.exp(a - m).sum()))


class ExpEProcess:
    """Mixture over lambda in (0,1) of prod lambda * exp((1 - lambda) x); E_H0[lambda e^{(1-lambda)X}] = 1 for X~Exp(1)."""

    def __init__(self, grid=None):
        self.lam = np.asarray(grid if grid is not None else np.linspace(0.3, 0.95, 14))
        self.logw = np.full(len(self.lam), -math.log(len(self.lam)))
        self.logL = np.zeros(len(self.lam))

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.logL += len(x) * np.log(self.lam) + (1 - self.lam) * x.sum()
        return self.log_e()

    def log_e(self):
        a = self.logw + self.logL
        m = a.max()
        return float(m + math.log(np.exp(a - m).sum()))


def eprocess_for(sch):
    return ExpEProcess() if sch.score == 'gumbel' else BernoulliEProcess(sch.gamma)


def scores_from_uniforms(u, sch):
    u = np.asarray(u)
    return -np.log1p(-u) if sch.score == 'gumbel' else (u < sch.gamma).astype(float)


def first_crossing(stream_batches, ep, alpha):
    """Feed batches of scores; return index (1-based) of the first batch at which E >= 1/alpha, else None."""
    thr = math.log(1 / alpha)
    for i, b in enumerate(stream_batches, 1):
        if len(b) and ep.update(b) >= thr:
            return i
    return None


class SPRT:
    """Wald SPRT for Bernoulli(g0) vs Bernoulli(g1). Decision: +1 reject H0, -1 accept H0, 0 continue.
    Wald's bound: P_H0(reject) <= alpha / (1 - beta) (approximately alpha)."""

    def __init__(self, g0, g1, alpha=1e-3, beta=0.1):
        self.a = math.log(beta / (1 - alpha))
        self.b = math.log((1 - beta) / alpha)
        self.w1, self.w0 = math.log(g1 / g0), math.log((1 - g1) / (1 - g0))
        self.llr, self.n, self.decision = 0.0, 0, 0

    def update(self, x):
        for xi in np.asarray(x, dtype=float):
            if self.decision:
                break
            self.n += 1
            self.llr += self.w1 if xi else self.w0
            if self.llr >= self.b:
                self.decision = 1
            elif self.llr <= self.a:
                self.decision = -1
        return self.decision
