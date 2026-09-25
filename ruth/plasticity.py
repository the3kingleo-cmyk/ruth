"""Online plasticity: the continuous decoder learns every moment it is alive.

Recursive least squares (the learning rule behind FORCE training, Sussillo &
Abbott 2009) gives the exact running least-squares readout with forgetting
factor lambda, O(d^2) per step, no gradients, no epochs, no GPU:
    k = P phi / (lambda + phi' P phi)
    W <- W + e k'            (e = observed - predicted)
    P <- (P - k phi' P) / lambda
P is also the posterior covariance of the readout, so phi' P phi measures how
novel a situation is -- used as epistemic value by active inference.
"""
from __future__ import annotations

import numpy as np


class RLSDecoder:
    def __init__(self, d_in: int, d_out: int, lam: float, delta: float):
        self.w = np.zeros((d_out, d_in))
        self.p = np.eye(d_in) / delta
        self.lam = lam

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return self.w @ phi

    def novelty(self, phi: np.ndarray) -> float:
        return float(phi @ self.p @ phi)

    def update(self, phi: np.ndarray, err: np.ndarray, lam: float | None = None) -> None:
        lam = self.lam if lam is None else lam
        pphi = self.p @ phi
        k = pphi / (lam + phi @ pphi)
        self.w += np.outer(err, k)
        buf = getattr(self, "_buf", None)
        if buf is None or buf.shape != self.p.shape:
            buf = self._buf = np.empty_like(self.p)
        np.outer(k, pphi, out=buf)       # reuse one buffer: no allocation per moment
        self.p -= buf
        if lam != 1.0:
            self.p *= 1.0 / lam

    def arrays(self) -> dict:
        return {"rls.w": self.w, "rls.p": self.p}

    def load(self, arr) -> None:
        self.w = np.array(arr["rls.w"])
        self.p = np.array(arr["rls.p"])


class Precision:
    """Running per-channel precision and surprise statistics (predictive coding)."""

    def __init__(self, dim: int, rate: float = 0.01):
        self.var = np.ones(dim)
        self.rate = rate
        self.s_mean = 0.0
        self.s_var = 1.0

    def observe(self, err: np.ndarray, k: float, mask: np.ndarray | None = None):
        """Precision-weighted prediction error (variational free energy of a
        Gaussian generative model, up to constants) over the *observed*
        channels, and whether it is an outlier (> mean + k std of recent
        surprise). Then adapt statistics. Unobserved channels carry no
        evidence, so they neither surprise nor teach."""
        m = np.ones_like(err) if mask is None else mask
        s = 0.5 * float(np.sum(m * err * err / self.var) / max(m.sum(), 1.0))
        outlier = s > self.s_mean + k * np.sqrt(max(self.s_var, 0.0))
        self.var += self.rate * m * (err * err - self.var)
        np.maximum(self.var, 1e-6, out=self.var)
        d = s - self.s_mean
        self.s_mean += self.rate * d
        self.s_var += self.rate * (d * d - self.s_var)
        return s, bool(outlier)

    def arrays(self) -> dict:
        return {"prec.var": self.var, "prec.stats": np.array([self.s_mean, self.s_var])}

    def load(self, arr) -> None:
        self.var = np.array(arr["prec.var"])
        self.s_mean, self.s_var = (float(x) for x in arr["prec.stats"])


def hebb_features(key: np.ndarray) -> np.ndarray:
    """Non-negative, sharpened code of a thought for Hebbian association."""
    f = np.concatenate([np.maximum(key, 0.0), np.maximum(-key, 0.0)])
    return f * f


class FastWeights:
    """Hippocampal fast synapses: a dual-trace Hebbian associative matrix.

        A <- l A + (1 - l) (target (x) f(h_act))      what followed each thought
        S <- l S + (1 - l) (observed (x) f(h_act))    how much evidence per channel

    Reading ``A f(h) / S f(h)`` recalls what recently followed thoughts like
    this one (Ba et al. 2016, "Using fast weights to attend to the recent
    past"). The trace fades within tens of moments; replay during sleep and
    idle time carries what matters into the slow decoder (neocortex).
    """

    def __init__(self, d_feat: int, d_out: int, decay: float):
        self.a = np.zeros((d_out, d_feat))
        self.s = np.zeros((d_out, d_feat))
        self.decay = decay

    def read(self, feat: np.ndarray):
        mass = self.s @ feat
        norm = float(feat @ feat) or 1.0
        value = (self.a @ feat) / np.maximum(mass, 1e-12)
        return value, np.clip(mass / norm, 0.0, 1.0)

    def write(self, feat: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        l = self.decay
        self.a *= l
        self.a += (1.0 - l) * np.outer(mask * target, feat)
        self.s *= l
        self.s += (1.0 - l) * np.outer(mask, feat)

    def arrays(self) -> dict:
        return {"fast.a": self.a, "fast.s": self.s}

    def load(self, arr) -> None:
        self.a = np.array(arr["fast.a"])
        self.s = np.array(arr["fast.s"])
