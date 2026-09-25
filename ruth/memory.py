"""Associative attractor memory: a modern continuous Hopfield network.

Ramsauer et al. 2020 ("Hopfield Networks is All You Need"):
    energy   E(xi) = -1/beta * lse(beta, K xi) + 1/2 xi.xi + const
    update   xi   <- K^T softmax(beta K xi)
Stored patterns are the minima (basins) of E; a query "rolls downhill" into
the nearest basin in one or a few updates. Keys live in a compact projected
thought-space; each key is paired with a value (what followed that thought),
so a recalled basin also yields an expectation about the world.

Each value also records *which of its channels were actually observed*.
Recall averages every channel only over real evidence, so a guess never
comes back later disguised as a memory.

Two stores, like hippocampus and neocortex:
* working  - fast, plastic ring buffer written when something is surprising;
* longterm - slow landscape; ``consolidate`` (sleep) merges working episodes
  into existing basins (running average) or digs new ones, evicting the least
  used basin when full.
"""
from __future__ import annotations

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class AttractorMemory:
    def __init__(self, key_dim: int, value_dim: int, working_capacity: int,
                 longterm_capacity: int, beta: float, iters: int = 1,
                 merge_threshold: float = 0.97):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.beta = beta
        self.iters = iters
        self.merge_threshold = merge_threshold
        self.wk = np.zeros((working_capacity, key_dim))
        self.wv = np.zeros((working_capacity, value_dim))
        self.wm = np.zeros((working_capacity, value_dim))   # observed channels
        self.w_n = 0          # filled slots
        self.w_head = 0       # ring pointer
        self.lk = np.zeros((longterm_capacity, key_dim))
        self.lv = np.zeros((longterm_capacity, value_dim))
        self.lm = np.zeros((longterm_capacity, value_dim))  # evidence per channel
        self.l_count = np.zeros(longterm_capacity)
        self.l_use = np.zeros(longterm_capacity)
        self.l_n = 0

    # -- storage -------------------------------------------------------
    @property
    def working_full(self) -> bool:
        return self.w_n == self.wk.shape[0]

    def store(self, key: np.ndarray, value: np.ndarray, mask=None) -> None:
        i = self.w_head
        self.wk[i] = _unit(key)
        self.wv[i] = value
        self.wm[i] = 1.0 if mask is None else mask
        self.w_head = (i + 1) % self.wk.shape[0]
        self.w_n = min(self.w_n + 1, self.wk.shape[0])

    def _patterns(self):
        k = np.concatenate([self.wk[: self.w_n], self.lk[: self.l_n]])
        v = np.concatenate([self.wv[: self.w_n], self.lv[: self.l_n]])
        m = np.concatenate([self.wm[: self.w_n], np.minimum(self.lm[: self.l_n], 1.0)])
        # a basin merged from c episodes carries mass c: log-prior on its score
        mass = np.concatenate([np.zeros(self.w_n), np.log(self.l_count[: self.l_n])])
        return k, v, m, mass

    # -- recall --------------------------------------------------------
    def recall(self, query: np.ndarray, track_use: bool = True):
        """Return (value, attractor_key, confidence, evidence).

        ``evidence`` (per channel, 0..1) is how much of the recalled mass
        actually observed that channel. Empty memory -> zeros. Working and
        long-term stores are scored in place (no copying per moment)."""
        wn, ln = self.w_n, self.l_n
        if wn + ln == 0:
            z = np.zeros(self.value_dim)
            return z, np.zeros(self.key_dim), 0.0, z.copy()
        wk, lk = self.wk[:wn], self.lk[:ln]
        mass = np.log(self.l_count[:ln])
        xi = _unit(query)
        for _ in range(max(1, self.iters)):
            sw = self.beta * (wk @ xi)
            sl = self.beta * (lk @ xi) + mass
            top = max(sw.max() if wn else -np.inf, sl.max() if ln else -np.inf)
            pw, pl = np.exp(sw - top), np.exp(sl - top)
            z = pw.sum() + pl.sum()
            pw /= z
            pl /= z
            xi = wk.T @ pw + lk.T @ pl
        if track_use and ln:
            self.l_use[:ln] += pl
        wm, lm = self.wm[:wn], np.minimum(self.lm[:ln], 1.0)
        weight = wm.T @ pw + lm.T @ pl
        value = ((self.wv[:wn] * wm).T @ pw + (self.lv[:ln] * lm).T @ pl) / np.maximum(weight, 1e-12)
        conf = float(max(pw.max() if wn else 0.0, pl.max() if ln else 0.0))
        return value, xi, conf, weight

    def energy(self, xi: np.ndarray) -> float:
        k, _, _, mass = self._patterns()
        if k.shape[0] == 0:
            return 0.5 * float(xi @ xi)
        s = self.beta * (k @ xi) + mass
        lse = s.max() + np.log(np.exp(s - s.max()).sum())
        return float(-lse / self.beta + 0.5 * xi @ xi)

    # -- sleep ---------------------------------------------------------
    def consolidate(self) -> dict:
        merged = created = evicted = 0
        for i in range(self.w_n):
            key, val, obs = self.wk[i], self.wv[i], self.wm[i]
            if self.l_n:
                sims = self.lk[: self.l_n] @ key
                j = int(np.argmax(sims))
                if sims[j] >= self.merge_threshold:
                    c = self.l_count[j]
                    self.lk[j] = _unit((c * self.lk[j] + key) / (c + 1))
                    ev = self.lm[j]
                    tot = ev + obs
                    self.lv[j] = np.where(tot > 0, (ev * self.lv[j] + obs * val) / np.maximum(tot, 1e-12),
                                          self.lv[j])
                    self.lm[j] = tot
                    self.l_count[j] = c + 1
                    merged += 1
                    continue
            if self.l_n < self.lk.shape[0]:
                j = self.l_n
                self.l_n += 1
            else:
                j = int(np.argmin(self.l_use[: self.l_n] + self.l_count[: self.l_n]))
                evicted += 1
            self.lk[j], self.lv[j], self.lm[j] = key, val, obs
            self.l_count[j] = 1.0
            self.l_use[j] = 0.0
            created += 1
        self.w_n = self.w_head = 0
        self.l_use *= 0.5  # usage decays each night
        return {"merged": merged, "created": created, "evicted": evicted,
                "longterm": self.l_n}

    # -- persistence ---------------------------------------------------
    def arrays(self) -> dict:
        return {"mem.wk": self.wk, "mem.wv": self.wv, "mem.lk": self.lk, "mem.lv": self.lv,
                "mem.wm": self.wm, "mem.lm": self.lm,
                "mem.l_count": self.l_count, "mem.l_use": self.l_use,
                "mem.meta": np.array([self.w_n, self.w_head, self.l_n], dtype=float)}

    def load(self, arr) -> None:
        for k in ("wk", "wv", "wm", "lk", "lv", "lm", "l_count", "l_use"):
            setattr(self, k, np.array(arr[f"mem.{k}"]))
        self.w_n, self.w_head, self.l_n = (int(x) for x in arr["mem.meta"])
