"""Continuous latent dynamics: field encoder, liquid NCP core, SSM memory field.

Maths used (all evaluated at an arbitrary elapsed time ``dt``, so irregular,
asynchronous streams are native and there is no sequence length anywhere):

Liquid core, closed-form continuous-time cell (Hasani et al. 2022, CfC):
    x(t+dt) = sigma(-f * dt/tau) * g + (1 - sigma(-f * dt/tau)) * h
    f = W_f v + b_f,  g = tanh(W_g v + b_g),  h = tanh(W_h v + b_h)

Liquid core, LTC ODE (Hasani et al. 2021), fused semi-implicit Euler:
    dx/dt = -(1/tau + f(v)) * x + f(v) * A
    x <- (x + h f A) / (1 + h (1/tau + f)),  f = sigmoid(W_f v + b_f)

    where v = [z, o_prev, x] (sensory latent, SSM feedback, own state), and
    W_* are sparse Neural-Circuit-Policy masks (sensory -> inter -> command
    -> motor, recurrent command loop).

Cortical columns (``cell="column"``), a continuous-time recurrent network:
    tau dv/dt = -v + W_rec tanh(v) + W_in u + b,   x = tanh(v)
    integrated exactly over dt for the drive held constant (exponential Euler):
    v <- e^{-dt/tau} v + (1 - e^{-dt/tau}) (W_rec tanh(v) + W_in u + b)
    Neurons form columns; each column is densely wired inside and laterally
    coupled to its two neighbours on a ring, so activity spreads as a field.

Liquid adaptation: per-neuron time constants tau drift online so saturated
neurons slow down and quiet ones speed up (homeostatic intrinsic plasticity).

Legendre sensory trace (Voelker et al. 2019, LMU; the HiPPO-LegT operator
underlying S4/Mamba): for every input line, an order-q linear ODE
    theta dm/dt = A m + B u
whose state m is the Legendre-polynomial expansion of the input over the last
``theta`` time units -- a continuous sliding window with no buffer. Exact ZOH
discretisation per dt: Ad = expm(A dt), Bd = A^-1 (Ad - I) B.

Selective state-space field (Gu & Dao 2023, diagonal, exact ZOH of
    dh/dt = -r (h - B(x) y),  y = P x,  delta = softplus(P_d x + b_d) * dt):
    h <- exp(-r delta) h + (1 - exp(-r delta)) B y ;  o = tanh(C(x) . h + D y)
    Decay rates r are log-spaced over three decades: O(1) memory per step,
    no context window, unbounded stream length.
"""
from __future__ import annotations

import numpy as np

from .config import BrainConfig


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


class FieldEncoder:
    """Perceiver-field style projection of a raw continuous signal into latent space."""

    def __init__(self, d_in: int, cfg: BrainConfig, rng: np.random.Generator):
        self.omega = rng.normal(0.0, 1.0, (cfg.fourier, d_in))
        feat = d_in + 2 * cfg.fourier
        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(feat), (cfg.encoder_hidden, feat))
        self.b1 = rng.normal(0.0, 0.1, cfg.encoder_hidden)
        self.w2 = rng.normal(0.0, 1.5 / np.sqrt(cfg.encoder_hidden), (cfg.latent, cfg.encoder_hidden))

    def __call__(self, u: np.ndarray) -> np.ndarray:
        wu = self.omega @ u
        f = np.concatenate([u, np.sin(wu), np.cos(wu)])
        return np.tanh(self.w2 @ np.tanh(self.w1 @ f + self.b1))

    def arrays(self, prefix: str) -> dict:
        return {f"{prefix}.omega": self.omega, f"{prefix}.w1": self.w1,
                f"{prefix}.b1": self.b1, f"{prefix}.w2": self.w2}

    def load(self, prefix: str, arr) -> None:
        for k in ("omega", "w1", "b1", "w2"):
            setattr(self, k, np.array(arr[f"{prefix}.{k}"]))


def column_mask(cfg: BrainConfig, n_in: int, rng: np.random.Generator) -> np.ndarray:
    """Cortical columns: sparse sensory drive, dense intra-column, lateral ring."""
    n = cfg.neurons
    mask = np.zeros((n, n_in + n))
    col = np.arange(n) // cfg.column_size
    ncol = int(col.max()) + 1
    for r in range(n):
        cols = rng.choice(n_in, size=min(cfg.sensory_fanin, n_in), replace=False)
        mask[r, cols] = 1.0
    d = np.abs(col[:, None] - col[None, :])
    d = np.minimum(d, ncol - d)                       # ring distance between columns
    p = np.where(d == 0, cfg.column_intra, np.where(d == 1, cfg.column_lateral, 0.0))
    np.fill_diagonal(p, 0.0)
    mask[:, n_in:] = (rng.random((n, n)) < p).astype(float)
    return mask


def ncp_mask(cfg: BrainConfig, n_in: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse layered wiring over v = [inputs (n_in), neurons (N)]."""
    if cfg.cell == "column":
        return column_mask(cfg, n_in, rng)
    i0, c0, m0 = 0, cfg.inter, cfg.inter + cfg.command
    n = cfg.neurons
    mask = np.zeros((n, n_in + n))

    def pick(row, lo, hi, k):
        cols = rng.choice(np.arange(lo, hi), size=min(k, hi - lo), replace=False)
        mask[row, cols] = 1.0

    for r in range(i0, c0):                       # inter <- sensory latent + SSM feedback
        pick(r, 0, n_in, cfg.sensory_fanin)
    for r in range(c0, m0):                       # command <- inter, command <- command
        pick(r, n_in + i0, n_in + c0, cfg.inter_fanin)
        pick(r, n_in + c0, n_in + m0, cfg.command_recurrent)
    base = cfg.inter + cfg.command + cfg.motor
    for r in range(m0, base):                     # motor <- command
        pick(r, n_in + c0, n_in + m0, cfg.motor_fanin)
    for r in range(base, n):                      # grown association neurons
        pick(r, 0, n_in, cfg.sensory_fanin)
        pick(r, n_in + c0, n_in + m0, cfg.inter_fanin)
    return mask


class LiquidCore:
    """Neural Circuit Policy of liquid neurons (CfC or LTC)."""

    def __init__(self, n_in: int, cfg: BrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        n = cfg.neurons
        self.n_in = n_in
        mask = ncp_mask(cfg, n_in, rng)
        fanin = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        scale = np.ones_like(mask)
        scale[:, n_in:] = cfg.recurrent_scale
        scale[:, :n_in] *= cfg.input_scale

        def head():
            return mask * scale * rng.normal(0.0, 1.0, mask.shape) / np.sqrt(fanin)

        self.wf, self.wg, self.wh = head(), head(), head()
        self.bf = rng.normal(0.0, 0.5, n)
        self.bg = rng.normal(0.0, 0.2, n)
        self.bh = rng.normal(0.0, 0.2, n)
        self.reversal = np.tanh(rng.normal(0.0, 1.0, n))  # LTC reversal potentials A
        self.tau = np.exp(rng.uniform(np.log(0.5), np.log(8.0), n))
        self.reset()

    def reset(self) -> None:
        n = self.cfg.neurons
        self.x = np.zeros(n)
        self.v = np.zeros(n)
        self.activity = np.full(n, self.cfg.activity_target)

    @property
    def motor(self) -> slice:
        c = self.cfg
        return slice(c.inter + c.command, c.inter + c.command + c.motor)

    def grow(self, k: int, rng: np.random.Generator) -> None:
        """Morphogenesis: append k neurons without changing what she computes.

        New neurons receive sparse input (senses + command layer) but start
        with zero outgoing weight, so every existing neuron's dynamics is
        exactly unchanged at the moment of growth (function-preserving, as in
        Net2Net); they begin to matter as the decoder learns to read them."""
        cfg = self.cfg
        n_old, n_in = cfg.neurons - k, self.n_in
        c0, m0 = cfg.inter, cfg.inter + cfg.command
        def widen(w):
            out = np.zeros((n_old + k, n_in + n_old + k))
            out[:n_old, :n_in + n_old] = w
            return out
        heads = [widen(w) for w in (self.wf, self.wg, self.wh)]
        for r in range(n_old, n_old + k):
            ins = rng.choice(n_in, size=min(cfg.sensory_fanin, n_in), replace=False)
            cmd = n_in + rng.choice(np.arange(c0, m0), size=min(cfg.inter_fanin, m0 - c0),
                                    replace=False)
            fan = np.sqrt(len(ins) + len(cmd))
            for w in heads:
                w[r, ins] = cfg.input_scale * rng.normal(0.0, 1.0, len(ins)) / fan
                w[r, cmd] = cfg.recurrent_scale * rng.normal(0.0, 1.0, len(cmd)) / fan
        self.wf, self.wg, self.wh = heads
        self.bf = np.concatenate([self.bf, rng.normal(0.0, 0.5, k)])
        self.bg = np.concatenate([self.bg, rng.normal(0.0, 0.2, k)])
        self.bh = np.concatenate([self.bh, rng.normal(0.0, 0.2, k)])
        self.reversal = np.concatenate([self.reversal, np.tanh(rng.normal(0.0, 1.0, k))])
        self.tau = np.concatenate([self.tau, np.exp(rng.uniform(np.log(0.5), np.log(8.0), k))])
        self.x = np.concatenate([self.x, np.zeros(k)])
        self.v = np.concatenate([self.v, np.zeros(k)])
        self.activity = np.concatenate([self.activity, np.full(k, cfg.activity_target)])

    def step(self, inputs: np.ndarray, dt: float) -> np.ndarray:
        cfg = self.cfg
        if cfg.cell == "column":
            n_in = self.n_in
            drive = self.wf[:, :n_in] @ inputs + self.wf[:, n_in:] @ np.tanh(self.v) + self.bf
            a = np.exp(-dt / self.tau)
            self.v = a * self.v + (1.0 - a) * drive
            x = np.tanh(self.v)
        elif cfg.cell == "ltc":
            h = dt / cfg.ltc_unfolds
            x = self.x
            for _ in range(cfg.ltc_unfolds):
                v = np.concatenate([inputs, x])
                f = sigmoid(self.wf @ v + self.bf)
                x = (x + h * f * self.reversal) / (1.0 + h * (1.0 / self.tau + f))
        else:
            v = np.concatenate([inputs, self.x])
            f = self.wf @ v + self.bf
            g = np.tanh(self.wg @ v + self.bg)
            hh = np.tanh(self.wh @ v + self.bh)
            gate = sigmoid(-f * (dt / self.tau))
            x = gate * g + (1.0 - gate) * hh
        self.x = x
        if cfg.homeostasis > 0.0:
            self.activity += 0.01 * (np.abs(x) - self.activity)
            self.tau *= np.exp(cfg.homeostasis * (self.activity - cfg.activity_target))
            np.clip(self.tau, cfg.tau_min, cfg.tau_max, out=self.tau)
        return x

    def arrays(self) -> dict:
        return {"core.wf": self.wf, "core.wg": self.wg, "core.wh": self.wh,
                "core.bf": self.bf, "core.bg": self.bg, "core.bh": self.bh,
                "core.reversal": self.reversal, "core.tau": self.tau,
                "core.x": self.x, "core.v": self.v, "core.activity": self.activity}

    def load(self, arr) -> None:
        for k in ("wf", "wg", "wh", "bf", "bg", "bh", "reversal", "tau", "x", "v", "activity"):
            setattr(self, k, np.array(arr[f"core.{k}"]))


class SelectiveSSM:
    """Diagonal selective state-space memory field with multi-scale decay."""

    def __init__(self, cfg: BrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        n, d, s = cfg.neurons, cfg.ssm_channels, cfg.ssm_state
        self.p_in = rng.normal(0.0, 1.0 / np.sqrt(n), (d, n))
        self.p_dt = rng.normal(0.0, 0.5 / np.sqrt(n), (d, n))
        self.b_dt = np.zeros(d)
        self.w_b = rng.normal(0.0, 1.0 / np.sqrt(n), (s, n))
        self.w_c = rng.normal(0.0, 1.0 / np.sqrt(n), (s, n))
        rates = np.geomspace(cfg.ssm_rate_min, cfg.ssm_rate_max, d)
        self.rate = rates[:, None] * np.linspace(1.0, 2.0, s)[None, :]
        self.skip = rng.normal(0.0, 0.5, d)
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((self.cfg.ssm_channels, self.cfg.ssm_state))
        self.o = np.zeros(self.cfg.ssm_channels)

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        y = self.p_in @ x
        delta = softplus(self.p_dt @ x + self.b_dt) * (dt * self.cfg.dt_scale)
        decay = np.exp(-self.rate * delta[:, None])
        b = np.tanh(self.w_b @ x)
        c = self.w_c @ x
        self.h = decay * self.h + (1.0 - decay) * (y[:, None] * b[None, :])
        self.o = np.tanh(self.h @ c / np.sqrt(self.cfg.ssm_state) + self.skip * y)
        return self.o

    def arrays(self) -> dict:
        return {"ssm.p_in": self.p_in, "ssm.p_dt": self.p_dt, "ssm.b_dt": self.b_dt,
                "ssm.w_b": self.w_b, "ssm.w_c": self.w_c, "ssm.rate": self.rate,
                "ssm.skip": self.skip, "ssm.h": self.h, "ssm.o": self.o}

    def load(self, arr) -> None:
        for k in ("p_in", "p_dt", "b_dt", "w_b", "w_c", "rate", "skip", "h", "o"):
            setattr(self, k, np.array(arr[f"ssm.{k}"]))


def expm(m: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring Taylor (small matrices)."""
    norm = np.linalg.norm(m, 1)
    k = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    x = m / (2.0 ** k)
    out = np.eye(m.shape[0])
    term = np.eye(m.shape[0])
    for i in range(1, 20):
        term = term @ x / i
        out = out + term
    for _ in range(k):
        out = out @ out
    return out


class LegendreTrace:
    """Continuous sliding-window memory of the raw input (LMU / HiPPO-LegT)."""

    def __init__(self, dim: int, order: int, window: float):
        i = np.arange(order)[:, None]
        j = np.arange(order)[None, :]
        self.a = (2 * i + 1) / window * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
        self.b = (2 * np.arange(order) + 1) * (-1.0) ** np.arange(order) / window
        self._cache = {}
        self.m = np.zeros((order, dim))

    def reset(self) -> None:
        self.m[:] = 0.0

    def discretise(self, dt: float):
        key = round(float(dt), 9)
        if key not in self._cache:
            ad = expm(self.a * dt)
            bd = np.linalg.solve(self.a, (ad - np.eye(ad.shape[0])) @ self.b)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = (ad, bd)
        return self._cache[key]

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        ad, bd = self.discretise(dt)
        self.m = ad @ self.m + np.outer(bd, u)
        return self.m
