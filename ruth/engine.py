"""The Continuous Latent Cognitive Engine -- Ruth's token-free brain.

One moment of cognition (``Brain.step``), organised like a brain:

  raw continuous signals u(t), elapsed dt              (no tokens, ever)
    -> prediction error vs. what she anticipated        (predictive coding)
    -> plasticity: neocortical decoder (RLS), hippocampal fast weights
       (Hebbian), episodic write into the attractor landscape
    -> field encoders per sense
    -> THALAMUS: energy-gated routing, g = softmax(-beta E_k),
       E_k(z) = 1/2 z.M z - b_k.z, with b_k learned from salience
    -> CORTEX: liquid core (CfC / LTC / cortical columns)  <- SSM feedback
    -> CEREBELLUM: forward model of the cortex; its error e = x_hat - x is
       latent surprise, and in imagination it damps drift
    -> SSM memory field, Legendre traces (fast + context)
    -> HIPPOCAMPUS: attractor recall (episodes) + fast-weight recall (recent)
    -> precision-weighted fusion of three experts -> prediction of the next
       moment. What she was *told* (e.g. privacy) comes from episodes only.

Speaking and thinking are the same loop run closed (see mind.py); the basal
ganglia (mind.py) choose which imagined trajectory is acted on.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from . import paths
from .config import BrainConfig
from .dynamics import FieldEncoder, LegendreTrace, LiquidCore, SelectiveSSM
from .memory import AttractorMemory
from .plasticity import FastWeights, Precision, RLSDecoder, hebb_features
from .senses import BYTE_DIM, byte_code, byte_decode, bytes_signal


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _insert(a: np.ndarray, axis: int, pos: int, k: int, fill: float = 0.0) -> np.ndarray:
    shape = list(a.shape)
    shape[axis] = k
    return np.insert(a, [pos] * k, 0.0, axis=axis) if fill == 0.0 else \
        np.concatenate([np.take(a, range(pos), axis=axis), np.full(shape, fill),
                        np.take(a, range(pos, a.shape[axis]), axis=axis)], axis=axis)


def _grow_p(p: np.ndarray, pos: int, k: int, delta: float) -> np.ndarray:
    """Insert k fresh, uncorrelated coordinates into an RLS inverse-covariance."""
    p = np.insert(np.insert(p, [pos] * k, 0.0, axis=0), [pos] * k, 0.0, axis=1)
    p[pos:pos + k, pos:pos + k] = np.eye(k) / delta
    return p


class Brain:
    LIVE = ("phi", "pred", "key", "wpred", "recall", "evidence", "fast", "fevid", "feat")

    def __init__(self, cfg: BrainConfig | None = None):
        self.cfg = cfg = cfg or BrainConfig()
        rng = np.random.default_rng(cfg.seed)
        grown, cfg.grown = cfg.grown, 0          # build the newborn, then regrow
        self.encoders = {name: FieldEncoder(dim, cfg, rng) for name, dim in cfg.senses.items()}
        self.core = LiquidCore(cfg.latent + cfg.ssm_channels, cfg, rng)
        self.ssm = SelectiveSSM(cfg, rng)
        self.in_slices, off = {}, 0
        for name, dim in cfg.senses.items():
            self.in_slices[name] = slice(off, off + dim)
            off += dim
        self.trace_index = (np.concatenate([np.arange(self.in_slices[n].start, self.in_slices[n].stop)
                                            for n in cfg.trace_senses]).astype(int)
                            if cfg.trace_senses else np.arange(cfg.input_dim))
        self.trace = LegendreTrace(self.trace_index.size, cfg.lmu_order, cfg.lmu_window)
        self.context = LegendreTrace(self.trace_index.size, cfg.context_order, cfg.context_window)
        self.episode = LegendreTrace(self.trace_index.size, cfg.episode_order, cfg.episode_window) \
            if cfg.episode_weight > 0 else None
        thought = cfg.neurons + cfg.ssm_channels
        self.key_proj = rng.normal(0.0, 1.0 / np.sqrt(thought), (cfg.key_dim, thought))
        key_dim = self.trace.m.size + self.context.m.size + cfg.key_dim + \
            (self.episode.m.size if self.episode is not None else 0)
        self.memory = AttractorMemory(key_dim, cfg.output_dim, cfg.working_capacity,
                                      cfg.longterm_capacity, cfg.beta, cfg.hopfield_iters,
                                      cfg.merge_threshold)
        self.fast = FastWeights(2 * key_dim, cfg.output_dim, cfg.fast_decay)
        trace_dim = self.trace.m.size if cfg.trace_in_decoder else 0
        self.phi_dim = cfg.input_dim + thought + trace_dim + cfg.output_dim + 2
        self.decoder = RLSDecoder(self.phi_dim, cfg.output_dim, cfg.rls_lambda, cfg.rls_delta)
        self.cerebellum = RLSDecoder(cfg.neurons + cfg.latent + 1, cfg.neurons,
                                     cfg.cerebellum_lambda, cfg.rls_delta)
        self.salience = {name: np.zeros(cfg.latent) for name in cfg.senses}   # thalamic b_k
        self.precision = Precision(cfg.output_dim)
        # running error variance of each expert: decoder, attractor recall, fast weights
        self.expert_var = np.ones((3, cfg.output_dim))
        self.replay_phi = np.zeros((cfg.replay_capacity, self.phi_dim))
        self.replay_y = np.zeros((cfg.replay_capacity, cfg.output_dim))
        self.replay_m = np.zeros((cfg.replay_capacity, cfg.output_dim))
        self.replay_seen = 0
        self._rng = np.random.default_rng(cfg.seed + 1)
        self.target_index = np.concatenate(
            [np.arange(self.in_slices[n].start, self.in_slices[n].stop) for n in cfg.target_names])
        self.out_slices, off = {}, 0
        for name in cfg.target_names:
            self.out_slices[name] = slice(off, off + cfg.senses[name])
            off += cfg.senses[name]
        self.taught_only = [self.in_slices[n] for n in cfg.teaching_signals if n in cfg.senses]
        # what one is *told* (e.g. "this is private") is episodic knowledge: it is
        # predicted from specific recalled episodes only, never from a habit
        self.episodic_out = np.concatenate(
            [np.arange(self.out_slices[n].start, self.out_slices[n].stop)
             for n in cfg.teaching_signals if n in self.out_slices] or [np.zeros(0)]).astype(int)
        self.steps = 0
        self.sleeps = 0
        for k in self.LIVE:
            setattr(self, f"_{k}", None)
        self.last = {}
        if grown:
            self.grow_neurons(grown, np.random.default_rng(cfg.seed + 2))

    # ------------------------------------------------------------------
    # one moment of cognition
    # ------------------------------------------------------------------
    def step(self, inputs: dict, dt: float = 1.0, learn: bool = True) -> np.ndarray:
        """Perceive ``inputs`` ({sense: vector}) after ``dt`` time units.

        Returns the brain's continuous prediction of the next moment's targets.
        """
        cfg = self.cfg
        u = np.zeros(cfg.input_dim)
        seen = np.zeros(cfg.input_dim)
        for name, vec in inputs.items():
            u[self.in_slices[name]] = vec
            seen[self.in_slices[name]] = 1.0
        target = u[self.target_index]
        mask = seen[self.target_index]
        for sl in self.taught_only:   # told, not perceived: target only
            u[sl] = 0.0
        perceived = [n for n in inputs if n not in cfg.teaching_signals]
        info = {}
        err = None
        if self._phi is not None:
            err = mask * (target - self._pred)
            werr = mask * (target - self._wpred)
            if learn:
                rate = self.precision.rate * mask
                self.expert_var[0] += rate * (werr * werr - self.expert_var[0])
                self.expert_var[1] += rate * self._evidence * ((target - self._recall) ** 2
                                                               - self.expert_var[1])
                self.expert_var[2] += rate * self._fevid * ((target - self._fast) ** 2
                                                            - self.expert_var[2])
                np.maximum(self.expert_var, 1e-6, out=self.expert_var)
                s, outlier = self.precision.observe(err, cfg.surprise_k, mask)
            else:  # imagination / own speech: measure, don't adapt
                s = 0.5 * float(np.sum(err * err / self.precision.var) / max(mask.sum(), 1.0))
                outlier = False
            info["surprise"] = s
            if learn:
                self.decoder.update(self._phi, werr)                    # neocortex (slow)
                self.fast.write(self._feat, target, mask)               # hippocampus (fast)
                self._replay_add(self._phi, target, mask)
                if outlier and self.steps >= cfg.warmup:
                    self.memory.store(self._key, target, mask)   # only what was observed
                    info["stored"] = True
                if cfg.auto_sleep and self.memory.working_full:
                    info["sleep"] = self.sleep()

        # THALAMUS: energy-gated routing of the senses into the cortex
        zs = {n: self.encoders[n](u[self.in_slices[n]]) for n in perceived}
        if not zs:
            z = np.zeros(cfg.latent)
        elif len(zs) == 1:
            z = next(iter(zs.values()))
        else:
            names = list(zs)
            energy = np.array([0.5 * zs[n] @ zs[n] - self.salience[n] @ zs[n] for n in names])
            g = np.exp(-cfg.thalamic_beta * (energy - energy.min()))
            g /= g.sum()
            z = sum(gi * zs[n] for gi, n in zip(g, names))
            info["routing"] = {n: round(float(gi), 3) for n, gi in zip(names, g)}
            if learn and err is not None:
                sal = {}
                for n in names:
                    if n in self.out_slices:
                        sl = self.out_slices[n]
                        sal[n] = float(np.mean(err[sl] ** 2 / self.precision.var[sl]))
                if len(sal) > 1:
                    mean = np.mean(list(sal.values()))
                    for n, v in sal.items():
                        b = self.salience[n]
                        b += cfg.thalamic_rate * ((v - mean) * zs[n] - 0.01 * b)

        # CORTEX
        x_prev = self.core.x
        x = self.core.step(np.concatenate([z, self.ssm.o]), dt)

        # CEREBELLUM: forward model of the cortex from its last state and the
        # efference copy of what is arriving now; e = anticipated - actual
        if cfg.cerebellum:
            psi = np.concatenate([x_prev, z, [1.0]])
            x_hat = self.cerebellum.predict(psi)
            e = x_hat - x
            info["latent_surprise"] = float(np.mean(e * e))
            if learn:
                self.cerebellum.update(psi, -e)
            elif cfg.cerebellar_gain > 0.0:      # imagination: damp drift toward the learnt flow
                x = np.clip(x + cfg.cerebellar_gain * e, -1.0, 1.0)
                self.core.x = x

        o = self.ssm.step(x, dt)
        trace = self.trace.step(u[self.trace_index], dt).ravel()
        context = self.context.step(u[self.trace_index], dt).ravel()
        thought = self.key_proj @ np.concatenate([x, o])
        parts = [_unit(trace), cfg.context_weight * _unit(context)]
        if self.episode is not None:
            parts.append(cfg.episode_weight * _unit(self.episode.step(u[self.trace_index], dt).ravel()))
        key = np.concatenate(parts + [cfg.thought_weight * _unit(thought)])

        # HIPPOCAMPUS: episodic attractors + recent fast associations
        r, _, conf, evidence = self.memory.recall(key, track_use=learn)
        feat = hebb_features(key)
        fast, fevid = self.fast.read(feat)

        parts = [u, x, o, trace, r, [conf, 1.0]] if cfg.trace_in_decoder else [u, x, o, r, [conf, 1.0]]
        phi = np.concatenate(parts)
        wpred = self.decoder.predict(phi)
        pw = 1.0 / self.expert_var[0]
        pr = (evidence / self.expert_var[1]) if conf > 0.0 else np.zeros_like(pw)
        pf = fevid / self.expert_var[2]
        pred = (pw * wpred + pr * r + pf * fast) / (pw + pr + pf)
        pred[self.episodic_out] = (evidence * r)[self.episodic_out]

        self._phi, self._pred, self._key = phi, pred, key
        self._wpred, self._recall, self._evidence = wpred, r, evidence
        self._fast, self._fevid, self._feat = fast, fevid, feat
        self.steps += 1
        info["confidence"] = conf
        self.last = info
        return pred

    def prediction(self, sense: str) -> np.ndarray:
        return self._pred[self.out_slices[sense]]

    def novelty(self) -> float:
        """Epistemic uncertainty of the current state (RLS posterior variance)."""
        return self.decoder.novelty(self._phi) if self._phi is not None else 1.0

    # ------------------------------------------------------------------
    # dynamic-state snapshots (imagination / counterfactual rollouts)
    # ------------------------------------------------------------------
    def snapshot(self):
        return {"x": self.core.x.copy(), "v": self.core.v.copy(),
                "activity": self.core.activity.copy(), "tau": self.core.tau.copy(),
                "h": self.ssm.h.copy(), "o": self.ssm.o.copy(), "trace": self.trace.m.copy(),
                "context": self.context.m.copy(), "steps": self.steps,
                "episode": None if self.episode is None else self.episode.m.copy(),
                **{k: getattr(self, f"_{k}") for k in self.LIVE}}

    def restore(self, snap) -> None:
        self.core.x, self.core.v = snap["x"].copy(), snap["v"].copy()
        self.core.activity, self.core.tau = snap["activity"].copy(), snap["tau"].copy()
        self.ssm.h, self.ssm.o = snap["h"].copy(), snap["o"].copy()
        self.trace.m, self.context.m = snap["trace"].copy(), snap["context"].copy()
        if self.episode is not None:
            self.episode.m = snap["episode"].copy()
        self.steps = snap["steps"]
        for k in self.LIVE:
            setattr(self, f"_{k}", snap[k])

    # ------------------------------------------------------------------
    # sleep and background replay (hippocampus -> neocortex)
    # ------------------------------------------------------------------
    def _replay_add(self, phi, y, mask) -> None:
        cap = self.cfg.replay_capacity
        self.replay_seen += 1
        if self.replay_seen <= cap:
            i = self.replay_seen - 1
        else:  # reservoir sampling keeps a uniform sample of the whole lifetime
            i = int(self._rng.integers(0, self.replay_seen))
            if i >= cap:
                return
        self.replay_phi[i] = phi
        self.replay_y[i] = y
        self.replay_m[i] = mask

    def replay(self, n: int | None = None) -> int:
        """Rehearse lived moments through the slow decoder without forgetting."""
        have = min(self.replay_seen, self.cfg.replay_capacity)
        idx = self._rng.permutation(have)[: (have if n is None else n)]
        for i in idx:
            phi = self.replay_phi[i]
            err = self.replay_m[i] * (self.replay_y[i] - self.decoder.predict(phi))
            self.decoder.update(phi, err, lam=1.0)
        return int(len(idx))

    def sleep(self) -> dict:
        """Offline consolidation: working memory -> long-term attractor basins,
        then rehearse a lifetime sample through the decoder without forgetting."""
        stats = self.memory.consolidate()
        stats["replayed"] = sum(self.replay() for _ in range(self.cfg.replay_passes))
        self.sleeps += 1
        return stats

    # ------------------------------------------------------------------
    # morphogenesis and introspection
    # ------------------------------------------------------------------
    def grow_neurons(self, k: int, rng: np.random.Generator | None = None) -> None:
        """Add k cortical neurons; function-preserving at the moment of growth."""
        rng = rng or np.random.default_rng(self.cfg.seed + 3 + self.cfg.grown)
        cfg = self.cfg
        n_old = cfg.neurons
        cfg.grown += k
        self.core.grow(k, rng)
        s = self.ssm
        s.p_in, s.p_dt = _insert(s.p_in, 1, n_old, k), _insert(s.p_dt, 1, n_old, k)
        s.w_b, s.w_c = _insert(s.w_b, 1, n_old, k), _insert(s.w_c, 1, n_old, k)
        self.key_proj = _insert(self.key_proj, 1, n_old, k)
        at = cfg.input_dim + n_old                  # position of new neurons inside phi
        self.decoder.w = _insert(self.decoder.w, 1, at, k)
        self.decoder.p = _grow_p(self.decoder.p, at, k, cfg.rls_delta)
        self.replay_phi = _insert(self.replay_phi, 1, at, k)
        cb = self.cerebellum
        cb.w = _insert(_insert(cb.w, 1, n_old, k), 0, n_old, k)
        cb.p = _grow_p(cb.p, n_old, k, cfg.rls_delta)
        self.phi_dim += k
        if self._phi is not None:
            self._phi = _insert(self._phi, 0, at, k)

    def grow_memory(self, extra: int) -> None:
        m = self.memory
        m.lk = np.concatenate([m.lk, np.zeros((extra, m.lk.shape[1]))])
        m.lv = np.concatenate([m.lv, np.zeros((extra, m.lv.shape[1]))])
        m.lm = np.concatenate([m.lm, np.zeros((extra, m.lm.shape[1]))])
        m.l_count = np.concatenate([m.l_count, np.zeros(extra)])
        m.l_use = np.concatenate([m.l_use, np.zeros(extra)])
        self.cfg.longterm_capacity += extra

    def graph(self) -> dict:
        """Her own operational graph, as data she can inspect and patch."""
        cfg = self.cfg
        n_in = self.core.n_in
        rec = self.core.wf[:, n_in:] != 0
        nodes = {
            "senses": {n: d for n, d in cfg.senses.items()},
            "thalamus": {"routes": len(cfg.senses), "beta": cfg.thalamic_beta},
            "cortex": {"cell": cfg.cell, "neurons": cfg.neurons, "grown": cfg.grown,
                       "synapses": int((self.core.wf != 0).sum()),
                       "recurrent_density": round(float(rec.mean()), 4),
                       "tau_range": [round(float(self.core.tau.min()), 3),
                                     round(float(self.core.tau.max()), 3)]},
            "cerebellum": {"inputs": self.cerebellum.w.shape[1], "outputs": self.cerebellum.w.shape[0]},
            "ssm_field": {"channels": cfg.ssm_channels, "state": cfg.ssm_state},
            "traces": {"fast": list(self.trace.m.shape), "context": list(self.context.m.shape)},
            "hippocampus": {"key_dim": self.memory.key_dim, "working": self.memory.w_n,
                            "longterm": self.memory.l_n, "capacity": self.memory.lk.shape[0],
                            "fast_weights": list(self.fast.a.shape)},
            "neocortex_decoder": {"inputs": self.phi_dim, "outputs": cfg.output_dim},
        }
        edges = [["senses", "thalamus"], ["thalamus", "cortex"], ["ssm_field", "cortex"],
                 ["cortex", "cerebellum"], ["cortex", "ssm_field"], ["senses", "traces"],
                 ["traces", "hippocampus"], ["cortex", "hippocampus"], ["ssm_field", "hippocampus"],
                 ["hippocampus", "neocortex_decoder"], ["cortex", "neocortex_decoder"]]
        params = sum(a.size for k, a in self.arrays().items()
                     if not k.startswith(("replay.", "mem.", "rls.p", "cb.p", "live.")))
        return {"nodes": nodes, "edges": edges, "parameters": int(params)}

    # ------------------------------------------------------------------
    # text as a continuous stream
    # ------------------------------------------------------------------
    def learn_bytes(self, data: bytes, dt: float = 1.0, learn: bool = True,
                    valence=None) -> dict:
        """Stream bytes through the brain; prequential (predict-then-learn) score.

        ``valence`` (optional, one value per byte) is what she is *told* about
        each moment (e.g. -1 = private). It is a teaching signal: learned as a
        target, never perceived as input."""
        sig = bytes_signal(data)
        told = "valence" in self.cfg.senses and valence is not None
        bits_ok = bytes_ok = 0
        sl = self.out_slices["text"]
        for t in range(sig.shape[0]):
            if self._pred is not None:
                guess = self._pred[sl]
                bits_ok += int(np.sum((guess[:8] > 0) == (sig[t, :8] > 0)))
                bytes_ok += int(byte_decode(guess) == data[t])
            inputs = {"text": sig[t]}
            if told:
                inputs["valence"] = [float(valence[t])]
            self.step(inputs, dt, learn)
        n = max(sig.shape[0], 1)
        return {"bytes": sig.shape[0], "bit_accuracy": bits_ok / (8 * n),
                "byte_accuracy": bytes_ok / n}

    def generate(self, prompt: bytes = b"", n: int = 64, temperature: float = 0.0,
                 learn_prompt: bool = True, stop: bytes | None = None) -> bytes:
        """Closed-loop articulation: emit predictions and hear them back."""
        for b in prompt:
            self.step({"text": byte_code(b)}, 1.0, learn_prompt)
        out = bytearray()
        sl = self.out_slices["text"]
        for _ in range(n):
            if self._pred is None:
                self.step({"text": byte_code(0x20)}, 1.0, False)
            v = self._pred[sl].copy()
            if temperature > 0:
                v[:8] += temperature * self._rng.normal(size=8)
            b = byte_decode(v)
            out.append(b)
            if stop is not None and bytes(out).endswith(stop):
                break
            self.step({"text": byte_code(b)}, 1.0, False)
        return bytes(out)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def arrays(self) -> dict:
        arr = {}
        for name, enc in self.encoders.items():
            arr.update(enc.arrays(f"enc.{name}"))
            arr[f"thal.{name}"] = self.salience[name]
        arr.update(self.core.arrays())
        arr.update(self.ssm.arrays())
        arr.update(self.memory.arrays())
        arr.update(self.decoder.arrays())
        arr.update(self.fast.arrays())
        arr["cb.w"], arr["cb.p"] = self.cerebellum.w, self.cerebellum.p
        arr.update(self.precision.arrays())
        arr["expert_var"] = self.expert_var
        arr["key_proj"] = self.key_proj
        arr["trace"] = self.trace.m
        arr["context"] = self.context.m
        if self.episode is not None:
            arr["episode"] = self.episode.m
        arr["replay.phi"] = self.replay_phi
        arr["replay.y"] = self.replay_y
        arr["replay.m"] = self.replay_m
        arr["counters"] = np.array([self.steps, self.sleeps, self.replay_seen], dtype=float)
        for k in self.LIVE:
            v = getattr(self, f"_{k}")
            if v is not None:
                arr[f"live.{k}"] = v
        return arr

    def save(self, path: str | None = None) -> str:
        path = path or paths.brain_state()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, config=np.array(self.cfg.to_json()), **self.arrays())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "Brain":
        path = path or paths.brain_state()
        with np.load(path, allow_pickle=False) as arr:
            brain = cls(BrainConfig.from_json(str(arr["config"])))
            for name, enc in brain.encoders.items():
                enc.load(f"enc.{name}", arr)
                brain.salience[name] = np.array(arr[f"thal.{name}"])
            brain.core.load(arr)
            brain.ssm.load(arr)
            brain.memory.load(arr)
            brain.decoder.load(arr)
            brain.fast.load(arr)
            brain.cerebellum.w, brain.cerebellum.p = np.array(arr["cb.w"]), np.array(arr["cb.p"])
            brain.precision.load(arr)
            brain.expert_var = np.array(arr["expert_var"])
            brain.key_proj = np.array(arr["key_proj"])
            brain.trace.m = np.array(arr["trace"])
            brain.context.m = np.array(arr["context"])
            if brain.episode is not None:
                brain.episode.m = np.array(arr["episode"])
            brain.replay_phi = np.array(arr["replay.phi"])
            brain.replay_y = np.array(arr["replay.y"])
            brain.replay_m = np.array(arr["replay.m"])
            brain.steps, brain.sleeps, brain.replay_seen = (int(x) for x in arr["counters"])
            for k in cls.LIVE:
                if f"live.{k}" in arr:
                    setattr(brain, f"_{k}", np.array(arr[f"live.{k}"]))
        return brain

    @classmethod
    def load_or_create(cls, path: str | None = None, cfg: BrainConfig | None = None) -> "Brain":
        path = path or paths.brain_state()
        return cls.load(path) if os.path.exists(path) else cls(cfg)

    def export_core(self, path: str) -> str:
        """Write the flat binary model read by the C runtime (ruth/csrc).

        The C runtime drives the text channel at dt = 1 (a byte stream); all
        other senses stay silent there. Format: b"RUTHBRN1", u32 count, then
        per array: u32 name length, name, u32 element count, float64 data.
        """
        cfg = self.cfg
        if "text" not in cfg.senses or "text" not in cfg.target_names:
            raise ValueError("the C runtime drives the text channel; brain has none")
        ad, bd = self.trace.discretise(1.0)
        scalars = {
            "input_dim": cfg.input_dim, "output_dim": cfg.output_dim,
            "text_in": self.in_slices["text"].start, "text_out": self.out_slices["text"].start,
            "neurons": cfg.neurons, "ssm_channels": cfg.ssm_channels, "ssm_state": cfg.ssm_state,
            "latent": cfg.latent, "key_dim": cfg.key_dim, "phi_dim": self.phi_dim,
            "cell": {"cfc": 0, "ltc": 1, "column": 2}[cfg.cell], "ltc_unfolds": cfg.ltc_unfolds,
            "fourier": cfg.fourier, "encoder_hidden": cfg.encoder_hidden,
            "lmu_order": cfg.lmu_order, "trace_lines": self.trace_index.size,
            "trace_in_decoder": int(cfg.trace_in_decoder), "thought_weight": cfg.thought_weight,
            "context_order": cfg.context_order, "context_weight": cfg.context_weight,
            "episode_order": cfg.episode_order if self.episode is not None else 0,
            "episode_weight": cfg.episode_weight if self.episode is not None else 0.0,
            "dt_scale": cfg.dt_scale, "homeostasis": cfg.homeostasis,
            "activity_target": cfg.activity_target, "tau_min": cfg.tau_min, "tau_max": cfg.tau_max,
            "beta": cfg.beta, "hopfield_iters": cfg.hopfield_iters,
            "merge_threshold": cfg.merge_threshold, "rls_lambda": cfg.rls_lambda,
            "surprise_k": cfg.surprise_k, "warmup": cfg.warmup, "auto_sleep": int(cfg.auto_sleep),
            "prec_rate": self.precision.rate, "steps": self.steps,
            "valence_out": self.out_slices["valence"].start if "valence" in self.out_slices else -1,
            "inhibit": getattr(self, "inhibit_level", 0.35),
            "fast_decay": cfg.fast_decay, "cerebellum": int(cfg.cerebellum),
            "cerebellar_gain": cfg.cerebellar_gain, "cerebellum_lambda": cfg.cerebellum_lambda,
            "has_live": 0 if self._phi is None else 1,
        }
        arrs = {f"cfg.{k}": np.array([float(v)]) for k, v in scalars.items()}
        arrs.update(self.encoders["text"].arrays("enc"))
        arrs.update(self.core.arrays())
        arrs.update(self.ssm.arrays())
        arrs.update(self.memory.arrays())
        arrs.update(self.decoder.arrays())
        arrs.update(self.fast.arrays())
        arrs["cb.w"], arrs["cb.p"] = self.cerebellum.w, self.cerebellum.p
        arrs.update(self.precision.arrays())
        arrs["expert_var"] = self.expert_var
        arrs["key_proj"] = self.key_proj
        arrs["target_index"] = self.target_index.astype(float)
        arrs["episodic_index"] = np.concatenate([self.episodic_out.astype(float), [-1.0]])
        arrs["trace.index"] = self.trace_index.astype(float)
        arrs["trace.ad"], arrs["trace.bd"], arrs["trace.m"] = ad, bd, self.trace.m
        cad, cbd = self.context.discretise(1.0)
        arrs["ctx.ad"], arrs["ctx.bd"], arrs["ctx.m"] = cad, cbd, self.context.m
        if self.episode is not None:
            ead, ebd = self.episode.discretise(1.0)
            arrs["ep3.ad"], arrs["ep3.bd"], arrs["ep3.m"] = ead, ebd, self.episode.m
        for k in self.LIVE:
            v = getattr(self, f"_{k}")
            if v is not None:
                arrs[f"live.{k}"] = v
        with open(path, "wb") as f:
            f.write(b"RUTHBRN1")
            f.write(struct.pack("<I", len(arrs)))
            for name, a in arrs.items():
                a = np.ascontiguousarray(a, dtype="<f8")
                nb = name.encode()
                f.write(struct.pack("<I", len(nb)) + nb)
                f.write(struct.pack("<I", a.size))
                f.write(a.tobytes())
        return path
