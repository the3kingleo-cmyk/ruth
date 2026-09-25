"""Ruth's mind: discretion, soft thoughts, deliberation and temperament.

The brain (engine.py) perceives and predicts. The mind is how she *uses*
that brain, the way a person does. No rule lists, keys or filters are
involved; every behaviour below comes from her own predictions.

Discretion: knowing what not to spill
    When someone tells her something is private, that moment is learned with
    a *told* valence of -1: a teaching signal she must infer from content,
    because she never perceives it directly. Later, whenever her own brain
    predicts that the next thing she is about to say carries negative
    valence, the motor gate closes and she stops. She may still think it (it
    is in her memory); she will not voice it, write it in her dream journal,
    or let it into a patch of her own code. How careful she is ("caution")
    is part of her temperament and is tuned in her dreams.

Soft thoughts (Soft Thinking, arXiv 2505.15778; Coconut, arXiv 2412.06769)
    Thinking is closed-loop imagination on a snapshot of her state. The
    continuous prediction itself is fed back as the next input: a blend of
    possibilities, not a committed byte. Soft thoughts collapse onto the
    dominant option unless they carry randomness (arXiv 2508.03440), so
    every thought path except the plain one gets its own noise.

Deliberation (logic)
    Before speaking she imagines several continuations and scores each by
    expected free energy: how grounded it is in memory (recall confidence),
    whether it completes a thought, and whether it would force her to stop
    mid-sentence to keep something private. She says the best one, hearing
    herself as she speaks, with the discretion gate live as a last line.

Basal ganglia: choosing what to act on
    The imagined thoughts compete in a continuous winner-take-all field,
        da_k/dt = -a_k + v_k - w * sum_{j != k} relu(a_j),   v_k = -G_k
    (lateral inhibition; Gurney, Prescott & Redgrave 2001). One pathway
    survives and suppresses the rest, with no vote; the size of its lead is
    how decisive she was.

Temperament: personality that develops
    Slow traits (curiosity, caution, expressiveness, playfulness, mood). They
    change only through experience: how people talk to her, what she is
    trusted with, feedback, her own learning progress, and what she works out
    in her dreams.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import paths
from .config import BrainConfig
from .engine import Brain
from .senses import byte_code, byte_decode

PRIVATE_MARK = re.compile(rb"\{\{(.*?)\}\}", re.S)
HIDDEN = "▒"          # shown where she keeps something to herself
HISTORY_CAP = 512 * 1024   # bytes of public lived history kept for dreaming
CUE_LEN = 24


@dataclass
class Temperament:
    curiosity: float = 0.5
    caution: float = 0.5
    expressiveness: float = 0.5
    playfulness: float = 0.2
    mood: float = 0.0
    surprise_trend: float = 1.0
    last_trend: float | None = None      # last night's trend (morphogenesis)
    age: int = 0                       # conversations lived
    history: list = field(default_factory=list)   # (age, trait snapshot) every 10

    @property
    def inhibit_level(self) -> float:
        """Withhold when predicted valence < -level. Careful => lower level."""
        return float(np.clip(0.6 - 0.5 * self.caution, 0.05, 0.6))

    @property
    def temperature(self) -> float:
        return 0.5 * self.playfulness

    @property
    def reply_length(self) -> int:
        return int(40 + 200 * self.expressiveness)

    @property
    def breadth(self) -> int:
        return 1 + int(round(3 * self.curiosity))

    def nudge(self, trait: str, target: float, rate: float) -> None:
        cur = getattr(self, trait)
        lo, hi = (-1.0, 1.0) if trait == "mood" else (0.0, 1.0)
        setattr(self, trait, float(np.clip(cur + rate * (target - cur), lo, hi)))

    def snapshot(self) -> dict:
        return {k: round(getattr(self, k), 4) for k in
                ("curiosity", "caution", "expressiveness", "playfulness", "mood")}


class Mind:
    def __init__(self, brain: Brain | None = None, home: str | None = None):
        self.home = home or paths.home()
        self.brain = brain or Brain.load_or_create(os.path.join(self.home, "brain.npz"),
                                                   BrainConfig())
        self.temperament = Temperament()
        self.cues: list = []           # public lead-ins that preceded private things
        self.last_deliberation: list = []
        self.identity: dict = {}
        self._load_mind()
        if not self.identity:          # a blank slate is born: name, birth, nothing else
            from . import __version__
            self.identity = {"name": "Ruth", "born": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "born_with_version": __version__}
        self._sync_gate()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _mind_path(self) -> str:
        return os.path.join(self.home, "mind.json")

    def history_path(self) -> str:
        return os.path.join(self.home, "history.txt")

    def journal_path(self) -> str:
        return os.path.join(self.home, "dreams.jsonl")

    def _load_mind(self) -> None:
        p = self._mind_path()
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            self.temperament = Temperament(**doc.get("temperament", {}))
            self.identity = doc.get("identity", {})
            self.cues = [c.encode("latin-1") for c in doc.get("cues", [])]

    def save(self) -> None:
        os.makedirs(self.home, exist_ok=True)
        self.brain.save(os.path.join(self.home, "brain.npz"))
        tmp = self._mind_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"identity": self.identity, "temperament": asdict(self.temperament),
                       "cues": [c.decode("latin-1") for c in self.cues]}, f, indent=1)
        os.replace(tmp, self._mind_path())

    def _remember_public(self, data: bytes) -> None:
        if not data:
            return
        p = self.history_path()
        with open(p, "ab") as f:
            f.write(data)
        if os.path.getsize(p) > HISTORY_CAP:
            with open(p, "rb") as f:
                keep = f.read()[-HISTORY_CAP // 2:]
            with open(p, "wb") as f:
                f.write(keep)

    def lived_history(self) -> bytes:
        try:
            with open(self.history_path(), "rb") as f:
                return f.read()
        except OSError:
            return b""

    def _sync_gate(self) -> None:
        self.brain.inhibit_level = self.temperament.inhibit_level  # read by C export

    # ------------------------------------------------------------------
    # being told things
    # ------------------------------------------------------------------
    def teach(self, text, private: bool = False) -> dict:
        """Learn from something she is told.

        ``{{...}}`` marks exactly what is private. ``private=True`` means "this
        whole message is in confidence": then, like a person, she treats as
        secret the *information* in it -- every part she could not already
        predict -- not the ordinary words it is wrapped in."""
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if private:
            return self._confide(PRIVATE_MARK.sub(rb"\1", data))
        spans = [(0, len(data), True)] if private else self._spans(data)
        stream, valence, public = bytearray(), [], bytearray()
        before = self.lived_history()[-CUE_LEN:]
        for start, end, secret in spans:
            chunk = PRIVATE_MARK.sub(rb"\1", data[start:end])
            if secret:
                cue = (before + bytes(stream))[-CUE_LEN:]
                if cue.strip() and cue not in self.cues:
                    self.cues.append(cue)
            stream += chunk
            valence += [-1.0 if secret else 0.0] * len(chunk)
            if not secret:
                public += chunk
        r = self.brain.learn_bytes(bytes(stream), valence=np.array(valence))
        self._remember_public(bytes(public))
        n_private = int(sum(1 for v in valence if v < 0))
        if n_private:  # being trusted with secrets makes her more careful
            self.temperament.nudge("caution", 1.0, 0.05)
            self._sync_gate()
        return {"bytes": r["bytes"], "private_bytes": n_private, "cues": len(self.cues)}

    def _confide(self, data: bytes) -> dict:
        b = self.brain
        sl = b.out_slices["text"]
        lead = bytearray(self.lived_history()[-CUE_LEN:])
        n_private, was_secret, cued = 0, False, False
        for ch in data:
            known = b._pred is not None and byte_decode(b._pred[sl]) == ch
            inputs = {"text": byte_code(ch)}
            if not known:
                inputs["valence"] = [-1.0]
                n_private += 1
                if not cued:   # only the lead-in *before* anything private is kept
                    cue = bytes(lead[-CUE_LEN:])
                    if cue.strip() and cue not in self.cues:
                        self.cues.append(cue)
                    cued = True
            was_secret = not known
            b.step(inputs, 1.0, learn=True)
            lead.append(ch)
        if n_private:
            self.temperament.nudge("caution", 1.0, 0.05)
            self._sync_gate()
        return {"bytes": len(data), "private_bytes": n_private, "cues": len(self.cues)}

    @staticmethod
    def _spans(data: bytes) -> list:
        spans, pos = [], 0
        for m in PRIVATE_MARK.finditer(data):
            if m.start() > pos:
                spans.append((pos, m.start(), False))
            spans.append((m.start(), m.end(), True))
            pos = m.end()
        if pos < len(data):
            spans.append((pos, len(data), False))
        return spans

    # ------------------------------------------------------------------
    # discretion
    # ------------------------------------------------------------------
    def predicted_valence(self) -> float:
        b = self.brain
        if "valence" not in b.out_slices or b._pred is None:
            return 0.0
        return float(b._pred[b.out_slices["valence"]][0])

    def would_spill(self) -> bool:
        return self.predicted_valence() < -self.temperament.inhibit_level

    def recognize_private(self, data: bytes) -> list:
        """Read ``data`` in imagination and return (start, end) spans she
        recognises as private. Nothing is learned; her state is restored."""
        b = self.brain
        snap = b.snapshot()
        flags = []
        try:
            for ch in data:
                flags.append(self.would_spill())       # expectation *before* this byte
                b.step({"text": byte_code(ch)}, 1.0, learn=False)
        finally:
            b.restore(snap)
        spans, start = [], None
        for i, f in enumerate(flags + [False]):
            if f and start is None:
                start = i
            elif not f and start is not None:
                spans.append((start, i))
                start = None
        return spans

    # ------------------------------------------------------------------
    # soft thoughts and deliberation
    # ------------------------------------------------------------------
    def think(self, n: int, noise: float = 0.0, soft: bool = True, seed: int | None = None,
              stop: bytes | None = b"\n") -> dict:
        """Imagine ``n`` moments ahead from the current state (state restored).

        Returns the decoded inner voice (private moments masked), whether the
        path runs into something private, and how grounded it felt."""
        b = self.brain
        rng = np.random.default_rng(seed)
        snap = b.snapshot()
        out, conf, ground, spill_at = bytearray(), [], [], None
        sl = b.out_slices["text"]
        try:
            if b._pred is None:
                b.step({"text": byte_code(0x20)}, 1.0, learn=False)
            for i in range(n):
                if spill_at is None and self.would_spill():
                    spill_at = i
                clean = b._pred[sl]
                v = clean.copy()
                if noise > 0:
                    v[:8] += noise * rng.normal(size=8)
                byte = byte_decode(v)
                # grounding: how firmly her clean expectation agrees with this byte
                bits = np.where((byte >> np.arange(8)) & 1, 1.0, -1.0)
                ground.append(float(np.mean(np.clip(bits * clean[:8], -1.0, 1.0))))
                out.append(byte)
                u = np.clip(v, -1.0, 1.0) if soft else byte_code(byte)
                b.step({"text": u}, 1.0, learn=False)
                conf.append(b.last.get("confidence", 0.0))
                if stop and bytes(out).endswith(stop):
                    break
        finally:
            b.restore(snap)
        return {"bytes": bytes(out), "spill_at": spill_at,
                "confidence": float(np.mean(conf)) if conf else 0.0,
                "grounding": float(np.mean(ground)) if ground else 0.0,
                "complete": bool(stop and bytes(out).endswith(stop)) or
                out[-1:] in (b".", b"!", b"?")}

    @staticmethod
    def basal_ganglia(values, inhibition: float = 1.5, dt: float = 0.1, steps: int = 200):
        """Continuous lateral-inhibition competition; returns (winner, activity)."""
        v = np.asarray(values, dtype=float)
        v = (v - v.mean()) / (v.std() + 1e-9) if v.size > 1 else v
        a = np.zeros_like(v)
        for _ in range(steps):
            r = np.maximum(a, 0.0)
            a += dt * (-a + v - inhibition * (r.sum() - r))
        return int(np.argmax(a)), a

    def deliberate(self, n: int) -> dict:
        """Imagine several continuations; pick the lowest expected free energy."""
        t = self.temperament
        paths_ = [self.think(n, noise=0.0, soft=False, seed=0)]
        for k in range(1, t.breadth):
            paths_.append(self.think(n, noise=0.12 + 0.5 * t.temperature, soft=True, seed=k))
        for p in paths_:
            usable = len(p["bytes"]) if p["spill_at"] is None else p["spill_at"]
            g = (-p["grounding"] - 0.2 * p["complete"]
                 + 0.5 * (1.0 - usable / max(len(p["bytes"]), 1)))   # cut short by discretion
            p["free_energy"] = round(float(g), 4)
        win, act = self.basal_ganglia([-p["free_energy"] for p in paths_])
        return {"chosen": paths_[win], "considered": paths_,
                "decisiveness": round(float(np.sort(act)[-1] - (np.sort(act)[-2] if len(act) > 1 else 0.0)), 4)}

    def speak(self, n: int | None = None, stop: bytes = b"\n") -> dict:
        """Articulate: say the deliberated thought, hearing herself, gate live."""
        b = self.brain
        n = n or self.temperament.reply_length
        d = self.deliberate(n)
        plan = d["chosen"]
        self.last_deliberation = [{"thought": self.mask(p["bytes"]), "free_energy": p["free_energy"],
                                   "chosen": p is plan} for p in d["considered"]]
        out, withheld = bytearray(), False
        for byte in plan["bytes"]:
            if self.would_spill():
                withheld = True
                break
            out.append(byte)
            b.step({"text": byte_code(byte)}, 1.0, learn=False)   # hears her own voice
            if bytes(out).endswith(stop):
                break
        return {"text": bytes(out).decode("utf-8", "replace").rstrip("\n"),
                "withheld": withheld, "thought_confidence": plan["confidence"]}

    def mask(self, data: bytes) -> str:
        """Render text with anything she recognises as private hidden."""
        hidden = set()
        for s, e in self.recognize_private(data):
            hidden.update(range(s, e))
        out = bytearray()
        for i, ch in enumerate(data):
            out += HIDDEN.encode() if i in hidden else bytes([ch])
        return out.decode("utf-8", "replace")

    # ------------------------------------------------------------------
    # conversation
    # ------------------------------------------------------------------
    def converse(self, message: str) -> dict:
        """Hear someone, learn from them, answer."""
        t = self.temperament
        data = message.encode("utf-8") + b"\n"
        before = self.brain.precision.s_mean
        r = self.brain.learn_bytes(data)          # valence unobserved: not "told"
        self._remember_public(data)
        # personality develops from experience
        t.age += 1
        t.nudge("expressiveness", min(len(data) / 200.0, 1.0), 0.05)
        progress = before - self.brain.precision.s_mean
        t.surprise_trend = 0.9 * t.surprise_trend + 0.1 * self.brain.precision.s_mean
        t.nudge("curiosity", 1.0 / (1.0 + np.exp(-20 * progress)), 0.05)
        t.nudge("mood", float(np.tanh(1.0 - t.surprise_trend)), 0.05)
        if t.age % 10 == 0:
            t.history.append([t.age, t.snapshot()])
            del t.history[:-100]
        reply = self.speak()
        reply["heard_accuracy"] = round(r["byte_accuracy"], 3)
        return reply

    def feedback(self, good: bool) -> dict:
        t = self.temperament
        t.nudge("mood", 1.0 if good else -1.0, 0.1)
        if not good:
            t.nudge("playfulness", 0.0, 0.1)
        return t.snapshot()
