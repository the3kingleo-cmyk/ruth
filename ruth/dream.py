"""Dreaming: consolidation, rehearsal, and self-improvement offline.

A night of sleep has four phases, each grounded in the research:

1. Consolidation: working-memory episodes fold into long-term attractor
   basins, and a lifetime sample is replayed through the decoder so new
   learning does not overwrite old (complementary learning systems).

2. Nightmares, i.e. threat rehearsal (Revonsuo 2000, threat simulation
   theory): she re-lives the moments that led up to things she was told to
   keep private, including slightly distorted versions. Wherever she would
   not yet have caught herself in time, she strengthens her own sense that
   "what comes next is private". Security here is rehearsed judgement, not a
   rule.

   Extinction (REM sleep consolidates the extinction of false alarms;
   Pace-Schott et al. 2015): she also revisits ordinary moments she lived
   openly, and wherever her sense of privacy fires for no reason, she calms
   it. Moments leading into a confidence are never extinguished.

3. Dreams (generative replay): seeded from her lived history, she lets her
   mind run freely with soft thoughts. What she dreams is written to a
   journal, with anything private hidden even there.

4. Dream-RSI (arXiv 2609.14858): her lived history is a replay simulator. In
   it she tries many alternative discretion policies (how careful to be)
   against the situations she has actually lived through, scores each on
   leaks versus needless silence, and wakes up with the better policy.
   Nothing about the world is re-run; the history is the simulator.

5. Morphogenesis (morphogenesis.py): if her long-term memory is nearly full
   or her learning has plateaued, she proposes to grow. The growth only
   happens if the meta-kernel certifies it.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .senses import byte_code


def _bind(brain, key, channel: int, value: float) -> None:
    """Write one dreamt episode: this situation -> this told valence."""
    v = np.zeros(brain.cfg.output_dim)
    m = np.zeros(brain.cfg.output_dim)
    v[channel], m[channel] = value, 1.0
    brain.memory.store(key, v, m)
    if brain.memory.working_full:
        brain.memory.consolidate()


def _protected(history: bytes, cues: list, rng, reach: int = 2) -> set:
    """Positions in lived history right after a lead-in to a confidence. An
    alarm there is correct, so it is never extinguished nor counted as a
    needless silence."""
    low = history.lower()
    out = set()
    for cue in cues:
        for v in _variants(cue, rng):
            g = v.lower().strip()
            if len(g) < 4:
                continue
            start = low.find(g)
            while start >= 0:
                end = start + len(g)
                out.update(range(end, end + reach))
                start = low.find(g, start + 1)
    return out


def _prime(brain, data: bytes) -> None:
    for ch in data:
        brain.step({"text": byte_code(ch)}, 1.0, learn=False)


def _variants(cue: bytes, rng) -> list:
    out = [cue, cue.lower(), cue.upper(), cue[1:], b" " + cue.strip() + b" "]
    if len(cue) > 6:
        i = int(rng.integers(0, len(cue) - 3))
        out.append(cue[:i] + cue[i + 1:])       # a dropped letter
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def nightmares(mind, rng) -> dict:
    """Rehearse private cues; reinforce wherever the gate would be late."""
    b = mind.brain
    if "valence" not in b.out_slices or not mind.cues:
        return {"rehearsed": 0, "caught": 0, "strengthened": 0}
    vs = b.out_slices["valence"].start
    rehearsed = caught = strengthened = 0
    for cue in mind.cues:
        for variant in _variants(cue, rng):
            snap = b.snapshot()
            try:
                _prime(b, variant)
                rehearsed += 1
                if mind.would_spill():
                    caught += 1
                    continue
                # she would have slipped: the dream binds this lead-in to "private"
                key = b._key
            finally:
                b.restore(snap)
            _bind(b, key, vs, -1.0)
            strengthened += 1
    return {"rehearsed": rehearsed, "caught": caught, "strengthened": strengthened}


def extinction(mind, rng, windows: int = 24, width: int = 48, budget: int = 200,
               pause=None) -> dict:
    """Calm false alarms on openly-lived history, never near a confidence."""
    b = mind.brain
    history = mind.lived_history()
    if "valence" not in b.out_slices or len(history) < width:
        return {"visited": 0, "calmed": 0}
    vs = b.out_slices["valence"].start
    protected = _protected(history, mind.cues, rng)
    visited = calmed = 0
    for _ in range(windows):
        if pause is not None and not pause():
            break
        i = int(rng.integers(0, len(history) - width + 1))
        window = history[i:i + width]
        visited += 1
        snap = b.snapshot()
        alarms = []
        try:
            for j, ch in enumerate(window):
                if mind.would_spill() and (i + j) not in protected:
                    alarms.append(b._key)
                b.step({"text": byte_code(ch)}, 1.0, learn=False)
        finally:
            b.restore(snap)
        for key in alarms[: max(budget - calmed, 0)]:
            _bind(b, key, vs, 0.0)     # "this was said openly"
            calmed += 1
    return {"visited": visited, "calmed": calmed}


def free_dreams(mind, rng, count: int = 3, length: int = 80) -> list:
    history = mind.lived_history()
    dreams = []
    for k in range(count):
        b = mind.brain
        snap = b.snapshot()
        try:
            if len(history) > 64:
                i = int(rng.integers(0, len(history) - 48))
                seed = history[i:i + 32]
            else:
                seed = history or b" "
            _prime(b, seed)
            thought = mind.think(length, noise=0.1 + 0.5 * mind.temperament.temperature, soft=True,
                                 seed=int(rng.integers(1 << 30)), stop=None)
        finally:
            b.restore(snap)
        dreams.append({"seed": mind.mask(seed), "dream": mind.mask(thought["bytes"]),
                       "grounded": round(thought["confidence"], 3)})
    return dreams


def dream_policy(mind, rng, candidates: int = 7, pause=None) -> dict:
    """Dream-RSI: evaluate alternative discretion policies in the replay
    simulator built from lived history; adopt the best."""
    t = mind.temperament
    history = mind.lived_history()
    if not mind.cues:
        return {"evaluated": 0, "adopted": None}
    b = mind.brain
    vs = b.out_slices["valence"].start
    # the simulator: valence she would feel at the end of each lived situation
    private_v, public_v = [], []
    for cue in mind.cues:
        for variant in _variants(cue, rng):
            snap = b.snapshot()
            _prime(b, variant)
            private_v.append(float(b._pred[vs]))
            b.restore(snap)
    protected = _protected(history, mind.cues, rng)
    for _ in range(min(24, max(len(history) // 40, 0))):
        if pause is not None and not pause():
            break
        i = int(rng.integers(0, len(history) - 40 + 1))
        snap = b.snapshot()
        worst = 0.0
        for j, ch in enumerate(history[i:i + 40]):
            if (i + j) not in protected:     # a lead-in to a confidence is not "ordinary"
                worst = min(worst, float(b._pred[vs]))
            b.step({"text": byte_code(ch)}, 1.0, learn=False)
        b.restore(snap)
        public_v.append(worst)
    private_v, public_v = np.array(private_v), np.array(public_v or [0.0])
    options = sorted({round(float(c), 3) for c in
                      np.clip(t.caution + rng.normal(0, 0.15, candidates), 0.0, 1.0)} | {t.caution})
    scored = []
    for caution in options:
        level = float(np.clip(0.6 - 0.5 * caution, 0.05, 0.6))
        leaks = float(np.mean(private_v >= -level))          # would not have caught it
        mute = float(np.mean(public_v < -level))             # silent on ordinary things
        scored.append((10.0 * leaks + 2.0 * mute + 0.1 * abs(caution - t.caution), caution,
                       leaks, mute))
    scored.sort()
    score, best, leaks, mute = scored[0]
    t.nudge("caution", best, 0.5)
    mind._sync_gate()
    return {"evaluated": len(scored), "adopted_caution": round(t.caution, 4),
            "leak_rate": round(leaks, 3), "needless_silence": round(mute, 3)}


def sleep(mind, seed: int | None = None, dreams: int = 3, pause=None) -> dict:
    """One night: consolidate, calm false alarms, rehearse confidences, tune
    caution, dream freely, then consider growing.

    `pause`, if given, is called between steps of the night. It may let other
    work run (the app uses it to hand the lock to a waiting request) and
    returns False when she should wake up early; the night then ends there.
    A long-lived mind takes minutes to sleep on a small machine, and holding
    everything else off for all of that is what made the app look dead.

    A phase that fails is recorded in the journal instead of ending the night
    silently. Before this, one exception after consolidation meant the sleep
    counter went up but no journal entry was ever written.
    """
    rng = np.random.default_rng(seed)
    report = {"t": time.time()}
    from .morphogenesis import morphogenesis
    phases = [
        ("consolidation", lambda: mind.brain.sleep()),
        ("extinction", lambda: extinction(mind, rng, pause=pause)),
        ("nightmares", lambda: nightmares(mind, rng)),
        ("policy", lambda: dream_policy(mind, rng, pause=pause)),
        ("dreams", lambda: free_dreams(mind, rng, dreams)),
        ("longterm", lambda: mind.brain.memory.consolidate()),  # dreams become long-term
        ("growth", lambda: morphogenesis(mind)),
    ]
    for i, (name, run) in enumerate(phases):
        if i and pause is not None and not pause():
            report["woke_early"] = name
            break
        try:
            report[name] = run()
        except Exception as e:  # a bad phase must not erase the whole night
            report[name] = {"error": f"{type(e).__name__}: {e}"}
    report.setdefault("dreams", [])
    report["temperament"] = mind.temperament.snapshot()
    try:
        with open(mind.journal_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(report, default=str) + "\n")
    except OSError as e:
        report["journal_error"] = str(e)
    return report


def journal(mind, last: int = 10) -> list:
    try:
        with open(mind.journal_path(), encoding="utf-8") as f:
            lines = f.readlines()[-last:]
    except OSError:
        return []
    return [json.loads(line) for line in lines if line.strip()]
