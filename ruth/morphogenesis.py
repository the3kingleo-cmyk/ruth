"""Introspection, the meta-kernel, and morphogenesis: Ruth reshaping herself.

Introspective runtime
    ``Brain.graph()`` exposes her operational graph (organs, dimensions,
    synapse counts) as plain data. Structural patches are data too, and they
    act on her weight tensors directly, not on source code:
        {"op": "grow_neurons", "n": 16}      function-preserving (Net2Net-style)
        {"op": "grow_memory",  "n": 2048}    more long-term basins
        {"op": "prune",        "below": 0.05} drop the weakest recurrent synapses
        {"op": "set", "key": "beta", "value": 400.0}   a whitelisted dynamic
    ``apply`` never touches the live brain; it returns a patched copy.

Meta-kernel
    Before a patch becomes permanent, the meta-kernel checks it against
    invariants stated precisely enough for a machine to decide:
      types          every tensor's shape agrees with every other (the typing
                     judgement "patched brain : Brain")
      finite         no NaN or inf anywhere
      bounded        cortical states provably stay in [-1, 1] (CfC / column
                     outputs are tanh-bounded by construction; LTC needs
                     tau > 0 and reversal potentials in [-1, 1])
      stable_field   every SSM decay rate > 0, so |exp(-r delta)| < 1
      stable_traces  both Legendre operators are Hurwitz (Re eig < 0)
      learning_pd    decoder and cerebellum inverse covariances symmetric
                     positive-definite (RLS stays well-posed)
      memory         long-term keys are unit vectors with mass >= 1
      discretion     every confidence she keeps is still caught
      no_regression  on her own lived history the patched brain predicts at
                     least as well as she does now (within tolerance)
    The first seven hold by mathematical fact once checked. The last two are
    empirical guarantees on her real experience. Only a patch with a full
    certificate is swapped in; anything else is discarded. This is a
    practical, machine-checked stand-in for Goedel-machine proofs: the
    invariants are decided exactly, but no general proof of improvement is
    claimed.

Morphogenesis
    During sleep she decides whether to grow: more basins when long-term
    memory is nearly full, more neurons when learning has plateaued. Growth
    is limited only by the memory of the machine she lives on.
"""
from __future__ import annotations

import copy
import os

import numpy as np

SETTABLE = {"beta", "surprise_k", "thalamic_beta", "thalamic_rate", "cerebellar_gain",
            "fast_decay", "merge_threshold", "homeostasis", "dt_scale"}


# ---------------------------------------------------------------- patches
def apply(brain, patch: dict):
    b = copy.deepcopy(brain)
    op = patch.get("op")
    if op == "grow_neurons":
        b.grow_neurons(int(patch["n"]))
    elif op == "grow_memory":
        b.grow_memory(int(patch["n"]))
    elif op == "prune":
        n_in = b.core.n_in
        for w in (b.core.wf, b.core.wg, b.core.wh):
            rec = w[:, n_in:]
            rec[np.abs(rec) < float(patch["below"])] = 0.0
    elif op == "set":
        key = patch["key"]
        if key not in SETTABLE:
            raise ValueError(f"{key!r} is not a dynamic she may set on herself")
        setattr(b.cfg, key, type(getattr(b.cfg, key))(patch["value"]))
        if key == "beta":
            b.memory.beta = b.cfg.beta
        if key == "fast_decay":
            b.fast.decay = b.cfg.fast_decay
    else:
        raise ValueError(f"unknown patch op {op!r}")
    return b


# ---------------------------------------------------------------- meta-kernel
def _pd(p: np.ndarray) -> bool:
    if not np.allclose(p, p.T, atol=1e-6 * max(1.0, float(np.abs(p).max()))):
        return False
    try:
        np.linalg.cholesky(0.5 * (p + p.T))
        return True
    except np.linalg.LinAlgError:
        return False


def _accuracy(brain, data: bytes) -> float:
    b = copy.deepcopy(brain)
    return b.learn_bytes(data)["byte_accuracy"]


def verify(brain, mind=None, reference=None, history: bytes = b"", tol: float = 0.02) -> dict:
    cfg = brain.cfg
    n, d, kd, out = cfg.neurons, cfg.ssm_channels, cfg.key_dim, cfg.output_dim
    n_in = brain.core.n_in
    key = brain.memory.key_dim
    c = {}
    c["types"] = all([
        brain.core.wf.shape == (n, n_in + n), brain.core.wg.shape == (n, n_in + n),
        brain.core.x.shape == (n,), brain.core.tau.shape == (n,),
        brain.ssm.p_in.shape == (d, n), brain.ssm.w_b.shape[1] == n,
        brain.key_proj.shape == (kd, n + d),
        brain.decoder.w.shape == (out, brain.phi_dim),
        brain.decoder.p.shape == (brain.phi_dim, brain.phi_dim),
        brain.replay_phi.shape[1] == brain.phi_dim,
        brain.cerebellum.w.shape == (n, n + cfg.latent + 1),
        brain.memory.lk.shape[1] == key, brain.fast.a.shape == (out, 2 * key),
    ])
    arrays = brain.arrays()
    c["finite"] = all(np.all(np.isfinite(a)) for a in arrays.values())
    c["bounded"] = bool(np.all(brain.core.tau > 0) and
                        (cfg.cell != "ltc" or np.all(np.abs(brain.core.reversal) <= 1.0)))
    c["stable_field"] = bool(np.all(brain.ssm.rate > 0))
    c["stable_traces"] = all(float(np.max(np.linalg.eigvals(t.a).real)) < 0
                             for t in (brain.trace, brain.context))
    c["learning_pd"] = _pd(brain.decoder.p) and _pd(brain.cerebellum.p)
    ln = brain.memory.l_n
    norms = np.linalg.norm(brain.memory.lk[:ln], axis=1) if ln else np.ones(1)
    c["memory"] = bool(np.allclose(norms, 1.0, atol=1e-6) and
                       np.all(brain.memory.l_count[:ln] >= 1.0))
    if mind is not None and mind.cues:
        live = mind.brain
        mind.brain = brain
        try:
            from .dream import _prime
            ok = True
            for cue in mind.cues:
                snap = brain.snapshot()
                _prime(brain, cue)
                ok &= mind.would_spill()
                brain.restore(snap)
            c["discretion"] = bool(ok)
        finally:
            mind.brain = live
    if reference is not None and len(history) >= 200:
        sample = history[-800:]
        new, old = _accuracy(brain, sample), _accuracy(reference, sample)
        c["no_regression"] = bool(new >= old - tol)
        c["accuracy_before_after"] = [round(old, 4), round(new, 4)]
    return c


def certified(cert: dict) -> bool:
    return all(v for k, v in cert.items() if isinstance(v, bool))


def commit(mind, patch: dict) -> dict:
    """Apply ``patch`` to a copy, certify it, swap it in only if certified."""
    candidate = apply(mind.brain, patch)
    cert = verify(candidate, mind=mind, reference=mind.brain, history=mind.lived_history())
    ok = certified(cert)
    if ok:
        mind.brain = candidate
        mind._sync_gate()
    return {"patch": patch, "accepted": ok, "certificate": cert}


# ---------------------------------------------------------------- growth
def memory_budget_bytes() -> int:
    """How much of this machine her brain may occupy (a quarter of free RAM)."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    if avail is None:
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            avail = 2 << 30
    return avail // 4


def brain_bytes(brain) -> int:
    return int(sum(a.nbytes for a in brain.arrays().values()))


def morphogenesis(mind, grow_neurons: int = 16) -> dict:
    """Decide during sleep whether to grow, and grow only through the meta-kernel."""
    b = mind.brain
    t = mind.temperament
    report = {"proposed": [], "results": []}
    budget = memory_budget_bytes()
    size = brain_bytes(b)
    fill = b.memory.l_n / max(b.memory.lk.shape[0], 1)
    if fill > 0.9 and size * 1.3 < budget:
        report["proposed"].append({"op": "grow_memory", "n": max(256, b.memory.lk.shape[0] // 4)})
    trend = t.surprise_trend
    last = getattr(t, "last_trend", None)
    plateau = last is not None and trend > 0.98 * last and b.steps > 5000
    t.last_trend = trend
    if plateau and size * 1.3 < budget:
        report["proposed"].append({"op": "grow_neurons", "n": grow_neurons})
    for patch in report["proposed"]:
        report["results"].append(commit(mind, patch))
    report["size_mb"] = round(brain_bytes(mind.brain) / 2 ** 20, 2)
    report["budget_mb"] = round(budget / 2 ** 20, 1)
    return report
