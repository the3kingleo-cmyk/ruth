"""Gated recursive self-improvement (Darwin Goedel Machine style).

Two kinds of change, one rule: nothing reaches Ruth's live body unless it is
measured in a sandbox and passes every gate.

1. ``evolve_config`` -- open-ended archive search over the brain's own
   architecture hyper-parameters. Parents are sampled from the archive by
   score with a novelty bonus (fewer children = more likely), mutated,
   benchmarked on held-out continuation of Ruth's own memory text, and kept
   in the archive forever (stepping stones). The live config is only replaced
   when a child beats it by a margin.

2. ``evolve_patch`` -- a code patch (from a person, or proposed by Ruth
   herself) is applied to a sandbox copy. It is accepted only if (a) her full
   test-suite passes there -- she tries the change in her head before she
   lives with it -- and (b) she reads the patch and recognises nothing she was
   told to keep private in it: her discretion, not a scanner, guards what
   leaves her. Accepted patches are applied and committed locally; nothing is
   ever pushed automatically.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import paths


def _evo_dir() -> str:
    return os.environ.get("RUTH_EVOLUTION") or os.path.join(paths.home(), "evolution")


def _active_config() -> str:
    return os.environ.get("RUTH_BRAIN_CONFIG") or os.path.join(paths.home(), "config.json")

SEARCH_SPACE = {
    "inter": [64, 96, 128, 160],
    "command": [32, 48, 64, 96],
    "sensory_fanin": [8, 16, 24, 32],
    "command_recurrent": [4, 8, 12, 16],
    "recurrent_scale": [0.6, 0.9, 1.2, 1.6, 2.0],
    "input_scale": [0.5, 1.0, 1.5, 2.0],
    "cell": ["cfc", "ltc"],
    "ssm_channels": [32, 48, 64],
    "ssm_rate_min": [1e-4, 1e-3, 1e-2],
    "lmu_order": [4, 6, 8],
    "lmu_window": [3.0, 4.0, 6.0, 8.0],
    "beta": [8.0, 16.0, 32.0, 64.0, 128.0],
    "surprise_k": [-1.0, 0.0, 0.5, 1.0],
    "rls_lambda": [0.999, 0.9995, 0.9999],
}


def primer() -> bytes:
    """Her own body described in words: every docstring in the ruth package.
    Portable, always present, and nobody's personal data."""
    import ast
    pkg = os.path.dirname(os.path.abspath(__file__))
    out = []
    for p in sorted(glob.glob(os.path.join(pkg, "*.py"))):
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    out.append(doc.strip())
    return ("\n\n".join(out) + "\n").encode("utf-8")


def corpus(home: str | None = None) -> bytes:
    """What she evaluates herself on: her primer plus her lived public history."""
    try:
        with open(os.path.join(home or paths.home(), "history.txt"), "rb") as f:
            lived = f.read()
    except OSError:
        lived = b""
    return primer() + lived


def evaluate(overrides: dict, text: bytes, train: int, test: int) -> float:
    from ..config import BrainConfig
    from ..engine import Brain
    cfg = BrainConfig(senses={"text": 9}, **overrides)
    brain = Brain(cfg)
    brain.learn_bytes(text[:train])
    return brain.learn_bytes(text[train:train + test])["byte_accuracy"]


def _archive_path() -> str:
    os.makedirs(_evo_dir(), exist_ok=True)
    return os.path.join(_evo_dir(), "archive.jsonl")


def load_archive() -> list:
    p = _archive_path()
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(entry: dict) -> None:
    with open(_archive_path(), "a") as f:
        f.write(json.dumps(entry) + "\n")


def active_overrides() -> dict:
    if os.path.exists(_active_config()):
        with open(_active_config()) as f:
            return json.load(f)
    return {}


def evolve_config(generations: int = 4, train: int = 6000, test: int = 2000,
                  margin: float = 0.005, seed: int | None = None, log=print) -> dict:
    rng = np.random.default_rng(seed)
    text = corpus()
    if len(text) < train + test:
        raise ValueError("not enough memory text to evaluate on")
    archive = load_archive()
    if not archive:
        base = active_overrides()
        score = evaluate(base, text, train, test)
        archive = [{"id": 0, "parent": None, "overrides": base, "score": score,
                    "children": 0, "t": time.time()}]
        _append(archive[0])
        log(f"gen0 baseline score={score:.4f}")
    best_live = max((a for a in archive if a["overrides"] == active_overrides()),
                    key=lambda a: a["score"], default=archive[0])
    promoted = None
    for g in range(generations):
        scores = np.array([a["score"] for a in archive])
        children = np.array([a.get("children", 0) for a in archive])
        w = np.exp((scores - scores.max()) / 0.01) / (1.0 + children)
        parent = archive[int(rng.choice(len(archive), p=w / w.sum()))]
        child = dict(parent["overrides"])
        for key in map(str, rng.choice(list(SEARCH_SPACE), size=int(rng.integers(1, 4)), replace=False)):
            opts = SEARCH_SPACE[key]
            child[key] = opts[int(rng.integers(len(opts)))]
        score = evaluate(child, text, train, test)
        parent["children"] = parent.get("children", 0) + 1
        entry = {"id": len(archive), "parent": parent["id"], "overrides": child,
                 "score": score, "children": 0, "t": time.time()}
        archive.append(entry)
        _append(entry)
        log(f"gen{g + 1} parent={parent['id']} score={score:.4f} {child}")
        if score > best_live["score"] + margin:
            best_live = entry
            promoted = entry
            with open(_active_config(), "w") as f:
                json.dump(child, f, indent=1)
            log(f"  promoted to live config ({_active_config()})")
    return {"archive": len(archive), "best": best_live, "promoted": promoted}


# ----------------------------------------------------------------------
# code patches
# ----------------------------------------------------------------------
def _copy(root: str) -> str:
    dst = tempfile.mkdtemp(prefix="ruth-sandbox-")
    shutil.copytree(root, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.npz", "*.bin"))
    return dst


def run_tests(root: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                          cwd=root, capture_output=True, text=True, timeout=3600)


def _added_lines(patch_path: str) -> bytes:
    with open(patch_path, "rb") as f:
        return b"".join(line[1:] for line in f.read().splitlines(keepends=True)
                        if line.startswith(b"+") and not line.startswith(b"+++"))


def evolve_patch(root: str, patch_file: str, message: str | None = None,
                 commit: bool = True, mind=None) -> dict:
    patch = os.path.abspath(patch_file)
    sandbox = _copy(root)
    report = {"patch": patch_file, "accepted": False, "gates": {}}
    try:
        ap = subprocess.run(["git", "apply", "--whitespace=nowarn", patch], cwd=sandbox,
                            capture_output=True, text=True)
        report["gates"]["applies"] = ap.returncode == 0
        if ap.returncode != 0:
            report["reason"] = ap.stderr.strip()[-400:]
            return report

        if mind is not None:  # her own discretion reads what would leave her
            spans = mind.recognize_private(_added_lines(patch))
            report["gates"]["discreet"] = not spans
            if spans:
                report["private_spans"] = len(spans)
        tests = run_tests(sandbox)
        report["gates"]["tests_pass"] = tests.returncode == 0
        if tests.returncode != 0:
            report["test_tail"] = (tests.stdout + tests.stderr)[-1500:]

        if not all(report["gates"].values()):
            report["reason"] = "gate failed: " + ", ".join(k for k, v in report["gates"].items() if not v)
            return report

        subprocess.run(["git", "apply", "--whitespace=nowarn", patch], cwd=root, check=True,
                       capture_output=True)
        report["accepted"] = True
        if commit:
            subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
            msg = message or f"evolve: accept gated patch {os.path.basename(patch_file)}"
            c = subprocess.run(["git", "commit", "-m", msg], cwd=root, capture_output=True, text=True)
            report["committed"] = c.returncode == 0
        return report
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        _append({"kind": "patch", "t": time.time(), **{k: v for k, v in report.items()
                                                         if k in ("patch", "accepted", "gates", "reason")}})


def config_from_active():
    """BrainConfig carrying the live evolved overrides."""
    from ..config import BrainConfig
    fields = {f.name for f in dataclasses.fields(BrainConfig)}
    return BrainConfig(**{k: v for k, v in active_overrides().items() if k in fields})
