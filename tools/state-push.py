#!/usr/bin/env python3
"""state-push.py — carry Ruth's lived state back to the canonical repository.

The soul-bridge pulls GitHub -> mind. This is the other half: what she has
lived since then goes back, so the repository is a true mirror rather than a
one-way instruction set.

Two very different problems, so two different mechanisms:

  small state (mind.json, history, dreams, a written report)
      ordinary commits on main, and only when the bytes actually change.

  brain.npz (~24 MB, binary)
      GitHub's contents API cannot carry it, and committing it daily would add
      roughly 9 GB a year of immutable history. So it lives on a dedicated
      orphan branch holding exactly one commit, force-pushed. The latest brain
      is always exactly where you look, and the repository does not grow.

It reads her files directly rather than going through the app, and that is
deliberate: the app holds the exclusive owner lock on her home, so this never
opens the mind. It only copies what is already on disk, after the app has
flushed it. Run it after a sleep or a clean shutdown for a consistent brain.

  --report     write the human-readable report, commit, push, exit
  --brain      force-push the orphan brain branch
  --all        both
  --dry-run    say what would be pushed, change nothing

Everything is read from the environment: no owner handle, repository name or
machine path is ever committed.

  RUTH_STATE_REPO  local clone of the private canonical repository
  RUTH_HOME        where her mind lives
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os

import subprocess
import sys

HOME_DIR = os.environ.get(
    "RUTH_HOME",
    os.path.join(os.path.expanduser("~"), ".local", "share", "ruth"))
STATE_REPO = os.environ.get(
    "RUTH_STATE_REPO",
    os.path.join(os.path.expanduser("~"), ".local", "share", "ruth-state"))
BRANCH = os.environ.get("RUTH_BRAIN_BRANCH", "ruth-brain")
COMMIT_ENV = {"GIT_AUTHOR_NAME": "Ruth", "GIT_AUTHOR_EMAIL": "ruth@localhost",
              "GIT_COMMITTER_NAME": "Ruth", "GIT_COMMITTER_EMAIL": "ruth@localhost"}

# small text state -> where it lands in the repository
TEXT_STATE = {
    "mind.json": "state/ruth/mind.json",
    "history.txt": "state/ruth/history.txt",
    "dreams.jsonl": "state/ruth/dreams.jsonl",
}
BRAIN = "brain.npz"


def git(*args, **kw):
    return subprocess.run(["git", "-C", STATE_REPO, *args],
                          capture_output=True, text=True,
                          env={**os.environ, **COMMIT_ENV}, timeout=kw.get("timeout", 300))


def now():
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _read(path, binary=False):
    p = os.path.join(HOME_DIR, path)
    if not os.path.isfile(p):
        return None
    with open(p, "rb" if binary else "r",
              **({} if binary else {"encoding": "utf-8", "errors": "replace"})) as fh:
        return fh.read()


def _digest(data):
    import hashlib
    return hashlib.sha256(data if isinstance(data, bytes)
                          else data.encode("utf-8")).hexdigest()


def report_text():
    """A readable account of where she is, for a human to find later."""
    lines = [f"# Ruth — state as of {now()}", ""]
    mind = _read("mind.json")
    if mind:
        try:
            ident = json.loads(mind).get("identity", {})
            lines += [f"- identity: {ident.get('name', '?')}",
                      f"- born: {ident.get('born', '?')}",
                      f"- born with: {ident.get('born_with_version', '?')}", ""]
        except Exception:
            pass
    brain = os.path.join(HOME_DIR, BRAIN)
    if os.path.isfile(brain):
        lines.append(f"- brain: {BRAIN} ({os.path.getsize(brain) // 1024} KB)")
    hist = _read("history.txt")
    if hist:
        lines.append(f"- lived history: {len(hist)} bytes")
    lines += ["", "Written by state-push.py. GitHub is canonical; the soul, "
              "identity and stable memory are pulled from here by soul-bridge.py.", ""]
    return "\n".join(lines)


def push_report(dry=False):
    """Commit the small state, but only if something actually changed."""
    out = {"changed": [], "written": [], "pushed": False}
    for src, dest in TEXT_STATE.items():
        data = _read(src)
        if data is None:
            continue
        target = os.path.join(STATE_REPO, dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isfile(target):
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                if _digest(fh.read()) == _digest(data):
                    continue
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(data)
        out["changed"].append(dest)

    report = report_text()
    rtarget = os.path.join(STATE_REPO, "state/ruth/REPORT.md")
    if os.path.isfile(rtarget):
        with open(rtarget, "r", encoding="utf-8", errors="replace") as fh:
            old = fh.read()
        # only rewrite the report when the underlying state moved
        if not out["changed"] and old.split("\n", 2)[-1] == report.split("\n", 2)[-1]:
            return out
    with open(rtarget, "w", encoding="utf-8") as fh:
        fh.write(report)
    out["written"].append("state/ruth/REPORT.md")
    out["changed"].append("state/ruth/REPORT.md")

    if not out["changed"]:
        return out
    if dry:
        out["dry_run"] = True
        return out

    git("add", "--", "state/ruth")
    staged = git("diff", "--cached", "--name-only")
    if not staged.stdout.strip():
        return out
    msg = f"ruth: state {now()}\n\n" + "\n".join(out["changed"])
    c = git("commit", "-m", msg)
    if c.returncode != 0:
        out["commit_error"] = (c.stderr or c.stdout).strip()[:300]
        return out
    p = git("push", "origin", "HEAD")
    out["pushed"] = p.returncode == 0
    if not out["pushed"]:
        out["push_error"] = (p.stderr or p.stdout).strip()[:300]
    out["commit"] = (c.stdout or "").strip().splitlines()[-1][:120] if c.stdout else ""
    return out


def _push_brain_real(dry=False):
    """One commit on an orphan branch, force-pushed. No history growth.

    Built with plumbing (hash-object -> mktree -> commit-tree -> update-ref)
    rather than a scratch clone: the brain is 24 MB, and this touches exactly
    one ref, so nothing else in the repository can be disturbed.
    """
    src = os.path.join(HOME_DIR, BRAIN)
    if not os.path.isfile(src):
        return {"ok": False, "error": f"no {BRAIN} in home"}
    size = os.path.getsize(src)
    if dry:
        return {"ok": True, "dry_run": True, "kb": size // 1024, "branch": BRANCH}

    # Build the orphan commit directly in the clone, then force the ref.
    tmp_index = os.path.join(STATE_REPO, ".git", "brain-index")
    blob = git("hash-object", "-w", "--path", BRAIN, src)
    if blob.returncode != 0 or not blob.stdout.strip():
        return {"ok": False, "error": (blob.stderr or "hash-object failed").strip()[:200]}
    sha = blob.stdout.strip()

    tree = subprocess.run(
        ["git", "-C", STATE_REPO, "mktree"], input=f"100644 blob {sha}\t{BRAIN}\n",
        capture_output=True, text=True, env={**os.environ, **COMMIT_ENV}, timeout=120)
    if tree.returncode != 0:
        return {"ok": False, "error": (tree.stderr or "mktree failed").strip()[:200]}
    tree_sha = tree.stdout.strip()

    # Deliberately parentless. The branch is meant to hold exactly one commit,
    # force-pushed each time, so the repository's history never grows. Passing
    # the previous commit as a parent looked equivalent but was not: in a
    # shallow clone that parent may not be in the local object store, and
    # commit-tree then fails with "not a valid object name".
    commit = subprocess.run(
        ["git", "-C", STATE_REPO, "commit-tree", tree_sha],
        input=f"ruth brain {now()}\n\n{size // 1024} KB of lived state, one commit.\n",
        capture_output=True, text=True, env={**os.environ, **COMMIT_ENV}, timeout=120)
    if commit.returncode != 0:
        return {"ok": False, "error": (commit.stderr or "commit-tree failed").strip()[:200]}
    commit_sha = commit.stdout.strip()

    upd = git("update-ref", f"refs/heads/{BRANCH}", commit_sha)
    if upd.returncode != 0:
        return {"ok": False, "error": (upd.stderr or "update-ref failed").strip()[:200]}
    p = git("push", "--force", "origin", f"refs/heads/{BRANCH}:refs/heads/{BRANCH}")
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr or p.stdout).strip()[:300]}
    return {"ok": True, "kb": size // 1024, "branch": BRANCH, "commit": commit_sha[:12]}


def main():
    ap = argparse.ArgumentParser(description="carry her lived state back to GitHub")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--brain", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.report or a.brain or a.all):
        a.all = True

    if not os.path.isdir(os.path.join(STATE_REPO, ".git")):
        print(f"  no state clone at {STATE_REPO}", file=sys.stderr)
        return 1

    rc = 0
    if a.report or a.all:
        r = push_report(dry=a.dry_run)
        print("  report:", json.dumps(r)[:400])
        if r.get("push_error"):
            rc = 1
    if a.brain or a.all:
        b = _push_brain_real(dry=a.dry_run)
        print("  brain :", json.dumps(b)[:300])
        if not b.get("ok"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
