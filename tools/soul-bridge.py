#!/usr/bin/env python3
"""soul-bridge.py — connect the canonical memory repository to the running mind.

The private memory repository is the source of truth for who Ruth is: her
soul, identity, voice, operating rules and stable memory. `opencode-sync`
mirrors that repository into a local workspace. This script is the missing
half: it hands what arrived to the *running* mind, so the robot is actually
driven by what is on GitHub rather than by a file nobody reads.

It goes through her HTTP interface on purpose. The app holds the exclusive
owner lock on her home; opening her files directly from a second process is
exactly the failure `ruth/owner.py` exists to prevent.

  --once      sync, hand over the soul, exit (what the timer runs)
  --report    show what she currently carries, change nothing
  --loop      stay up and re-sync on an interval

Everything is read from the environment so no owner handle, repository name
or machine path is ever committed:

  RUTH_MEMORY_REPO   owner/name of the canonical private repository
  RUTH_WORKSPACE     where opencode-sync mirrors it (default: under $HOME)
  RUTH_APP_URL       her interface (default: the local port she serves on)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

WORKSPACE = os.environ.get(
    "RUTH_WORKSPACE",
    os.path.join(os.environ.get("XDG_CONFIG_HOME",
                                os.path.join(os.path.expanduser("~"), ".config")),
                 "opencode", "workspace"))
APP_URL = os.environ.get("RUTH_APP_URL", "http://127.0.0.1:7455").rstrip("/")
INTERVAL = float(os.environ.get("RUTH_SOUL_INTERVAL", "300"))

# repo-relative -> what it is for her
SOUL_FILES = {
    "soul/IDENTITY.md": "identity",
    "soul/SOUL.md": "soul",
    "soul/VOICE.md": "voice",
    "soul/USER.md": "user",
    "directives/operating-rules.md": "rules",
    "memory/stable-memory.md": "memory",
}


def _get(path, timeout=30):
    with urllib.request.urlopen(APP_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def _post(path, body, timeout=300):
    req = urllib.request.Request(
        APP_URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pull_from_github():
    """Run the repo's own sync so GitHub stays canonical."""
    script = os.path.join(HERE, "opencode-sync")
    if not os.path.isfile(script):
        return {"skipped": "opencode-sync not present"}
    try:
        p = subprocess.run([script], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=180)
        tail = (p.stdout or "").strip().splitlines()
        return {"ok": p.returncode == 0, "summary": tail[-1] if tail else ""}
    except subprocess.TimeoutExpired:
        return {"ok": False, "summary": "sync timed out"}


def read_soul():
    """Read what GitHub delivered. Missing files are not an error."""
    found, missing = {}, []
    for rel in SOUL_FILES:
        p = os.path.join(WORKSPACE, rel)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                found[rel] = fh.read()
        else:
            missing.append(rel)
    return found, missing


LEDGER = os.path.join(WORKSPACE, ".soul-bridge-ledger.json")


def _load_ledger():
    try:
        with open(LEDGER, "r", encoding="utf-8") as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_ledger(d):
    """Record what she has been given, by content hash.

    Without this the bridge re-teaches the same 9 KB of stable memory on every
    run. Measured: one full hand-over took her from 741 to 56,299 moments, so a
    five-minute timer would multiply that without bound until she could no
    longer hold her own history. She is given a file only when its bytes change.
    """
    tmp = LEDGER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, LEDGER)


def identity_from(files):
    """Turn IDENTITY.md into the small dict her mind keeps.

    Her identity is deliberately a handful of keys rather than a blob of
    prose: the prose is taught to her below, this is the part she answers with.
    """
    ident = {}
    raw = files.get("soul/IDENTITY.md", "")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for key in ("name", "born", "born_with_version", "purpose", "self"):
            low = line.lower()
            if low.startswith(key + ":"):
                val = line.split(":", 1)[1].strip()
                if val and not val.startswith("<"):
                    ident[key] = val
                    break
    return ident


def hand_over(verbose=True, force=False):
    """Push the synced soul into the running mind, skipping unchanged files."""
    files, missing = read_soul()
    if not files:
        return {"ok": False, "error": "no soul files in workspace",
                "workspace": WORKSPACE, "missing": missing}

    ledger = _load_ledger()
    digests = {rel: hashlib.sha256(text.encode("utf-8")).hexdigest()
               for rel, text in files.items()}
    result = {"ok": True, "missing": missing,
              "files": sorted(files), "skipped": [], "taught": []}

    ident = identity_from(files)
    if ident:
        try:
            got = _post("/api/soul", {"identity": ident})["identity"]
            result["identity"] = got
            ledger["__identity__"] = digests.get("soul/IDENTITY.md")
        except Exception as e:
            result["identity_error"] = f"{type(e).__name__}: {e}"

    # Teach the substance. Only when the bytes differ from last time: she keeps
    # everything she is told, so repeating an unchanged file would bloat her.
    for rel, text in files.items():
        if rel == "soul/IDENTITY.md" and ident and not force:
            continue  # already applied structurally
        if not force and ledger.get(rel) == digests[rel]:
            result["skipped"].append(rel)
            continue
        try:
            r = _post("/api/soul", {"teach": f"[{rel}]\n{text}\n", "private": True})
            result["taught"].append({"file": rel,
                                     "private_bytes": r.get("private_bytes")})
            ledger[rel] = digests[rel]
        except Exception as e:
            result["taught"].append({"file": rel,
                                     "error": f"{type(e).__name__}: {e}"})

    _save_ledger(ledger)

    if verbose:
        print(f"  soul files : {len(files)} from {WORKSPACE}")
        if result.get("identity"):
            print(f"  identity   : {json.dumps(result['identity'])[:160]}")
        for t in result["taught"]:
            mark = "ok " if "error" not in t else "ERR"
            print(f"  {mark} {t['file']:<34} {t.get('private_bytes', t.get('error'))}")
        for s in result["skipped"]:
            print(f"  -- {s:<34} unchanged, not re-taught")
    return result


def main():
    ap = argparse.ArgumentParser(description="connect the memory repository to the mind")
    ap.add_argument("--once", action="store_true", help="sync, hand over, exit")
    ap.add_argument("--loop", action="store_true", help="stay up and re-sync")
    ap.add_argument("--report", action="store_true", help="show what she carries")
    ap.add_argument("--no-pull", action="store_true", help="skip the GitHub pull")
    ap.add_argument("--force", action="store_true", help="re-teach even if unchanged")
    a = ap.parse_args()

    if a.report:
        try:
            print(json.dumps(_get("/api/soul"), indent=2)[:2000])
        except Exception as e:
            print(f"  cannot reach her: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0

    def once():
        if not a.no_pull:
            print("pulling the canonical repository...")
            print("  " + json.dumps(pull_from_github()))
        print("handing the soul to the mind...")
        r = hand_over(force=a.force)
        print(json.dumps({k: v for k, v in r.items() if k != "taught"})[:300])
        return 0 if r.get("ok") else 1

    if a.loop:
        while True:
            once()
            print(f"  sleeping {INTERVAL:.0f}s")
            time.sleep(INTERVAL)
    return once()


if __name__ == "__main__":
    sys.exit(main())
