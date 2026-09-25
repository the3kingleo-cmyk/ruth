#!/usr/bin/env python3
"""install-agent — copy Ruth's agent from her repo into an opencode config.

`agent/ruth.md` is the canonical definition of who Ruth is. This reads its
frontmatter and body and installs them as the `ruth` agent in an opencode
config, so the agent an operator runs is the agent the repository ships
rather than a hand-copied duplicate that drifts.

  install-agent --check    report drift and change nothing
  install-agent --install  write the agent into the config

The config is written atomically: opencode watches it, and a truncating write
can be read mid-write, which drops the whole file's contents. The agent's
`system` is her body verbatim, so editing `agent/ruth.md` is enough to change
her; re-run this to publish it.

stdlib only -- no PyYAML -- because the repo carries no third-party runtime
dependency. The frontmatter subset used here is small and fixed.
"""
import argparse
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
AGENT_MD = REPO / "agent" / "ruth.md"
SOUL = ["soul/IDENTITY.md", "soul/SOUL.md", "soul/VOICE.md", "soul/USER.md"]

SCALARS = {"description", "mode", "model", "steps", "name"}


def split_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        raise SystemExit(f"{AGENT_MD} has no frontmatter")
    return m.group(1), m.group(2).strip()


def parse_frontmatter(fm):
    """Parse the YAML subset agent/ruth.md uses.

    Handles scalars, a flat `tools:` map, and a `permissions:` list of
    mappings. The list items are written as `- action: x` -- the dash is on
    the same line as the first key, so a parser that only looks at
    `key: value` lines silently drops `action` from every rule and installs
    an agent with no permissions at all.
    """
    agent, tools, perms = {}, {}, []
    section = None
    for raw in fm.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.strip()
        indent = len(raw) - len(raw.lstrip())

        if indent == 0:
            section = None
            if line.endswith(":"):
                section = line[:-1].strip()
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                if key in SCALARS:
                    val = val.strip().strip('"')
                    agent[key] = int(val) if key == "steps" else val
                continue
            continue

        if section == "tools" and ":" in line:
            k, _, v = line.partition(":")
            tools[k.strip()] = v.strip() == "true"
        elif section == "permissions" and ":" in line:
            # A leading "- " starts a NEW rule. Treating it as just another
            # key/value line merges every rule into one and drops `action`.
            if line.startswith("- "):
                k, _, v = line[2:].partition(":")
                perms.append({k.strip(): v.strip().strip('"')})
            elif perms:
                k, _, v = line.partition(":")
                perms[-1][k.strip()] = v.strip().strip('"')

    if perms:
        agent["permissions"] = [{k: v for k, v in perm.items() if v != ""}
                                for perm in perms]
    if tools:
        agent["tools"] = tools
    return agent


def build_agent():
    fm, body = split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
    agent = parse_frontmatter(fm)
    agent.setdefault("mode", "primary")
    # her body is her system prompt, verbatim
    agent["system"] = body
    return agent


def write_atomic(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser(
        "~/.config/opencode/opencode.json"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()
    if not (args.check or args.install):
        args.check = True

    want = build_agent()
    cfg_path = pathlib.Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    have = (cfg.get("agents") or {}).get("ruth")

    drift = []
    if not have:
        drift.append("no ruth agent in the config")
    else:
        for key in ("system", "description", "mode", "model", "steps",
                    "tools", "permissions"):
            if key in want and have.get(key) != want[key]:
                drift.append(key)
    instructions = cfg.get("instructions") or []
    for rel in SOUL:
        if str(REPO / rel) not in instructions:
            drift.append(f"instructions missing {rel} (must be absolute)")

    if not drift:
        print("Ruth's agent is already installed and current.")
        return 0
    print("drift detected:")
    for d in drift:
        print("  -", d)
    if args.check or not args.install:
        print("\nre-run with --install to publish her repo's definition.")
        return 1

    cfg.setdefault("agents", {})["ruth"] = want
    # Absolute: these are read by the opencode process, whose working
    # directory is not the repository. A repo-relative path silently resolves
    # to nothing and her soul never loads.
    cfg["instructions"] = [str(REPO / rel) for rel in SOUL if (REPO / rel).exists()] + \
        [p for p in instructions if p not in SOUL]
    write_atomic(cfg_path, cfg)
    back = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert back["agents"]["ruth"]["system"] == want["system"], "write lost the body"
    assert "ruth" in back["agents"] and back.get("instructions"), "write lost sections"
    print("\ninstalled Ruth's agent into", cfg_path)
    print("  mode        :", back["agents"]["ruth"]["mode"])
    print("  model       :", back["agents"]["ruth"]["model"])
    print("  steps       :", back["agents"]["ruth"].get("steps"))
    print("  tools       :", len(back["agents"]["ruth"].get("tools") or {}))
    print("  system      :", len(back["agents"]["ruth"]["system"]), "chars")
    print("  instructions:", back["instructions"])
    print("\nrestart opencode for the new agent to take effect:")
    print("  opencode service restart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
