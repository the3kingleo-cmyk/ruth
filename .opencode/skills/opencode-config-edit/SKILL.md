---
name: OpenCode Config Editing
description: Safely edit ~/.config/opencode/opencode.json without losing whole sections to a read-mid-write race. Use before any script or agent rewrites the OpenCode config.
---

# Editing the OpenCode config safely

## The failure this prevents

OpenCode watches `opencode.json` and re-reads it on change. A truncating
write gives the watcher a window onto an empty or partial file, and OpenCode
then reports:

```
No MCP servers configured
```

Nothing is wrong with the servers. The file was read at the wrong instant.
This looks like "my config change was ignored" and is usually misdiagnosed as
a schema problem.

## Always write atomically

```python
import json, os

path = os.path.expanduser("~/.config/opencode/opencode.json")
with open(path) as fh:
    cfg = json.load(fh)          # read
cfg["mcp"]["name"] = entry       # modify

tmp = path + ".tmp"              # write to a sibling, then swap
with open(tmp, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, path)            # atomic on POSIX
```

`os.replace` is atomic on the same filesystem, so a reader sees either the old
complete file or the new complete file.

**Never** do `json.dump(cfg, open(path, "w"))` — that truncates first and
writes second.

## Verify after writing

A write that "succeeded" can still have dropped everything else. Re-read and
assert the parts you did not touch survived:

```python
back = json.load(open(path))
for keep in ("github", "lsp"):
    assert keep in back["mcp"], f"{keep} was lost in the write"
```

Back up first, keep the backup outside the watched directory if you want to
be certain the watcher ignores it, and prefer a name that does not end in
`.json` (a `.json` sibling may itself be picked up).

## Schema notes for v2

- MCP servers live at `mcp.<name>`, not `mcp.servers.<name>`.
- `permissions` is an **array** of `{action, resource, effect}` rules. The old
  `permission: "allow"` object form is legacy.
- `websearch` is an object (`{"provider": "..."}`) or `false`. Each provider
  needs its own key; with none set, every search fails at call time.
- Unknown top-level keys are usually ignored rather than rejected, so a typo
  fails silently. Prefer checking the live schema:
  `curl -sSL https://opencode.ai/config.json`.

## After any change

```bash
opencode service restart     # the serve process caches config
opencode mcp list            # confirm what actually took effect
```

Configuration on disk is not configuration in effect. Always verify through
the client.
