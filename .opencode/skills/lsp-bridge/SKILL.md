---
name: LSP Bridge
description: Operate and debug the local LSP bridge (tools/lsp_bridge_mcp.py) and the language servers behind it. Use for diagnostics, hover, definition, or when the bridge reports failed/Connection closed.
---

# The LSP bridge

`tools/lsp_bridge_mcp.py` supplies real language-server feedback to OpenCode
v2, whose binary has no LSP client of its own. It is an **LSP client** speaking
Content-Length framing to real servers, and an **MCP server** speaking newline
JSON-RPC to the agent.

## Tools and their argument names

| Tool | Arguments |
|---|---|
| `status` | none |
| `diagnostics` | `path` (absolute) |
| `hover` | `path`, `line`, `character` (0-based) |
| `definition` | `path`, `line`, `character` |
| `references` | `path`, `line`, `character` |
| `document_symbols` | `path` |
| `workspace_symbols` | `query` |

`diagnostics` takes **`path`**. Calling it with `file` fails with
`tool diagnostics failed: 'path'` — the schema is not the argument you guess.

## Start with status

```bash
python3 - <<'PY'
import json, subprocess
# or just: lsp_status via the MCP tool
PY
```

`status` reports each language server's state, pid, RSS, open documents and
last error. Check it before blaming a tool call — a `stopped` server with an
`error` field explains every downstream failure.

## The servers

| Language | Binary |
|---|---|
| typescript | `typescript-language-server` |
| python | `pyright-langserver` |
| shell | `bash-language-server` |

They are resolved from `PATH` at spawn time. A spawn failure logged as
`No such file or directory: 'typescript-language-server'` means the `PATH` the
bridge inherited lacks their directory — the shims live in `~/.local/bin`.

## Memory and the eager-start guard

`MIN_FREE_MB` (default 600) gates eager start; below it the bridge logs
`low memory … lazy start` and starts servers on first use instead. That is
healthy, not a failure.

Orphaned language servers are the real hazard — a stray pyright holds
1.5–2.1 GB RSS. The bridge reaps children on `SIGTERM`/`SIGINT`/`SIGHUP`.
After many restarts, check for strays:

```bash
pgrep -af "pyright-langserver|typescript-language-server|bash-language-server"
```

## Low memory makes startup flaky

Under memory pressure the bridge logs `spawn failed` and the MCP client may
report `Connection closed`. When several bridge processes plus their servers
have accumulated, reclaim first, then restart — see the `mcp-server` skill.

## Live self-reload

The bridge watches its own mtime and `os.execv`s in place, so an edit applies
without an MCP reconnect, the pid stays stable and the client never notices.
It defers the reload while a tool call is in flight. Because the file is
re-executed in place, a **syntax error takes the server down instantly** —
compile before saving.

## Diagnostics caveats

- `tsserver` emits a premature *empty* publish about 5 s before the real one,
  so an empty result is grace-waited rather than reported as a clean file.
- A missing import in pyright (`reportMissingImports`) is often a venv path
  issue, not a code defect. Check `pyrightconfig.json` points at the right
  interpreter before "fixing" the import.

## Verifying a change

```bash
python3 -m py_compile tools/lsp_bridge_mcp.py
lsp diagnostics <the file you edited>     # must be a real result, not an error
```
