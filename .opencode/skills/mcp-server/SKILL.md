---
name: MCP Server Registration
description: Add, repair, or verify an MCP server in OpenCode v2 — correct config location, atomic writes, service restart, and reading "Connection closed" correctly. Use when a server is missing, shows failed, or the MCP list looks wrong.
---

# Registering and verifying MCP servers (OpenCode v2)

## The one rule that breaks everything

OpenCode v2 registers MCP servers **directly under `mcp.<name>`**:

```json
{ "mcp": { "github": { ... }, "lsp": { "command": [...] } } }
```

`mcp.servers.<name>` is a **v1 shape that v2 silently ignores**. A server written
there does not error — it just never appears, and code that reads it back sees
nothing. Confirm what is actually registered before changing anything else:

```bash
python3 -c "import json;print(sorted(json.load(open('$HOME/.config/opencode/opencode.json'))['mcp']))"
```

## Procedure

1. **Put the source on disk** in the project's `tools/`, and install it to
   `~/.local/bin/`. The repo is the source of truth; the local copy is a build
   artifact. Keep them byte-identical (`diff -q`) or debugging becomes fiction.

2. **Verify the server standalone first**, speaking MCP by hand before wiring
   it into any config. A server that only fails inside OpenCode means the bug
   is the config or the environment, not the server:

   ```bash
   printf '%s\n%s\n' \
     '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' \
     '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
   | timeout 30 python3 ~/.local/bin/<server>.py
   ```

   A stdlib server needs no `mcp` package. If importing `mcp` fails, that
   package is a red herring — implement the JSON-RPC loop directly.

3. **Register atomically.** A truncating write (`json.dump(open(p,"w"))`) can
   be read mid-write by OpenCode's file watcher, which then reports **"No MCP
   servers configured"** — the entire `mcp` section vanishes. Always temp-file
   plus `os.replace` (see the `opencode-config-edit` skill).

4. **Restart the service.** A long-running `opencode serve --service` caches
   its MCP registry. A newly added server reports `Connection closed` until:

   ```bash
   opencode service restart
   ```

   This is the single most common cause of "I added it and it is still
   broken". Editing the config is not enough.

5. **Confirm:** `opencode mcp list` shows `✓ <name>  connected`.

## Reading "Connection closed"

The process was spawned and then exited or was killed. In order of likelihood:

- **Stale `serve` process** — the registry predates your change. Fix: restart.
- **Config written in the v1 shape**, or written non-atomically. See steps 1 and 3.
- **It crashes on the first real frame.** Capture the traceback instead of
  guessing — pipe stderr somewhere the client does not swallow it. Two bugs
  found this way: a handler that re-parsed an already-parsed frame
  (`TypeError`, server dies, client shows only "Connection closed"), and a
  binary whose *name* is a directory on the restricted `PATH`, giving `EACCES`
  when spawned.
- **A directory shadows the binary.** An MCP server gets a narrow `PATH`. Give
  it the real bin directory *and* resolve the executable in code, skipping
  anything that is not a file.

## Environment handed to the server

A local server inherits a **restricted** environment: your shell's `PATH` does
not apply. If the server needs a tool:

- add the directory to the server's `environment.PATH` in the config, **and**
- resolve it in code with a fallback list, so it survives a narrow `PATH`.

## Server design rules learned here

- Never let a tool raise out of the read loop. One exception kills the server.
- `if "error" in msg` is the **failure** test. Storing that as `ok` and then
  raising when `ok` is false inverts every success into a failure.
- Keep handshake results. Discarding the `initialize` return leaves every
  later field reading `None`.
- Guard input, not just output: `max(1, min(int(n or 8), 25))` turns an explicit
  `0` into `8`.
- Prefer stdlib and no credential. A keyless server is a feature.
