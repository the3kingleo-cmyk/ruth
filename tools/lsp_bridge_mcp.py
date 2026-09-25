#!/usr/bin/env python3
"""lsp_bridge_mcp.py — real LSP client <-> MCP server bridge (stdlib only).

opencode v2 has no LSP client: the binary contains zero textDocument/*
handlers and the docs say built-in LSP is intentionally gone. This bridge
supplies it: it speaks the LSP *client* protocol (Content-Length framing) to
real language servers, and the MCP *agent* protocol (newline JSON-RPC, stdio)
to opencode — so the agent gets live diagnostics/hover/definition/symbols.

language servers (all already on PATH, smoke-tested):
  typescript  -> typescript-language-server --stdio
  python      -> pyright-langserver --stdio
  shell       -> bash-language-server --stdio

MCP tools: status, diagnostics, hover, definition, references,
           document_symbols, workspace_symbols

No third-party deps. Logging goes to ~/.local/status/lsp_bridge.log.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

LOG = Path(os.environ.get(
    "LSP_BRIDGE_LOG", os.path.expanduser("~/.local/status/lsp_bridge.log")))
ROOT = os.environ.get("LSP_BRIDGE_ROOT") or os.path.expanduser("~")
IDLE_SHUTDOWN = float(os.environ.get("LSP_BRIDGE_IDLE", "1200"))
INIT_TIMEOUT = float(os.environ.get("LSP_BRIDGE_INIT_TIMEOUT", "30"))
REQ_TIMEOUT = 12.0
DIAG_TIMEOUT = 15.0
# tsserver emits a premature *empty* publish before the computed one (measured
# +5.3s empty / +5.4s real on a file with 3 errors), so after an empty publish
# we grace-wait once for the follow-up instead of reporting a clean file.
DIAG_GRACE = 3.0
MIN_FREE_MB = int(os.environ.get("LSP_BRIDGE_MIN_FREE_MB", "600"))

# ---- live self-reload -------------------------------------------------
# opencode owns this process, so an edited bridge normally needs an MCP
# reconnect (i.e. a restart) to take effect. Instead we watch our own file
# and re-exec in place: the pid, the stdio pipes and the MCP client all stay
# put, so an update is applied in real time with zero service restarts.
SELF = os.path.realpath(__file__)
CODE_MTIME = os.path.getmtime(SELF)
BOOT_AT = time.time()
RELOAD_POLL = float(os.environ.get("LSP_BRIDGE_RELOAD_POLL", "10"))
BRIDGE_VERSION = "1.1.0"

_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT = 0


def _flight_enter():
    global _IN_FLIGHT
    with _FLIGHT_LOCK:
        _IN_FLIGHT += 1


def _flight_exit():
    global _IN_FLIGHT
    with _FLIGHT_LOCK:
        _IN_FLIGHT -= 1


def _in_flight():
    with _FLIGHT_LOCK:
        return _IN_FLIGHT


SERVERS = {
    "typescript": {
        "cmd": ["typescript-language-server", "--stdio"],
        "label": "typescript-language-server",
        "exts": {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"},
        # tsserver v6 refuses to start unless it can resolve a typescript
        # install from the workspace root; HOME has none, so point it at the
        # copy already on this box instead of polluting ~/node_modules.
        "init": {
            "tsserver": {
                "path": os.path.expanduser(
                    "~/.local/opt/tslsp/node_modules/typescript/lib/"
                    "tsserver.js"),
            },
        },
    },
    "python": {
        "cmd": ["pyright-langserver", "--stdio"],
        "label": "pyright-langserver",
        "exts": {".py", ".pyi", ".pyw"},
    },
    "shell": {
        # v5+ subcommand: bare `--stdio` is not a valid command (usage shows
        # `bash-language-server start`), which made initialize hang until
        # timeout on the first probe.
        "cmd": ["bash-language-server", "start"],
        "label": "bash-language-server",
        "exts": {".sh", ".bash", ".zsh"},
    },
}

LANGUAGE_ID = {
    ".ts": "typescript", ".tsx": "typescriptreact",
    ".js": "javascript", ".jsx": "javascriptreact",
    ".mjs": "javascript", ".cjs": "javascript",
    ".mts": "typescript", ".cts": "typescript",
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".sh": "shellscript", ".bash": "shellscript", ".zsh": "shellscript",
}

SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}
KIND = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object",
    20: "key", 21: "null", 22: "enum-member", 23: "struct", 24: "event",
    25: "operator", 26: "type-parameter",
}


def log(msg):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.time():.3f}] {msg}\n")
    except OSError:
        # logging must never raise back into the caller
        return


def path_to_uri(path):
    return Path(path).resolve().as_uri()


def uri_to_path(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    return unquote(parsed.path)


def pos(line, character):
    return {"line": int(line), "character": int(character)}


def range_dict(rng):
    if not isinstance(rng, dict):
        return None
    return {
        "start": {"line": rng.get("start", {}).get("line", 0),
                  "character": rng.get("start", {}).get("character", 0)},
        "end": {"line": rng.get("end", {}).get("line", 0),
                "character": rng.get("end", {}).get("character", 0)},
    }


def range_text(text, rng):
    """Human-readable one-liner for a range (1-based, like editors show)."""
    if not isinstance(rng, dict):
        return text
    s, e = rng.get("start", {}), rng.get("end", {})
    return f"{s.get('line', 0) + 1}:{s.get('character', 0) + 1}-" \
           f"{e.get('line', 0) + 1}:{e.get('character', 0) + 1} {text}"


def rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError):
        # pid vanished or /proc entry unreadable -> unknown, not fatal
        return None
    return None


def mem_available_mb():
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        # meminfo unreadable -> assume low memory so we stay lazy
        return 0
    return 0


class LspError(Exception):
    pass


class LanguageServer:
    """One language server process + LSP client state for it."""

    def __init__(self, key, cfg):
        self.key = key
        self.cfg = cfg
        self.proc = None
        self.next_id = 0
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.docs = {}
        self.diags = {}
        self.diag_events = {}
        self.fresh = {}
        self.state = "stopped"
        self.error = None
        self.started_at = 0.0
        self.last_used = 0.0
        self.in_flight = 0
        self.lock = threading.Lock()
        self.ready = threading.Event()

    # ---------- process / framing ----------
    def start(self):
        with self.lock:
            if self.state == "running" and self.proc and self.proc.poll() is None:
                self.last_used = time.time()
                return True
            try:
                self.proc = subprocess.Popen(
                    self.cfg["cmd"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.state = "failed"
                self.error = f"spawn failed: {exc}"
                log(f"{self.key}: {self.error}")
                return False
            self.state = "initializing"
            self.error = None
            self.started_at = time.time()
            self.last_used = time.time()
            self.ready.clear()
            threading.Thread(target=self._reader, daemon=True,
                             name=f"lsp-read-{self.key}").start()
        params = {
            "processId": os.getpid(),
            "clientInfo": {"name": "lsp-bridge-mcp", "version": "1.0"},
            "rootUri": path_to_uri(ROOT),
            "workspaceFolders": [
                {"uri": path_to_uri(ROOT), "name": os.path.basename(ROOT)}],
            "initializationOptions": self.cfg.get("init"),
            "capabilities": {
                "workspace": {
                    "configuration": False,
                    "workspaceFolders": True,
                    "applyEdit": False,
                },
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "publishDiagnostics": {"relatedInformation": False},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": False},
                    "references": {},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True},
                },
                "window": {"workDoneProgress": True},
            },
        }
        try:
            self.request("initialize", params, INIT_TIMEOUT)
        except LspError as exc:
            self.state = "failed"
            self.error = f"initialize failed: {exc}"
            log(f"{self.key}: {self.error}")
            self.stop()
            return False
        self.notify("initialized", {})
        self.state = "running"
        self.ready.set()
        log(f"{self.key}: running pid={self.proc.pid}")
        return True

    def stop(self):
        proc = self.proc
        self.proc = None
        self.state = "stopped"
        self.ready.clear()
        self.docs.clear()
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                try:
                    proc.kill()
                except OSError:
                    # already reaped by someone else
                    log(f"{self.key}: kill raced an exited process")

    def _reader(self):
        proc = self.proc
        if proc is None:
            log(f"{self.key}: reader: no process")
            return
        out = proc.stdout
        if out is None:
            log(f"{self.key}: reader: stdout unavailable")
            return
        try:
            while True:
                headers = {}
                while True:
                    line = out.readline()
                    if not line:
                        return
                    line = line.strip()
                    if not line:
                        break
                    if b":" in line:
                        k, _, v = line.partition(b":")
                        headers[k.decode().strip().lower()] = v.decode().strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                body = out.read(length)
                if not body:
                    return
                try:
                    msg = json.loads(body.decode("utf-8", "replace"))
                except ValueError:
                    # one malformed frame must not kill the reader loop
                    log(f"{self.key}: dropped unparsable frame")
                    continue
                self._dispatch(msg)
        except (OSError, ValueError) as exc:
            log(f"{self.key}: reader died: {exc}")
        if self.state == "running":
            self.state = "died"
            self.error = "language server exited"

    def _dispatch(self, msg):
        if "method" in msg:
            if "id" in msg:
                threading.Thread(target=self._answer, args=(msg,),
                                 daemon=True).start()
            else:
                self._on_notification(msg)
            return
        mid = msg.get("id")
        with self.pending_lock:
            waiter = self.pending.pop(mid, None)
        if waiter:
            waiter[1] = msg
            waiter[0].set()

    def _answer(self, msg):
        method, mid = msg.get("method"), msg.get("id")
        result = None
        if method == "workspace/configuration":
            items = msg.get("params", {}).get("items", [])
            result = [{} for _ in items]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": path_to_uri(ROOT),
                       "name": os.path.basename(ROOT)}]
        elif method in ("client/registerCapability",
                        "client/unregisterCapability",
                        "window/workDoneProgress/create"):
            result = None
        elif method == "workspace/applyEdit":
            result = {"applied": False,
                      "failureReason": "read-only bridge"}
        elif method in ("window/showMessageRequest",):
            result = None
        else:
            result = None
        self.send({"jsonrpc": "2.0", "id": mid, "result": result})

    def _on_notification(self, msg):
        method = msg.get("method")
        if method == "textDocument/publishDabilities":  # defensive typo-guard
            return
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri")
            self.diags[uri] = params.get("diagnostics", [])
            ev = self.diag_events.get(uri)
            if ev:
                ev.set()
        elif method in ("window/logMessage", "window/showMessage",
                        "$/progress", "telemetry/event"):
            pass

    # ---------- requests ----------
    def send(self, payload):
        with self.write_lock:
            if not self.proc or not self.proc.stdin:
                raise LspError(f"{self.key} not running")
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
            self.proc.stdin.write(header + raw)
            self.proc.stdin.flush()

    def request(self, method, params, timeout=REQ_TIMEOUT):
        if not self.proc:
            raise LspError(f"{self.key} not running")
        self.next_id += 1
        mid = self.next_id
        waiter = [threading.Event(), None]
        with self.pending_lock:
            self.pending[mid] = waiter
        self.send({"jsonrpc": "2.0", "id": mid, "method": method,
                   "params": params})
        if not waiter[0].wait(timeout):
            with self.pending_lock:
                self.pending.pop(mid, None)
            raise LspError(f"timeout waiting for {method} ({self.key})")
        msg = waiter[1]
        if "error" in msg:
            raise LspError("{}: {}".format(method, msg["error"]))
        return msg.get("result")

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    # ---------- document sync ----------
    def sync_doc(self, path, wait_diags=True):
        uri = path_to_uri(path)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        mtime = os.path.getmtime(path)
        language = LANGUAGE_ID.get(Path(path).suffix.lower(), "plaintext")
        doc = self.docs.get(uri)
        if doc is None:
            ev = threading.Event()
            self.diag_events[uri] = ev
            self.notify("textDocument/didOpen", {
                "textDocument": {"uri": uri, "languageId": language,
                                 "version": 1, "text": text}})
            self.docs[uri] = {"version": 1, "mtime": mtime}
        elif doc["mtime"] != mtime:
            ev = self.diag_events.setdefault(uri, threading.Event())
            ev.clear()
            doc["version"] += 1
            doc["mtime"] = mtime
            self.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": doc["version"]},
                "contentChanges": [{"text": text}]})
        if wait_diags:
            ev = self.diag_events.get(uri)
            if ev:
                deadline = time.time() + DIAG_TIMEOUT
                saw_publish = False
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    ev.wait(remaining)
                    saw_publish = True
                    if self.diags.get(uri):
                        break
                    # empty publish: allow one follow-up to land
                    ev.clear()
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    ev.wait(min(DIAG_GRACE, remaining))
                    if self.diags.get(uri) or ev.is_set():
                        break
                    break
                self.fresh[uri] = saw_publish
            else:
                self.fresh[uri] = False
        return uri

    def diagnostics_for(self, uri):
        return self.diags.get(uri, [])


BRIDGE_SERVERS = {k: LanguageServer(k, v) for k, v in SERVERS.items()}
DOC_OWNER = {}          # uri -> server key
GUARD = threading.Lock()

# Editor/backup/temp suffixes that must not defeat source-language detection:
# lsp_bridge_mcp.py.new, .tmp, .bak, .orig, *~ ... all typecheck as Python.
_TEMP_SUFFIXES = (".bak", ".copy", ".download", ".fixed", ".new", ".old",
                  ".orig", ".original", ".part", ".save", ".swp", ".tmp")


def _source_suffix(path):
    """Real source suffix ('.py', '.ts', ...) with temp/backup tails stripped."""
    name = Path(path).name.lower()
    changed = True
    while changed:
        changed = False
        if name.endswith("~"):
            name = name[:-1]
            changed = True
        for suf in _TEMP_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)]
                changed = True
                break
    return Path(name).suffix


def server_for(path):
    suffix = _source_suffix(path)
    for key, cfg in SERVERS.items():
        if suffix in cfg["exts"]:
            return BRIDGE_SERVERS[key]
    return None


def ensure_server(key):
    srv = BRIDGE_SERVERS[key]
    if srv.state == "running" and srv.proc and srv.proc.poll() is None:
        srv.last_used = time.time()
        return srv
    if not srv.start():
        raise LspError("{} unavailable: {}".format(SERVERS[key]["label"],
                                               srv.error))
    return srv


def open_document(path, wait_diags=False):
    path = os.path.realpath(os.path.abspath(path))
    if not os.path.isfile(path):
        raise LspError(f"no such file: {path}")
    srv = server_for(path)
    if srv is None:
        raise LspError(
            "no language server for {} (have: {})".format(
                _source_suffix(path) or "no extension",
                ", ".join(sorted(e for c in SERVERS.values()
                                  for e in c["exts"]))))
    srv = ensure_server(srv.key)
    srv.in_flight += 1
    srv.last_used = time.time()
    try:
        uri = srv.sync_doc(path, wait_diags=wait_diags)
    finally:
        srv.in_flight -= 1
    DOC_OWNER[uri] = srv.key
    return srv, uri, path


def file_line_dict(uri, rng):
    return {"file": uri_to_path(uri), "range": range_dict(rng),
            "display": range_text("", rng).strip()}


def flatten_symbols(symbols, out=None, depth=0):
    out = out if out is not None else []
    for sym in symbols or []:
        entry = {
            "name": sym.get("name"),
            "kind": KIND.get(sym.get("kind"), sym.get("kind")),
            "range": range_dict(sym.get("selectionRange") or sym.get("range")),
            "depth": depth,
            "detail": sym.get("detail"),
        }
        if entry["range"]:
            entry["display"] = f"{entry['range']['start']['line'] + 1}:" \
                               f"{entry['range']['start']['character'] + 1} " \
                               f"{entry['name']}"
        out.append(entry)
        flatten_symbols(sym.get("children"), out, depth + 1)
    return out


# ---------------------------------------------------------------- MCP tools
TOOLS = [
    {
        "name": "status",
        "description": (
            "Live status of the LSP bridge: which language servers are "
            "running, pids, memory, open documents, plus bridge health. "
            "Call this first if a language tool misbehaves."),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "diagnostics",
        "description": (
            "Live compiler/linter diagnostics for one file, straight from "
            "the real language server (tsserver / pyright / bashls). Returns "
            "every problem with exact 1-based line:col, severity, source and "
            "message. Opens the file over LSP first, so results reflect the "
            "file as it exists on disk right now."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "absolute path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "hover",
        "description": ("LSP hover/type info at a position. line and "
                        "character are 0-based (character counts UTF-16 code "
                        "units, i.e. plain chars for ASCII)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer", "description": "0-based line"},
                "character": {"type": "integer",
                              "description": "0-based column"},
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "definition",
        "description": ("Jump to the definition of the symbol at a position; "
                        "returns every target file with line:col. 0-based "
                        "line/character."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer"},
                "character": {"type": "integer"},
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "references",
        "description": ("Find all references to the symbol at a position "
                        "across the workspace, with file and line:col. "
                        "0-based line/character."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer"},
                "character": {"type": "integer"},
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "document_symbols",
        "description": ("Outline of one file: functions, classes, methods "
                        "and their exact line:col positions, from the live "
                        "language server."),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "workspace_symbols",
        "description": ("Search symbols by name across the whole workspace "
                        "(go-to-symbol-in-workspace), with file and "
                        "line:col."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "symbol name"},
            },
            "required": ["query"],
        },
    },
]


def tool_status(_args):
    servers = []
    total_rss = 0.0
    for key, srv in BRIDGE_SERVERS.items():
        info = {
            "language": key,
            "server": SERVERS[key]["label"],
            "state": srv.state,
            "pid": srv.proc.pid if (srv.proc and srv.proc.poll() is None)
                   else None,
            "open_documents": len(srv.docs),
            "diagnostics_tracked": len(srv.diags),
            "idle_seconds": round(time.time() - srv.last_used, 1)
                            if srv.last_used else None,
            "error": srv.error,
        }
        pid = info["pid"]
        if pid:
            info["rss_mb"] = rss_mb(pid)
            if info["rss_mb"]:
                total_rss += info["rss_mb"]
        servers.append(info)
    return {
        "bridge": f"lsp-bridge-mcp {BRIDGE_VERSION}",
        "pid": os.getpid(),
        "code_mtime": round(CODE_MTIME, 3),
        "live_reload": os.environ.get("LSP_BRIDGE_RELOAD", "1") != "0",
        "uptime_seconds": round(time.time() - BOOT_AT, 1),
        "protocol": "LSP client (Content-Length) -> MCP agent protocol "
                    "(newline JSON-RPC, stdio)",
        "root": ROOT,
        "mem_available_mb": mem_available_mb(),
        "total_lsp_rss_mb": round(total_rss, 1),
        "idle_shutdown_seconds": IDLE_SHUTDOWN,
        "extensions": {k: sorted(SERVERS[k]["exts"]) for k in SERVERS},
        "servers": servers,
    }


def tool_diagnostics(args):
    srv, uri, path = open_document(args["path"], wait_diags=True)
    fresh = bool(srv.fresh.get(uri))
    raw = srv.diagnostics_for(uri)
    items = []
    counts = {"error": 0, "warning": 0, "info": 0, "hint": 0}
    for d in raw:
        sev = SEVERITY.get(d.get("severity"), "error")
        counts[sev] = counts.get(sev, 0) + 1
        items.append({
            "severity": sev,
            "line": (d.get("range", {}).get("start", {}).get("line", 0) + 1),
            "character": (d.get("range", {}).get("start", {})
                          .get("character", 0) + 1),
            "display": range_text(d.get("message", "").split("\n")[0],
                                  d.get("range")).strip(),
            "message": d.get("message"),
            "source": d.get("source"),
            "code": d.get("code"),
        })
    items.sort(key=lambda i: (i["line"], i["character"]))
    return {
        "file": path,
        "server": SERVERS[srv.key]["label"],
        "clean": not items,
        "diagnostics_fresh": fresh,
        "counts": counts,
        "diagnostics": items,
        **({} if fresh else {
            "note": "server had not published diagnostics for this document "
                    f"within {DIAG_TIMEOUT}s — results may be incomplete, retry"}),
    }


def _position_args(args):
    return pos(args.get("line", 0), args.get("character", 0))


def tool_hover(args):
    srv, uri, path = open_document(args["path"])
    with srv.lock:
        srv.in_flight += 1
    try:
        result = srv.request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": _position_args(args)})
    finally:
        srv.in_flight -= 1
    if not result:
        return {"file": path, "hover": None, "note": "no hover info here"}
    contents = result.get("contents")
    if isinstance(contents, dict):
        text = contents.get("value", "")
    elif isinstance(contents, list):
        text = "\n".join(c.get("value", "") if isinstance(c, dict) else str(c)
                         for c in contents)
    else:
        text = str(contents)
    return {"file": path, "server": SERVERS[srv.key]["label"],
            "hover": text.strip()}


def _locations(result):
    out = []
    if isinstance(result, dict):
        result = [result]
    for loc in result or []:
        if not isinstance(loc, dict):
            continue
        uri = loc.get("uri") or (loc.get("targetUri") if loc else None)
        rng = loc.get("range") or loc.get("targetSelectionRange") or \
            loc.get("targetRange")
        if not uri:
            continue
        out.append({
            "file": uri_to_path(uri),
            "line": (rng or {}).get("start", {}).get("line", 0) + 1,
            "character": (rng or {}).get("start", {}).get("character", 0) + 1,
            "display": range_text("", rng or {}).strip() or None,
        })
    return out


def tool_definition(args):
    srv, uri, path = open_document(args["path"])
    with srv.lock:
        srv.in_flight += 1
    try:
        result = srv.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": _position_args(args)})
    finally:
        srv.in_flight -= 1
    found = _locations(result)
    return {"file": path, "server": SERVERS[srv.key]["label"],
            "definitions": found, "count": len(found)}


def tool_references(args):
    srv, uri, path = open_document(args["path"])
    with srv.lock:
        srv.in_flight += 1
    try:
        result = srv.request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": _position_args(args),
            "context": {"includeDeclaration": True}})
    finally:
        srv.in_flight -= 1
    found = _locations(result)
    return {"file": path, "server": SERVERS[srv.key]["label"],
            "references": found, "count": len(found)}


def tool_document_symbols(args):
    srv, uri, path = open_document(args["path"])
    with srv.lock:
        srv.in_flight += 1
    try:
        result = srv.request("textDocument/documentSymbol",
                             {"textDocument": {"uri": uri}})
    finally:
        srv.in_flight -= 1
    if result and isinstance(result[0], dict) and "location" in result[0]:
        flat = []
        for sym in result:
            loc = sym.get("location", {})
            rng = loc.get("range", {})
            flat.append({
                "name": sym.get("name"),
                "kind": KIND.get(sym.get("kind"), sym.get("kind")),
                "file": uri_to_path(loc.get("uri", uri)),
                "line": rng.get("start", {}).get("line", 0) + 1,
                "character": rng.get("start", {}).get("character", 0) + 1,
                "depth": 0,
                "display": f"{rng.get('start', {}).get('line', 0) + 1}:"
                           f"{rng.get('start', {}).get('character', 0) + 1} "
                           f"{sym.get('name')}",
            })
    else:
        flat = flatten_symbols(result or [])
    return {"file": path, "server": SERVERS[srv.key]["label"],
            "symbols": flat, "count": len(flat)}


def tool_workspace_symbols(args):
    query = str(args.get("query", "")).strip()
    if not query:
        raise LspError("query must not be empty")
    results = []
    servers_used = []
    for key, srv in BRIDGE_SERVERS.items():
        if srv.state != "running":
            srv = ensure_server(key)
        with srv.lock:
            srv.in_flight += 1
        try:
            found = srv.request("workspace/symbol", {"query": query})
        except LspError:
            # one language server being down must not empty the whole result
            found = []
        finally:
            srv.in_flight -= 1
        servers_used.append(SERVERS[key]["label"])
        for sym in found or []:
            loc = sym.get("location", {})
            uri = loc.get("uri")
            rng = loc.get("range", {})
            if not uri:
                continue
            results.append({
                "name": sym.get("name"),
                "kind": KIND.get(sym.get("kind"), sym.get("kind")),
                "file": uri_to_path(uri),
                "line": rng.get("start", {}).get("line", 0) + 1,
                "character": rng.get("start", {}).get("character", 0) + 1,
            })
    results.sort(key=lambda r: (r["file"], r["line"]))
    return {"query": query, "servers": servers_used, "symbols": results[:200],
            "count": len(results)}


DISPATCH = {
    "status": tool_status,
    "diagnostics": tool_diagnostics,
    "hover": tool_hover,
    "definition": tool_definition,
    "references": tool_references,
    "document_symbols": tool_document_symbols,
    "workspace_symbols": tool_workspace_symbols,
}


# ---------------------------------------------------------------- idle sweep
def idle_sweep():
    while True:
        time.sleep(60)
        for srv in BRIDGE_SERVERS.values():
            if srv.state != "running" or not srv.last_used:
                continue
            idle = time.time() - srv.last_used
            if idle > IDLE_SHUTDOWN and srv.in_flight == 0:
                log(f"{srv.key}: idle {idle:.0f}s -> shutdown")
                srv.stop()


def eager_start():
    if mem_available_mb() < MIN_FREE_MB:
        log(f"low memory ({mem_available_mb()} MB) -> lazy start")
        return
    for key in SERVERS:
        try:
            BRIDGE_SERVERS[key].start()
        except (LspError, OSError) as exc:
            log(f"eager start {key} failed: {exc}")


# ---------------------------------------------------------------- MCP stdio
def mcp_reply(msg_id, result):
    sys.stdout.write(json.dumps(
        {"jsonrpc": "2.0", "id": msg_id, "result": result},
        ensure_ascii=False) + "\n")
    sys.stdout.flush()


def mcp_error(msg_id, code, message):
    sys.stdout.write(json.dumps({
        "jsonrpc": "2.0", "id": msg_id,
        "error": {"code": code, "message": message}}) + "\n")
    sys.stdout.flush()


def handle(msg):
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        client_version = params.get("protocolVersion", "2025-06-18")
        mcp_reply(msg_id, {
            "protocolVersion": client_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "lsp-bridge", "version": BRIDGE_VERSION},
            "instructions": (
                "Live LSP diagnostics and navigation for TypeScript/"
                "JavaScript, Python and Shell. Call `status` to see server "
                "health; `diagnostics` after any code edit returns the real "
                "compiler errors with 1-based line:col."),
        })
        threading.Thread(target=eager_start, daemon=True).start()
        return

    if method == "tools/list":
        mcp_reply(msg_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            mcp_error(msg_id, -32602, "tool name must be a string")
            return
        args = params.get("arguments") or {}
        fn = DISPATCH.get(name)
        if fn is None:
            mcp_error(msg_id, -32602, f"unknown tool: {name}")
            return
        try:
            _flight_enter()
            try:
                result = fn(args)
            finally:
                _flight_exit()
            text = json.dumps(result, ensure_ascii=False, indent=1)
            mcp_reply(msg_id, {"content": [{"type": "text", "text": text}],
                               "isError": False})
        except Exception as exc:  # noqa: BLE001 - a tool must never kill MCP
            text = f"ERROR ({name}): {exc}"
            log(f"tool {name} failed: {exc}")
            mcp_reply(msg_id, {"content": [{"type": "text", "text": text}],
                               "isError": True})
        return

    if method == "ping":
        mcp_reply(msg_id, {})
        return

    if method in ("logging/setLevel", "completion/complete"):
        mcp_reply(msg_id, {})
        return

    if msg_id is not None:
        mcp_error(msg_id, -32601, f"method not found: {method}")


def _live_reload_watch():
    """Apply edits to this file without restarting opencode or the MCP link.

    Strategy: poll our own mtime; when it changes, debounce (editors write in
    bursts), wait until no tool call is in flight, reap the language servers
    so they are not orphaned by the image swap, then `os.execv` — which keeps
    the pid and the stdio pipes, so opencode never notices a disconnect.
    """
    seen = CODE_MTIME
    while True:
        time.sleep(RELOAD_POLL)
        try:
            mt = os.path.getmtime(SELF)
        except OSError:
            # file replaced mid-edit; retry next tick
            continue
        if mt == seen:
            continue
        time.sleep(RELOAD_POLL)          # debounce: let the write settle
        try:
            mt2 = os.path.getmtime(SELF)
        except OSError:
            continue
        if mt2 != mt:
            seen = mt2
            continue
        if _in_flight():                 # never yank a live tool call
            log("code changed but a tool call is in flight -> deferred")
            continue
        seen = mt2
        log(f"code changed (mtime {mt2:.3f}) -> live reload via execv")
        for srv in BRIDGE_SERVERS.values():
            srv.stop()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as exc:
            # keep running the old image rather than dying
            log(f"live reload execv failed, staying on current code: {exc}")


def _install_signal_handlers():
    """Reap language servers on SIGTERM/SIGINT.

    Without this the default action kills the bridge only, leaving pyright/
    bashls/tsserver alive (measured: an orphaned pyright held 1.5-2.1 GB RSS
    across a service restart). Raising SystemExit runs main()'s finally.
    """

    def _bye(signum, _frame):
        log(f"signal {signum} -> shutting down children")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _bye)
        except (OSError, ValueError):
            # not the main thread, or platform refuses this signal
            log(f"could not install handler for {sig}")


def main():
    _install_signal_handlers()
    if os.environ.get("LSP_BRIDGE_RELOAD", "1") != "0":
        threading.Thread(target=_live_reload_watch, daemon=True,
                         name="lsp-reload-watch").start()
    log(f"bridge start pid={os.getpid()} root={ROOT} "
        f"mem={mem_available_mb()}MB version={BRIDGE_VERSION} "
        f"code_mtime={CODE_MTIME:.3f} live_reload=on")
    threading.Thread(target=idle_sweep, daemon=True).start()
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length"):
            headers = {}
            while True:
                hdr = stdin.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
                if b":" in hdr:
                    k, _, v = hdr.partition(b":")
                    headers[k.decode().strip().lower()] = \
                        v.decode().strip()
            length = int(headers.get("content-length", "0"))
            body = stdin.read(length) if length else b""
        else:
            body = stripped
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            # malformed client frame -> skip it, keep the loop alive
            log("dropped unparsable client frame")
            continue
        if isinstance(msg, list):
            for one in msg:
                handle(one)
        else:
            handle(msg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # service restarts send SIGTERM; reap children there too, otherwise
        # pyright/bashls/tsserver survive as 1-2 GB orphans
        for srv in BRIDGE_SERVERS.values():
            srv.stop()
