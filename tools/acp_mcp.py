#!/usr/bin/env python3
"""acp_mcp — a real Agent Client Protocol client, exposed as an MCP server.

`opencode acp` speaks ACP over stdio to whichever editor launches it, so ACP
never shows up among the MCP servers an agent can see. This server closes that
gap honestly: it *is* an ACP client. It launches `opencode acp`, performs the
ACP handshake, and exposes the live session surface as MCP tools.

  acp_status        handshake state, negotiated capabilities, agent info
  acp_capabilities  the capability map the agent negotiated
  acp_sessions      ACP sessions on this box
  acp_new_session   open a new ACP session in a directory
  acp_prompt        send a prompt to a session and return the reply text
  acp_restart       drop the child and re-handshake

One child is kept alive and reused, so the handshake cost is paid once. stdlib
only; no third-party dependency.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

VERSION = "1.0.0"


def resolve_opencode():
    """Find an executable `opencode`, never a directory that shadows it.

    ~/.local/bin/opencode exists as a *directory* on this box and comes first
    on the restricted PATH an MCP server is given, so a bare "opencode" spawn
    raised EACCES instead of running the real binary in ~/.opencode/bin.
    """
    env = os.environ.get("OPENCODE_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("opencode")
    if found and os.path.isfile(found) and os.access(found, os.X_OK):
        return found
    for cand in (os.path.expanduser("~/.opencode/bin/opencode"),
                 "/usr/local/bin/opencode", "/usr/bin/opencode"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return "opencode"


OPENCODE = resolve_opencode()
HANDSHAKE_TIMEOUT = float(os.environ.get("ACP_HANDSHAKE_TIMEOUT", "45"))
CALL_TIMEOUT = float(os.environ.get("ACP_CALL_TIMEOUT", "60"))
PROMPT_TIMEOUT = float(os.environ.get("ACP_PROMPT_TIMEOUT", "180"))
DEFAULT_CWD = os.environ.get("ACP_DEFAULT_CWD", os.path.expanduser("~"))

CLIENT_CAPS = {"fs": {"readTextFile": True, "writeTextFile": True},
               "terminal": True}


class AcpError(Exception):
    pass


class AcpClient:
    """A long-lived `opencode acp` child, spoken to over JSON-RPC stdio."""

    def __init__(self):
        self.proc = None
        self.lock = threading.RLock()
        self._next_id = 0
        self._pending = {}          # id -> threading.Event
        self._results = {}          # id -> (ok, payload)
        self._notes = []            # notifications, drained per call
        self._notes_lock = threading.Lock()
        self.info = None            # initialize result

    # ---- lifecycle -------------------------------------------------
    def _spawn(self):
        self.proc = subprocess.Popen(
            [OPENCODE, "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._read_loop, daemon=True,
                         name="acp-reader").start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                mid = msg.get("id")
                if mid is not None and mid in self._pending:
                    # ok == "this was a result, not an error". Storing
                    # ("error" in msg) inverted it and turned every
                    # successful response into a failure.
                    failed = "error" in msg
                    self._results[mid] = (not failed, msg.get("error") if failed
                                          else msg.get("result"))
                    self._pending[mid].set()
                elif msg.get("method"):        # notification
                    with self._notes_lock:
                        self._notes.append(msg)
                    if len(self._notes) > 500:      # bound memory
                        del self._notes[:-200]
        except (ValueError, OSError):
            pass
        finally:
            for ev in list(self._pending.values()):
                ev.set()                      # unblock waiters on death

    def ensure(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.info = None
            self._spawn()
        # keep the handshake result: protocol version and capabilities are
        # read from it by every other tool
        self.info = self.rpc("initialize", {"protocolVersion": 1,
                                            "clientCapabilities": CLIENT_CAPS},
                             HANDSHAKE_TIMEOUT) or {}
        self.rpc("notifications/initialized", None, 10, notify=True)

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            self.info = None

    # ---- rpc -------------------------------------------------------
    def rpc(self, method, params=None, timeout=CALL_TIMEOUT, notify=False):
        self.ensure_up()
        with self.lock:
            self._next_id += 1
            rid = self._next_id
        frame = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            frame["params"] = params
        if notify:
            frame.pop("id")
            self._write(frame)
            return None
        ev = threading.Event()
        self._pending[rid] = ev
        self._write(frame)
        if not ev.wait(timeout):
            self._pending.pop(rid, None)
            raise AcpError(f"{method} timed out after {timeout:.0f}s")
        self._pending.pop(rid, None)
        ok, payload = self._results.pop(rid, (False, None))
        if not ok:
            raise AcpError(f"{method} error: {json.dumps(payload)[:300]}")
        return payload

    def _write(self, frame):
        try:
            self.proc.stdin.write(json.dumps(frame) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise AcpError(f"ACP child is not writable: {exc}") from exc

    def ensure_up(self):
        if self.proc is None or self.proc.poll() is not None:
            self.ensure()

    def drain_notes(self):
        with self._notes_lock:
            out, self._notes = self._notes, []
        return out

    @staticmethod
    def reply_text(notes):
        """Pull assistant text out of ACP session/update notifications."""
        chunks = []
        for n in notes:
            params = n.get("params") or {}
            upd = params.get("update") or params
            if not isinstance(upd, dict):
                continue
            kind = upd.get("sessionUpdate") or upd.get("kind")
            if kind in ("agent_message_chunk", "agent_message"):
                content = upd.get("content") or {}
                txt = content.get("text") if isinstance(content, dict) else None
                if txt:
                    chunks.append(txt)
        return "".join(chunks)


CLIENT = AcpClient()


def tool_acp_status():
    CLIENT.ensure()
    info = CLIENT.info or {}
    caps = info.get("agentCapabilities") or {}
    agent = info.get("agentInfo") or {}
    try:
        sessions = (CLIENT.rpc("session/list", {}, 30) or {}).get("sessions") or []
        session_err = None
    except AcpError as exc:
        sessions, session_err = [], str(exc)
    return {
        "connected": True,
        "protocol_version": info.get("protocolVersion"),
        "agent": f"{agent.get('name', '?')} {agent.get('version', '')}".strip(),
        "session_count": len(sessions),
        "session_list_error": session_err,
        "capabilities": caps,
        "auth_methods": [a.get("id") for a in (info.get("authMethods") or [])],
    }


def tool_acp_capabilities():
    CLIENT.ensure()
    return {"protocol_version": (CLIENT.info or {}).get("protocolVersion"),
            "capabilities": (CLIENT.info or {}).get("agentCapabilities") or {}}


def tool_acp_sessions():
    res = CLIENT.rpc("session/list", {}, 40) or {}
    out = []
    for s in res.get("sessions") or []:
        out.append({"sessionId": s.get("sessionId"), "cwd": s.get("cwd"),
                    "title": s.get("title"),
                    "updated": s.get("updatedAt") or s.get("updated_at")})
    return {"count": len(out), "sessions": out}


def tool_acp_new_session(cwd=None, mcp_servers=None):
    path = os.path.abspath(os.path.expanduser(cwd or DEFAULT_CWD))
    if not os.path.isdir(path):
        raise AcpError(f"not a directory: {path}")
    res = CLIENT.rpc("session/new", {"cwd": path,
                                     "mcpServers": mcp_servers or []}, 60)
    return {"sessionId": (res or {}).get("sessionId"), "cwd": path,
            "modes": (res or {}).get("modes"),
            "models": (res or {}).get("models")}


def tool_acp_prompt(prompt, session_id=None, cwd=None):
    text = str(prompt or "").strip()
    if not text:
        raise AcpError("prompt is required")
    if session_id:
        sid = session_id
    else:
        sid = tool_acp_new_session(cwd)["sessionId"]
    CLIENT.drain_notes()
    res = CLIENT.rpc("session/prompt",
                     {"sessionId": sid,
                      "prompt": [{"type": "text", "text": text}]},
                     PROMPT_TIMEOUT)
    notes = CLIENT.drain_notes()
    return {"sessionId": sid, "stopReason": (res or {}).get("stopReason"),
            "reply": CLIENT.reply_text(notes)}


def tool_acp_restart():
    CLIENT.stop()
    CLIENT.ensure()
    return {"restarted": True, "protocol_version": (CLIENT.info or {}).get("protocolVersion")}


TOOLS = [
    {"name": "acp_status", "description":
     "ACP connection state: handshake result, negotiated protocol version, "
     "agent name, live ACP session count and capabilities.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "acp_capabilities", "description":
     "The Agent Client Protocol capability map this agent negotiated.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "acp_sessions", "description":
     "List ACP sessions on this machine (id, cwd, title).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "acp_new_session", "description":
     "Open a new ACP session rooted at a directory.",
     "inputSchema": {"type": "object", "properties": {
         "cwd": {"type": "string", "description": "working directory"}}}},
    {"name": "acp_prompt", "description":
     "Send a prompt to an ACP session and return the agent's reply text.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"},
         "session_id": {"type": "string",
                        "description": "reuse a session; omit to create one"},
         "cwd": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "acp_restart", "description":
     "Restart the ACP child and redo the handshake.",
     "inputSchema": {"type": "object", "properties": {}}},
]
HANDLERS = {"acp_status": tool_acp_status, "acp_capabilities": tool_acp_capabilities,
            "acp_sessions": tool_acp_sessions, "acp_new_session": tool_acp_new_session,
            "acp_prompt": tool_acp_prompt, "acp_restart": tool_acp_restart}


def _respond(req, result):
    sys.stdout.write(json.dumps(
        {"jsonrpc": "2.0", "id": req.get("id"), "result": result}) + "\n")
    sys.stdout.flush()


def handle(msg):
    if isinstance(msg, (str, bytes, bytearray)):
        try:
            msg = json.loads(msg)
        except ValueError:
            return
    if not isinstance(msg, dict):
        return
    req, m = msg, msg.get("method", "")
    if m == "initialize":
        pv = req.get("params", {}).get("protocolVersion", "2025-06-18")
        _respond(req, {"protocolVersion": pv, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "acp", "version": VERSION}})
    elif m == "ping":
        _respond(req, {})
    elif m == "tools/list":
        _respond(req, {"tools": TOOLS})
    elif m in ("resources/list", "prompts/list"):
        _respond(req, {"resources" if m[0] == "r" else "prompts": []})
    elif m == "tools/call":
        params = req.get("params", {})
        name, args = params.get("name", ""), params.get("arguments", {}) or {}
        if name not in HANDLERS:
            _respond(req, {"content": [{"type": "text", "text": f"unknown tool: {name}"}],
                           "isError": True})
            return
        try:
            out = HANDLERS[name](**args)
            _respond(req, {"content": [{"type": "text",
                                        "text": json.dumps(out, ensure_ascii=False)}],
                           "isError": False})
        except AcpError as exc:
            _respond(req, {"content": [{"type": "text", "text": f"{name}: {exc}"}],
                           "isError": True})
        except Exception as e:  # noqa: BLE001 - a tool must never kill MCP
            _respond(req, {"content": [{"type": "text", "text": f"{name} error: {e}"}],
                           "isError": True})
    elif m.startswith("notifications/"):
        pass
    elif req.get("id") is not None:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"],
                                     "error": {"code": -32601,
                                               "message": f"method not found: {m}"}}) + "\n")
        sys.stdout.flush()


def _bye(_s, _f):
    CLIENT.stop()
    raise SystemExit(0)


def main():
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _bye)
        except (OSError, ValueError):
            pass
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
                    headers[k.decode().strip().lower()] = v.decode().strip()
            body = stdin.read(int(headers.get("content-length", "0")) or b"")
        else:
            body = stripped
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            continue
        if isinstance(msg, list):
            for one in msg:
                handle(one)
        else:
            handle(msg)
    CLIENT.stop()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        CLIENT.stop()
