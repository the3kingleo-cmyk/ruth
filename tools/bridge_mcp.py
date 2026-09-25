#!/usr/bin/env python3
"""browser-sense MCP server (stdio) - exposes a local browser-sensor bridge to
the agent as MCP tools.

The capture runs in a real browser on whatever machine the operator chooses: a
page opens BRIDGE_URL, getUserMedia grants camera+mic, frames POST /frame and
4s WAV slices POST /audio. This server lets the agent see/listen/speak through
that live connection.

Every deployment-specific value (URL, port, paths, python) is an environment
variable; nothing about any particular operator's machine is baked in.

tools:
  sense_status  -> bridge health, who is streaming, frame/audio counters
  sense_look    -> latest camera frame -> save jpg + describe via cloud vision
  sense_listen  -> transcribe the newest unheard mic slice (your voice)
  sense_speak   -> push text to the open browser page -> spoken on speakers

stdio JSON-RPC only; no third-party deps.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8080")
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", os.path.expanduser("~/.local/status/audio")))
SHOTS = Path(os.environ.get("SHOTS_DIR", os.path.expanduser("~/.local/status/shots")))
SEEN = Path(os.environ.get("SEEN_FILE", os.path.expanduser("~/.local/status/bridge_seen.txt")))
PYTHON = os.environ.get("SENSE_PYTHON", sys.executable)
EAR_STT = os.environ.get("EAR_STT", os.path.expanduser("~/.local/bin/ear_stt.py"))
EYES_LOOK = os.environ.get("EYES_LOOK", os.path.expanduser("~/.local/bin/eyes_look.py"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SHOTS.mkdir(parents=True, exist_ok=True)

# --- live self-reload: apply edits to this file without an MCP/service restart.
SELF = os.path.realpath(__file__)
CODE_MTIME = os.path.getmtime(SELF)
RELOAD_POLL = float(os.environ.get("BRIDGE_RELOAD_POLL", "10"))
BRIDGE_MCP_VERSION = "1.1.0"
_LOG = Path(os.environ.get("BRIDGE_MCP_LOG",
                           os.path.expanduser("~/.local/status/bridge_mcp.log")))
_FLIGHT_LOCK = __import__("threading").Lock()
_IN_FLIGHT = 0


def _blog(msg):
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.time():.3f}] {msg}\n")
    except OSError:
        return


def _flight_enter():
    global _IN_FLIGHT
    with _FLIGHT_LOCK:
        _IN_FLIGHT += 1


def _flight_exit():
    global _IN_FLIGHT
    with _FLIGHT_LOCK:
        _IN_FLIGHT -= 1


def _live_reload_watch():
    """re-exec in place on code change: pid + stdio pipes survive, MCP never drops."""
    seen = CODE_MTIME
    while True:
        time.sleep(RELOAD_POLL)
        try:
            mt = os.path.getmtime(SELF)
        except OSError:
            continue
        if mt == seen:
            continue
        time.sleep(RELOAD_POLL)          # debounce partial writes
        try:
            mt2 = os.path.getmtime(SELF)
        except OSError:
            continue
        if mt2 != mt:
            seen = mt2
            continue
        if _IN_FLIGHT:                   # never yank a live tool call
            _blog("code changed but a tool call is in flight -> deferred")
            continue
        seen = mt2
        _blog(f"code changed (mtime {mt2:.3f}) -> live reload via execv")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as e:
            _blog(f"live reload execv failed, staying on current code: {e}")


def _http(path, method="GET", body=None, ctype=None, timeout=20):
    data = body.encode() if isinstance(body, str) else body
    headers = {"X-Iris-Sense": "mcp"} if ctype is None else {"Content-Type": ctype, "X-Iris-Sense": "mcp"}
    req = urllib.request.Request(BRIDGE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError as e:
        # connection refused / timeout / DNS: URLError subclasses OSError
        raise RuntimeError(f"bridge unreachable at {BRIDGE}: {e}") from e


def _jsonget(path):
    st, raw = _http(path)
    if st != 200:
        raise RuntimeError(f"bridge {path} -> {st}: {raw[:200]!r}")
    return json.loads(raw)


def sense_status():
    st = _jsonget("/status")
    au = _jsonget("/audio/status")
    auds = sorted(AUDIO_DIR.glob("bridge_*.wav"))
    newest = auds[-1].name if auds else None
    lines = [
        f"bridge version: {st.get('version')}",
        f"status: {st.get('status')}  has_frame: {st.get('has_frame')}",
        f"frames_total: {st.get('frames_total')}  last_frame_age_ms: {st.get('frame_age_ms')}",
        f"audio_slices: {st.get('audio_slices')}  newest_slice: {newest or 'none'}",
        f"audio_status: {au}",
        f"streaming client: os={st.get('client', {}).get('os')}  hello_total={st.get('client', {}).get('hello_total')}",
        f"client_ua: {st.get('client', {}).get('ua')}",
    ]
    has = st.get("has_frame")
    if not has:
        lines.append(f"NO LIVE FEED. Open {BRIDGE} in a browser and grant camera+microphone "
                     "Chrome and click 'Grant Camera + Microphone'.")
    else:
        os_ = (st.get("client", {}).get("os") or "").lower()
        if "cros" in os_ or "chromeos" in os_ or "linux x86_64" in os_ and st.get("client", {}).get("ua", "").find("CrOS") >= 0:
            lines.append("feed source: browser (real camera+mic) - LIVE")
        else:
            lines.append(f"WARNING: feed client os={os_} - if this is not a real desktop browser, the camera is likely the fake device.")
    return "\n".join(lines)


def sense_look():
    st, raw = _http("/frame")
    if st != 200:
        raise RuntimeError(f"no frame yet (bridge {st}). Open {BRIDGE} in a browser and grant camera+mic.")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jpg = SHOTS / f"mcp_look_{ts}.jpg"
    jpg.write_bytes(raw)
    info = _jsonget("/status")
    caption = None
    try:
        r = subprocess.run([sys.executable, str(EYES_LOOK), str(jpg)],
                           capture_output=True, text=True, timeout=110,
                           check=False)
        caption = r.stdout.strip() or r.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        caption = f"vision describe unavailable: {e}"
    out = [
        f"frame saved: {jpg} ({jpg.stat().st_size} bytes)",
        f"frame_age_ms: {info.get('frame_age_ms')}  frames_total: {info.get('frames_total')}",
    ]
    if caption:
        out.append("vision: " + caption)
    return "\n".join(out)


def sense_listen():
    auds = sorted(AUDIO_DIR.glob("bridge_*.wav"))
    if not auds:
        raise RuntimeError(f"no mic slices yet. Open {BRIDGE} in a browser and grant camera+microphone.")
    seen = SEEN.read_text().strip() if SEEN.exists() else ""
    newest = auds[-1]
    if newest.name == seen:
        # look one slice back in case a slice was skipped
        cand = [a for a in reversed(auds) if a.name != seen][:2]
        if not cand:
            return f"listening; no new voice slice yet (last heard up to {newest.name}). Say something."
        newest = cand[0]
    r = subprocess.run([str(PYTHON), str(EAR_STT), str(newest)],
                       capture_output=True, text=True, timeout=180,
                       check=False)
    text = "\n".join(l.strip() for l in r.stdout.splitlines() if l.strip())
    SEEN.write_text(newest.name)
    if not text:
        return f"heard {newest.name}: (no speech detected in this slice)"
    return f"you said ({newest.name}):\n{text}"


def sense_speak(text):
    text = (text or "").strip()[:400]
    if not text:
        raise RuntimeError("missing text")
    st, raw = _http("/outbox", method="POST", body=json.dumps({"text": text}), ctype="application/json")
    if st != 200:
        raise RuntimeError(f"outbox -> {st}: {raw[:200]!r}")
    d = json.loads(raw)
    return f"queued for the open browser page to speak (outbox id {d.get('id')}): {text}"


TOOLS = [
    {
        "name": "sense_status",
        "description": "Live status of the browser sensor bridge: has a camera frame, frame age, "
                       "how many audio slices arrived, and which browser/os is streaming in.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sense_look",
        "description": "See through the live browser camera. Grabs the newest frame, saves it as a "
                       "jpg on the box, and returns its path plus a vision-model description.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sense_listen",
        "description": "Hear through the live browser microphone. Transcribes the newest unheard "
                       "voice slice (faster-whisper) and returns what was said.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sense_speak",
        "description": "Speak through the open browser page's speakers. Push text to the bridge "
                       "outbox; the live page picks it up within ~1.2s and speaks it.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to speak aloud"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
]

HANDLERS = {
    "sense_status": sense_status,
    "sense_look": sense_look,
    "sense_listen": sense_listen,
    "sense_speak": sense_speak,
}


def _respond(req, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}) + "\n")
    sys.stdout.flush()


def _error(req, code, message):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req.get("id"),
                                 "error": {"code": code, "message": message}}) + "\n")
    sys.stdout.flush()


def _call_tool(name, args):
    if name not in HANDLERS:
        raise RuntimeError(f"unknown tool: {name}")
    return HANDLERS[name](**args) if isinstance(args, dict) else HANDLERS[name]()


def handle(msg):
    req = json.loads(msg)
    m = req.get("method", "")
    if m == "initialize":
        pv = req.get("params", {}).get("protocolVersion", "2025-06-18")
        _respond(req, {"protocolVersion": pv, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "browser-sense", "version": "2.1.0"}})
    elif m == "ping":
        _respond(req, {})
    elif m == "tools/list":
        _respond(req, {"tools": TOOLS})
    elif m == "resources/list":
        _respond(req, {"resources": []})
    elif m == "prompts/list":
        _respond(req, {"prompts": []})
    elif m == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        try:
            _flight_enter()
            try:
                out = _call_tool(name, args)
            finally:
                _flight_exit()
            _respond(req, {"content": [{"type": "text", "text": str(out)}], "isError": False})
        except Exception as e:  # noqa: BLE001 - a tool must never kill MCP
            _respond(req, {"content": [{"type": "text", "text": f"{name} error: {e}"}], "isError": True})
    elif m.startswith("notifications/"):
        pass
    else:
        _error(req, -32601, f"method not found: {m}")


def main():
    if os.environ.get("BRIDGE_RELOAD", "1") != "0":
        __import__("threading").Thread(
            target=_live_reload_watch, daemon=True, name="bridge-reload-watch").start()
    _blog(f"bridge_mcp start pid={os.getpid()} version={BRIDGE_MCP_VERSION} "
          f"code_mtime={CODE_MTIME:.3f} live_reload=on")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(line)
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill MCP
            sys.stderr.write(f"[bridge_mcp] bad frame: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()