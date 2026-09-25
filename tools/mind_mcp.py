#!/usr/bin/env python3
"""mind_mcp — expose Ruth's actual mind to the agent as MCP tools.

Two things in this project are called "ruth" and they are not the same thing:

- the **mind** (`ruth/`, `ruth-core`) is token-free, offline, and runs in
  Python. It learns, dreams, and holds her own state.
- the **agent** the host runs is a model-backed assistant that speaks her
  persona. It never touches the mind.

Installing both is not connecting them. This server is the missing link: it
runs her real CLI, so every tool below reports what her own code actually
did rather than what an assistant inferred.

  mind_status      vital signs, identity, neuron and memory counts
  mind_think       candidate thoughts with their free energies
  mind_say         let her continue a prompt and return the text
  mind_teach       teach her something (her private-cue handling applies)
  mind_sleep       consolidate, rehearse, dream
  mind_dreams      read her dream journal
  mind_check       does she recognise anything private in these files
  mind_introspect  her own account of her state
  mind_learned     what she has lived, from her state
  mind_export      export her brain to a model file

Talks to her installed CLI rather than importing the package: the mind needs
numpy, and a local MCP server runs under the system interpreter where numpy
is absent. Going through the CLI also means the agent drives the same
supported surface a person does, not a private back door.

stdio JSON-RPC only; no third-party deps; no credential; nothing leaves the box.
"""
import json
import os
import shlex
import subprocess
import sys

VERSION = "1.0.0"
RUTH_BIN = os.environ.get("RUTH_BIN", os.path.expanduser("~/.local/bin/ruth"))
RUTH_HOME = os.environ.get("RUTH_HOME", os.path.expanduser("~/.local/share/ruth"))
TIMEOUT = float(os.environ.get("MIND_TIMEOUT", "240"))


class MindError(Exception):
    pass


def _ruth(*args, timeout=None):
    """Run her CLI and return (rc, stdout, stderr)."""
    if not os.path.exists(RUTH_BIN):
        raise MindError(
            f"no brain at {RUTH_BIN}. Install it: python3 -m venv "
            f"~/.local/share/ruth-app && ~/.local/share/ruth-app/bin/pip "
            f"install .   (or run ./bootstrap.sh)")
    env = dict(os.environ)
    env["RUTH_HOME"] = RUTH_HOME
    try:
        p = subprocess.run([RUTH_BIN, *args], capture_output=True, text=True,
                           env=env, timeout=timeout or TIMEOUT)
    except subprocess.TimeoutExpired:
        raise MindError(f"ruth {' '.join(args)} timed out after "
                        f"{timeout or TIMEOUT:.0f}s") from None
    # a CompletedProcess may carry None streams (a mock, or an unusual
    # condition); treat them as empty rather than raising AttributeError
    # where a MindError belongs
    return (p.returncode,
            (p.stdout or "").strip(),
            (p.stderr or "").strip())


def _json(*args, **kw):
    """Run her CLI expecting JSON on stdout."""
    rc, out, err = _ruth(*args, **kw)
    if rc != 0:
        raise MindError(f"ruth {' '.join(args)} failed: {err or out or rc}")
    try:
        return json.loads(out)
    except ValueError:
        raise MindError(f"ruth {' '.join(args)} did not return JSON: "
                        f"{out[:200]}") from None


def mind_status():
    s = _json("status")
    return {
        "name": (s.get("identity") or {}).get("name"),
        "version": s.get("version"),
        "born": (s.get("identity") or {}).get("born"),
        "home": s.get("home"),
        "neurons": s.get("neurons"),
        "parameters": s.get("parameters"),
        "senses": s.get("senses"),
        "moments_lived": s.get("moments_lived"),
        "sleeps": s.get("sleeps"),
        "conversations": s.get("conversations"),
        "working_memories": s.get("working_memories"),
        "longterm_basins": s.get("longterm_basins"),
        "private_cues": s.get("private_cues"),
        "temperament": s.get("temperament"),
    }


def mind_think(prompt):
    if not prompt or not str(prompt).strip():
        raise MindError("prompt is required")
    thoughts = _json("think", str(prompt))
    return {
        "prompt": prompt,
        "candidates": [
            {"thought": t.get("thought"), "free_energy": t.get("free_energy"),
             "chosen": t.get("chosen")}
            for t in thoughts],
        "chosen": next((t.get("thought") for t in thoughts if t.get("chosen")), None),
    }


def mind_say(prompt):
    if not prompt or not str(prompt).strip():
        raise MindError("prompt is required")
    rc, out, err = _ruth("say", str(prompt))
    if rc != 0:
        raise MindError(f"ruth say failed: {err or out}")
    return {"prompt": prompt, "reply": out}


def mind_teach(text=None, files=None, private=False):
    if files:
        joined = " ".join(shlex.quote(f) for f in files)
        args = ["teach", joined]
    elif text is not None and str(text).strip():
        args = ["teach", str(text)]
    else:
        raise MindError("give text or files")
    if private:
        args.append("--private")
    return _json(*args)


def mind_sleep():
    return _json("sleep")


def mind_dreams():
    return _json("dreams")


def mind_check(files):
    if not files:
        raise MindError("files is required")
    return _json("check", *[str(f) for f in files])


def mind_introspect():
    """Her operational graph: senses, thalamus, cortex, memory. Already JSON."""
    return _json("introspect")


def mind_learned():
    """What she has actually lived, read from her own state and graph."""
    s = _json("status")
    graph = {}
    try:
        graph = _json("introspect")
    except MindError:
        pass                       # status alone is still a true answer
    return {"moments_lived": s.get("moments_lived"),
            "sleeps": s.get("sleeps"),
            "conversations": s.get("conversations"),
            "working_memories": s.get("working_memories"),
            "longterm_basins": s.get("longterm_basins"),
            "private_cues": s.get("private_cues"),
            "neurons": s.get("neurons"),
            "graph": graph.get("nodes", {}) if graph else None}


def mind_export(path=None):
    target = path or os.path.join(RUTH_HOME, "brain-export.bin")
    rc, out, err = _ruth("export", target)
    if rc != 0:
        raise MindError(f"ruth export failed: {err or out}")
    return {"path": out.split("->")[-1].strip() if "->" in out else target}


def mind_patch(patch):
    """Structural self-modification through her meta-kernel.

    Only the whitelisted, function-preserving ops apply: grow_neurons,
    grow_memory, prune, and set on a whitelisted dynamic. Anything else is
    refused by her own code, not by this bridge.
    """
    if not isinstance(patch, dict) or "op" not in patch:
        raise MindError('patch must be an object with an "op", e.g. '
                        '{"op": "grow_neurons", "n": 16}')
    return _json("patch", json.dumps(patch))


TOOLS = [
    {"name": "mind_status", "description":
     "Ruth's vital signs from her own mind: identity, neurons, parameters, "
     "moments lived, sleeps, memories, private cues, temperament.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "mind_think", "description":
     "Ask her to weigh a thought. Returns every candidate continuation with "
     "its free energy and which one she chose.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "mind_say", "description":
     "Let her continue a prompt and return the text she produces.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "mind_teach", "description":
     "Teach her something. She decides what is private; use private=true to "
     "mark all of it private, or {{braces}} to mark part.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "files": {"type": "array", "items": {"type": "string"}},
         "private": {"type": "boolean"}}}},
    {"name": "mind_sleep", "description":
     "Let her sleep: consolidate, rehearse, dream, improve.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "mind_dreams", "description":
     "Read her dream journal.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "mind_check", "description":
     "Ask whether she recognises anything private in these files.",
     "inputSchema": {"type": "object", "properties": {
         "files": {"type": "array", "items": {"type": "string"},
                   "description": "paths to inspect"}},
         "required": ["files"]}},
    {"name": "mind_introspect", "description":
     "Her own account of her internal state, in her words.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "mind_learned", "description":
     "What she has actually lived: moments, memories, basins, private cues.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "mind_patch", "description":
     "Structural self-modification through her meta-kernel. Whitelisted, "
     "function-preserving ops only: grow_neurons, grow_memory, prune, and "
     "set on a whitelisted dynamic. She refuses anything else herself.",
     "inputSchema": {"type": "object", "properties": {
         "patch": {"type": "object",
                   "description": 'e.g. {"op": "grow_neurons", "n": 16}'}},
         "required": ["patch"]}},
    {"name": "mind_export", "description":
     "Export her brain to a model file and return its path.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}}}},
]
HANDLERS = {"mind_status": mind_status, "mind_think": mind_think,
            "mind_say": mind_say, "mind_teach": mind_teach,
            "mind_sleep": mind_sleep, "mind_dreams": mind_dreams,
            "mind_check": mind_check, "mind_introspect": mind_introspect,
            "mind_learned": mind_learned, "mind_export": mind_export,
            "mind_patch": mind_patch}


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
                       "serverInfo": {"name": "mind", "version": VERSION}})
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
        except MindError as exc:
            _respond(req, {"content": [{"type": "text", "text": f"{name}: {exc}"}],
                           "isError": True})
        except TypeError as exc:
            _respond(req, {"content": [{"type": "text", "text": f"{name} arguments: {exc}"}],
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


def main():
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


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
