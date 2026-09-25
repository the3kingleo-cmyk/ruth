"""Ruth's local interface: a tiny offline web app served by the standard library.

Runs on any machine with Python. It binds to this computer only
(127.0.0.1) and loads nothing from the internet. The page talks to one mind
through a few JSON endpoints.
"""
from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .. import __version__, dream, paths
from ..config import BrainConfig
from ..engine import Brain
from ..mind import Mind
from ..senses import Cochlea, EventRetina, Voice, byte_code

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class Life:
    """One mind plus the live sensory organs attached to it."""

    def __init__(self, home: str | None = None):
        self.home = home or paths.home()
        state = os.path.join(self.home, "brain.npz")
        self.mind = Mind(Brain.load_or_create(state, BrainConfig()), home=self.home)
        self.lock = threading.Lock()
        self.ear = None
        self.eye = None
        self.surprise = deque(maxlen=240)
        self.saved = time.time()
        self.touched = time.time()
        self.replayed = 0
        self.stop = threading.Event()

    def background(self, idle_replay: float = 20.0, idle_sleep: float = 900.0) -> None:
        """Hippocampus -> neocortex while nobody is talking: short replays after
        a quiet moment, a full night's sleep after a long quiet."""
        slept_at = time.time()
        while not self.stop.wait(5.0):
            idle = time.time() - self.touched
            if idle < idle_replay:
                continue
            with self.lock:
                if idle > idle_sleep and self.touched > slept_at:
                    dream.sleep(self.mind)
                    slept_at = time.time()
                    self.autosave(force=True)
                else:
                    self.replayed += self.mind.brain.replay(32)

    def note(self) -> None:
        s = self.mind.brain.last.get("surprise")
        if s is not None:
            self.surprise.append(round(float(s), 4))

    def autosave(self, force: bool = False) -> None:
        if force or time.time() - self.saved > 30:
            self.mind.save()
            self.saved = time.time()

    # -- endpoints (called with the lock held) --------------------------
    def state(self, _body):
        m, b = self.mind, self.mind.brain
        g = b.graph()["nodes"]
        return {"version": __version__, "identity": m.identity, "moments": b.steps, "sleeps": b.sleeps,
                "neurons": b.cfg.neurons, "grown": b.cfg.grown, "synapses": g["cortex"]["synapses"],
                "cell": b.cfg.cell, "background_replays": self.replayed,
                "latent_surprise": round(float(b.last.get("latent_surprise", 0.0)), 4),
                "working": b.memory.w_n, "longterm": b.memory.l_n,
                "conversations": m.temperament.age, "cues": len(m.cues),
                "temperament": m.temperament.snapshot(),
                "caution_gate": round(m.temperament.inhibit_level, 3),
                "valence": round(m.predicted_valence(), 3),
                "activity": np.round(b.core.x, 3).tolist(),
                "field": np.round(b.ssm.o, 3).tolist(),
                "surprise": list(self.surprise),
                "senses": list(b.cfg.senses)}

    def talk(self, body):
        text = str(body.get("text", ""))[:4000]
        if body.get("private"):
            r = self.mind.teach(text + "\n", private=True)
            self.autosave()
            return {"kept": r["private_bytes"]}
        reply = self.mind.converse(text)
        self.note()
        self.autosave()
        return {**reply, "considered": self.mind.last_deliberation}

    def teach(self, body):
        r = self.mind.teach(str(body.get("text", ""))[:200000], private=bool(body.get("private")))
        self.note()
        self.autosave()
        return r

    def feedback(self, body):
        return self.mind.feedback(bool(body.get("good")))

    def sleep(self, _body):
        r = dream.sleep(self.mind)
        self.autosave(force=True)
        return r

    def dreams(self, _body):
        return dream.journal(self.mind, 8)

    def hear(self, body):
        rate = int(body.get("rate", 16000))
        if self.ear is None or self.ear.sr != rate:
            self.ear = Cochlea(rate, self.mind.brain.cfg.senses["ears"])
        samples = np.asarray(body.get("samples", []), dtype=float)[: rate * 2]
        n = 0
        for feat, dt in self.ear.process(samples):
            self.mind.brain.step({"ears": feat}, dt / 0.01)
            self.note()
            n += 1
        return {"moments": n}

    def see(self, body):
        w, h = int(body.get("w", 0)), int(body.get("h", 0))
        px = np.asarray(body.get("pixels", []), dtype=float)
        if w * h == 0 or px.size != w * h or w * h > 128 * 128:
            return {"error": "expected w*h grey pixels in 0..1"}
        if self.eye is None:
            self.eye = EventRetina(grid=int(round((self.mind.brain.cfg.senses["eyes"] // 2) ** 0.5)))
        ev, dt = self.eye.observe(px.reshape(h, w), float(body.get("t", time.time())))
        if dt > 0:
            self.mind.brain.step({"eyes": ev}, dt / 0.01)
            self.note()
        return {"events": int(round(ev.sum() * 100))}

    def voice(self, body):
        text = str(body.get("text", ""))[:400].encode("utf-8")
        b = self.mind.brain
        snap = b.snapshot()
        curves = []
        try:
            for ch in text:
                b.step({"text": byte_code(ch)}, 1.0, learn=False)
                curves.append(Voice.params_from_latent(3.0 * b.core.x[b.core.motor][:4]))
        finally:
            b.restore(snap)
        if not curves:
            return {"rate": 16000, "samples": []}
        wav = Voice(16000).render(np.array(curves), frame_seconds=0.06)
        return {"rate": 16000, "samples": np.round(wav, 4).tolist()}


ROUTES = {("GET", "/api/state"): "state", ("GET", "/api/dreams"): "dreams",
          ("POST", "/api/talk"): "talk", ("POST", "/api/teach"): "teach",
          ("POST", "/api/feedback"): "feedback", ("POST", "/api/sleep"): "sleep",
          ("POST", "/api/hear"): "hear", ("POST", "/api/see"): "see",
          ("POST", "/api/voice"): "voice"}


def make_handler(life: Life):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Ruth/" + __version__

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _local(self) -> bool:
            # only pages served by this app on this machine may drive her
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS or host == self.server.server_address[0]

        def _route(self, method):
            path = self.path.split("?", 1)[0]
            if not self._local():
                return self._send(403, {"error": "local use only"})
            name = ROUTES.get((method, path))
            if name is None:
                if method == "GET":
                    return self._static(path)
                return self._send(404, {"error": "no such thing"})
            body = {}
            if method == "POST":
                if "application/json" not in (self.headers.get("Content-Type") or ""):
                    return self._send(415, {"error": "json only"})
                n = min(int(self.headers.get("Content-Length") or 0), 8_000_000)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._send(400, {"error": "bad json"})
            with life.lock:
                life.touched = time.time()
                try:
                    return self._send(200, getattr(life, name)(body))
                except Exception as e:  # keep her alive whatever a request does
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def _static(self, path):
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.realpath(os.path.join(STATIC, rel))
            if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            with open(full, "rb") as f:
                return self._send(200, f.read(), TYPES.get(os.path.splitext(full)[1], "text/plain"))

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return Handler


def serve(host: str = "127.0.0.1", port: int = 7438, open_browser: bool = True,
          home: str | None = None) -> None:
    life = Life(home)
    threading.Thread(target=life.background, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(life))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{httpd.server_address[1]}/"
    print(f"Ruth is awake at {url}  (Ctrl-C to let her rest)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        life.stop.set()
        with life.lock:
            life.autosave(force=True)
        httpd.server_close()
        print("Ruth saved her mind and is resting.")
