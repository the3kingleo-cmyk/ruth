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
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .. import __version__, dream, paths
from ..owner import MindBusy, owner
from ..config import BrainConfig
from ..engine import Brain
from ..mind import Mind
from ..senses import Cochlea, EventRetina, Voice, byte_code

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# How long a request waits for her before answering "she is dreaming". Each
# step of a night is short (seconds), and between steps the dream hands the
# lock over, so a request normally gets in well within this.
LOCK_WAIT = float(os.environ.get("RUTH_LOCK_WAIT", "20"))


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
        self.dreaming = False
        self.stop = threading.Event()

    def background(self, idle_replay: float = 20.0, idle_sleep: float = 900.0) -> None:
        """Hippocampus -> neocortex while nobody is talking: short replays after
        a quiet moment, a full night's sleep after a long quiet.

        Nothing here may end the thread. An exception used to escape the loop,
        after which she never replayed or slept again and nothing said so.

        A night is long for a mind with tens of thousands of lived moments, so
        it is not run under one unbroken hold of the lock: between each step
        the dream lets any waiting request in, and wakes up if someone is here.
        """
        slept_at = time.time()

        def quiet() -> bool:
            return time.time() - self.touched >= idle_replay

        def pause() -> bool:
            # Hand the lock to whoever is waiting, then continue only if the
            # room is still quiet.
            self.lock.release()
            try:
                time.sleep(0.05)
            finally:
                self.lock.acquire()
            return quiet() and not self.stop.is_set()

        while not self.stop.wait(5.0):
            if not quiet():
                continue
            try:
                with self.lock:
                    if not quiet():  # someone arrived while we waited
                        continue
                    idle = time.time() - self.touched
                    if idle > idle_sleep and self.touched > slept_at:
                        self.dreaming = True
                        try:
                            dream.sleep(self.mind, pause=pause)
                        finally:
                            self.dreaming = False
                        slept_at = time.time()
                        self.autosave(force=True)
                    else:
                        self.replayed += self.mind.brain.replay(32)
            except Exception:
                traceback.print_exc()
                self.stop.wait(30.0)

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
                "dreaming": self.dreaming,
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

    def soul(self, body):
        """Let an external canonical source (the private memory repository)
        set who she is and what she knows.

        Her identity and her taught experience both live in files that other
        programs also write, so they arrive through this endpoint rather than
        by opening her home directly: the app holds the exclusive owner lock,
        and a second writer would clobber whatever is in memory.

        GET  reports what she currently carries.
        POST {"identity": {...}}  merges keys into her identity.
        POST {"teach": "..."}     teaches text privately (lived, not public).
        """
        if not body:
            m = self.mind
            return {"identity": m.identity,
                    "temperament": m.temperament.snapshot(),
                    "history_bytes": len(m.lived_history()),
                    "journal": str(m.journal_path())}

        out = {}
        ident = body.get("identity")
        if isinstance(ident, dict) and ident:
            m = self.mind
            # Never let a remote source rewrite her birth. She was born once.
            born = m.identity.get("born")
            m.identity.update({str(k): v for k, v in ident.items() if v is not None})
            if born:
                m.identity["born"] = born
            out["identity"] = m.identity

        teach = body.get("teach")
        if isinstance(teach, str) and teach.strip():
            r = self.mind.teach(teach, private=bool(body.get("private", True)))
            out["taught"] = {k: v for k, v in r.items() if k != "private_bytes"}
            out["private_bytes"] = r.get("private_bytes")

        if out:
            self.note()
            self.autosave(force=True)
        return out or {"error": "expected identity or teach"}

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
        if samples.size == 0:
            return {"moments": 0, "error": "no samples"}
        n = 0
        for feat, dt in self.ear.process(samples):
            self.mind.brain.step({"ears": feat}, dt / 0.01)
            self.note()
            n += 1
        # A buffer shorter than one cochlear hop yields nothing, which used to
        # read as "she heard but had nothing to say". Say which it was.
        return {"moments": n, "samples": int(samples.size),
                "too_short": bool(n == 0)}

    def _eye(self):
        if self.eye is None:
            self.eye = EventRetina(
                grid=int(round((self.mind.brain.cfg.senses["eyes"] // 2) ** 0.5)))
        return self.eye

    def see(self, body):
        w, h = int(body.get("w", 0)), int(body.get("h", 0))
        px = np.asarray(body.get("pixels", []), dtype=float)
        if w * h == 0 or px.size != w * h or w * h > 512 * 512:
            return {"error": f"expected w*h grey pixels in 0..1 "
                            f"(got {px.size} for {w}x{h})"}
        eye = self._eye()
        # The client captures whatever its camera gives (the app asks for
        # 160x120 = 19200 px) while the retina is grid x grid (4x4 = 16).
        # Reshaping to the client's w/h and handing that to the retina has
        # never worked: it raised a broadcast error on every single frame, so
        # her eyes were dead in the interface. Her retina decides its own
        # input size; downsample to it instead of trusting the client.
        frame = px.reshape(h, w)
        g = eye.grid
        if (h, w) != (g, g):
            rows = np.linspace(0, h, g + 1).astype(int)
            cols = np.linspace(0, w, g + 1).astype(int)
            frame = np.array([[frame[rows[i]:max(rows[i + 1], rows[i] + 1),
                                      cols[j]:max(cols[j + 1], cols[j] + 1)].mean()
                               for j in range(g)] for i in range(g)])
        ev, dt = eye.observe(frame, float(body.get("t", time.time())))
        if dt > 0:
            self.mind.brain.step({"eyes": ev}, dt / 0.01)
            self.note()
        return {"events": int(round(ev.sum() * 100)), "grid": g,
                "resized_from": [int(w), int(h)] if (w, h) != (g, g) else None}

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
          ("GET", "/api/soul"): "soul", ("POST", "/api/soul"): "soul",
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
            # Say someone is here before waiting for her: a dream in progress
            # checks this between steps and wakes up. (It used to be set only
            # after the lock was won, so a waiting request could not wake her.)
            life.touched = time.time()
            if not life.lock.acquire(timeout=LOCK_WAIT):
                return self._send(503, {"error": "she is dreaming", "retry_in": 5})
            try:
                life.touched = time.time()
                return self._send(200, getattr(life, name)(body))
            except (BrokenPipeError, ConnectionResetError):
                # The browser gave up and closed the socket. There is nobody
                # left to answer, and a 500 would be a lie. Stay quiet.
                return
            except Exception as e:  # keep her alive whatever a request does
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                life.lock.release()

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
    """Take exclusive ownership of her mind, then run her.

    The app is her long-lived owner. Without this, `ruth app` and `ruth teach`
    each loaded a full independent copy of brain.npz and whichever exited
    last destroyed the other. Measured here: the app held 7,262 moments in
    memory while the CLI had already written a 689-moment copy over the same
    file, ready to be clobbered in turn.
    """
    try:
        lock = owner(home or paths.home(), label="ruth app")
        lock.__enter__()
    except MindBusy as exc:
        raise SystemExit(f"another Ruth is already awake -- {exc}")
    try:
        _awake(host, port, open_browser, home)
    finally:
        try:
            lock.__exit__(None, None, None)
        except Exception:
            pass


def _awake(host: str = "127.0.0.1", port: int = 7438, open_browser: bool = True,
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
