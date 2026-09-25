"""Tests for Ruth's senses through the app server.

Two real bugs lived here, both invisible because nothing exercised the
sensory endpoints:

- `/api/see` reshaped the incoming pixels to the *client's* w/h and handed
  that array to the event retina, which is grid x grid. Her app captures
  160x120 = 19200 px while her retina is 4x4 = 16, so every single frame
  raised `operands could not be broadcast together with shapes (8,8) (32,32)`.
  Her eyes had never worked through the interface.
- `/api/hear` returned `{"moments": 0}` for a buffer shorter than one
  cochlear hop, which reads as "she heard but had nothing to say" rather
  than "there was not enough to hear".

These drive the real endpoints over HTTP against a real mind.
"""
import contextlib
import io
import json
import pathlib
import math
import os
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from ruth.app.server import Life, make_handler


def post(url, payload, timeout=60):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class TestSenses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.life = Life(cls.tmp)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.life))
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _grid(self):
        eyes = self.life.mind.brain.cfg.senses["eyes"]
        return int(round((eyes // 2) ** 0.5))

    # ---------------------------------------------------------------- voice
    def test_voice_returns_real_audio(self):
        r = post(f"{self.base}/api/voice", {"text": "hello Ruth"})
        self.assertEqual(r["rate"], 16000)
        self.assertGreater(len(r["samples"]), 1000, "no audio produced")
        self.assertGreater(max(abs(s) for s in r["samples"]), 0.01,
                           "audio is silent")

    def test_voice_handles_empty_text(self):
        r = post(f"{self.base}/api/voice", {"text": ""})
        self.assertEqual(r["samples"], [])

    def test_voice_does_not_permanently_alter_her(self):
        """She must not learn from her own voice output."""
        before = self.life.mind.brain.steps
        post(f"{self.base}/api/voice", {"text": "something she should not absorb"})
        self.assertEqual(self.life.mind.brain.steps, before,
                         "speaking taught her something; it should learn=False")

    # ----------------------------------------------------------------- eyes
    def test_eye_accepts_the_apps_real_capture_size(self):
        """Regression: 160x120 = 19200 px against a 4x4 retina used to raise a
        broadcast error on every frame."""
        w, h = 160, 120
        px = [0.0] * (w * h)
        for y in range(h):
            for x in range(40, 46):
                px[y * w + x] = 1.0
        r = post(f"{self.base}/api/see", {"w": w, "h": h, "pixels": px, "t": 1.0})
        self.assertNotIn("error", r, r)
        self.assertEqual(r["grid"], self._grid())
        self.assertEqual(r["resized_from"], [w, h])

    def test_eye_sees_motion_and_ignores_a_still_frame(self):
        """An event camera responds to change, not to stillness."""
        w = h = self._grid()

        def bar(col):
            px = [0.0] * (w * h)
            px[col] = 1.0
            return px

        # settle on a reference, then compare
        post(f"{self.base}/api/see", {"w": w, "h": h, "pixels": bar(0), "t": 10.0})
        same = post(f"{self.base}/api/see", {"w": w, "h": h, "pixels": bar(0), "t": 10.05})
        moved = post(f"{self.base}/api/see", {"w": w, "h": h, "pixels": bar(2), "t": 10.10})
        self.assertEqual(same["events"], 0,
                         "a still frame must produce no events: she responds to "
                         "change, not to presence")
        self.assertGreater(moved["events"], 0,
                           "moving the bar must fire events")

    def test_eye_rejects_a_mismatched_payload_clearly(self):
        r = post(f"{self.base}/api/see", {"w": 8, "h": 8, "pixels": [0.0] * 10, "t": 1.0})
        self.assertIn("error", r)
        self.assertIn("grey pixels", r["error"])

    def test_eye_rejects_an_absurd_payload(self):
        r = post(f"{self.base}/api/see", {"w": 4096, "h": 4096, "pixels": [], "t": 1.0})
        self.assertIn("error", r)

    def test_eye_handles_a_single_pixel_frame(self):
        r = post(f"{self.base}/api/see", {"w": 1, "h": 1, "pixels": [1.0], "t": 20.0})
        self.assertNotIn("error", r, r)

    # ----------------------------------------------------------------- ears
    def test_ear_hears_a_realistic_buffer(self):
        sr = 16000
        samples = [0.4 * math.sin(2 * math.pi * 220 * t / sr) for t in range(sr // 2)]
        r = post(f"{self.base}/api/hear", {"rate": sr, "samples": samples})
        self.assertGreater(r["moments"], 0, "half a second of tone produced nothing")
        self.assertFalse(r["too_short"])

    def test_ear_says_when_the_buffer_is_too_short(self):
        """Regression: a sub-hop buffer reported moments=0 with no explanation,
        which read as 'she heard but had nothing to say'."""
        r = post(f"{self.base}/api/hear", {"rate": 16000, "samples": [0.1, -0.1, 0.2, -0.2]})
        self.assertEqual(r["moments"], 0)
        self.assertTrue(r["too_short"], "a short buffer must be labelled")
        self.assertEqual(r["samples"], 4)

    def test_ear_rejects_an_empty_buffer(self):
        r = post(f"{self.base}/api/hear", {"rate": 16000, "samples": []})
        self.assertIn("error", r)

    def test_ear_hears_silence_without_error(self):
        r = post(f"{self.base}/api/hear", {"rate": 16000, "samples": [0.0] * 8000})
        self.assertNotIn("error", r)

    def test_ear_truncates_an_overlong_buffer(self):
        """A client streaming a minute of audio must not stall her."""
        r = post(f"{self.base}/api/hear", {"rate": 16000,
                                          "samples": [0.0] * (16000 * 10)})
        self.assertLessEqual(r["samples"], 16000 * 2)


if __name__ == "__main__":
    unittest.main()


class TestEarsDaemon(unittest.TestCase):
    """tools/ears.py is the always-on half of her hearing.

    The bug worth pinning: arecord's -d is DURATION in whole seconds and 0
    means "record until killed", so a fractional window truncated to 0 hung
    the daemon forever instead of streaming.
    """
    @staticmethod
    def _load():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ears",
            pathlib.Path(__file__).resolve().parent.parent / "tools" / "ears.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def setUp(self):
        self.ears = self._load()

    def test_window_is_never_truncated_to_zero(self):
        self.assertGreaterEqual(
            self.ears.WINDOW, 1,
            "a window under 1s truncates arecord -d to 0, which records forever")

    def test_missing_arecord_is_reported_not_raised(self):
        with mock.patch.object(self.ears.shutil, "which", return_value=None):
            self.assertIsNone(self.ears.arecord_bin())
            samples, err = self.ears.capture_window(None)
        self.assertIsNone(samples)
        self.assertIn("arecord", err)

    def test_capture_failure_returns_a_reason(self):
        with mock.patch.object(self.ears.shutil, "which",
                               return_value="/bin/false"):
            samples, err = self.ears.capture_window(None)
        self.assertIsNone(samples)
        self.assertTrue(err)

    def test_post_to_a_dead_app_is_an_error_not_a_crash(self):
        moments, err = self.ears.her_ears("http://127.0.0.1:1", [0.0] * 1600)
        self.assertEqual(moments, 0)
        self.assertIn("cannot reach", err)

    def test_check_mode_reports_silence_honestly(self):
        """A device that exists but carries no audio must not look like a
        working microphone."""
        with mock.patch.object(self.ears, "capture_devices",
                               return_value=(["card 0"], None)), \
             mock.patch.object(self.ears, "capture_window",
                               return_value=([0.0] * 16000, None)), \
             mock.patch.object(self.ears, "her_ears", return_value=(3, None)), \
             contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(self.ears.check("http://127.0.0.1:1", None), 0)
        self.assertIn("SILENCE", buf.getvalue())

    def test_check_mode_fails_when_capture_is_broken(self):
        with mock.patch.object(self.ears, "capture_devices",
                               return_value=(["card 0"], None)), \
             mock.patch.object(self.ears, "capture_window",
                               return_value=(None, "boom")):
            self.assertEqual(self.ears.check("http://127.0.0.1:1", None), 1)


class TestSingleOwner(unittest.TestCase):
    """One mind, one owner.

    Every entry point used to open her independently, so the app and the CLI
    each held a full copy of brain.npz and whichever saved last destroyed the
    other. Measured on this box: 7,262 moments in the app's memory against a
    689-moment copy already written to the same file by the CLI.
    """
    @staticmethod
    def _owner_mod():
        from ruth.owner import MindBusy, owner, who_owns
        return MindBusy, owner, who_owns

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def test_a_free_mind_can_be_taken(self):
        _, owner, who_owns = self._owner_mod()
        with owner(self.tmp, label="first"):
            self.assertIn("first", who_owns(self.tmp) or "")

    def test_a_second_owner_is_refused(self):
        _, owner, _ = self._owner_mod()
        with owner(self.tmp, label="app"):
            with self.assertRaises(Exception) as cm:
                with owner(self.tmp, label="cli"):
                    pass
        self.assertIn("owns", str(cm.exception).lower())

    def test_the_refusal_names_the_owner_and_the_consequence(self):
        _, owner, _ = self._owner_mod()
        with owner(self.tmp, label="ruth app"):
            with self.assertRaises(Exception) as cm:
                with owner(self.tmp, label="ruth teach"):
                    pass
        msg = str(cm.exception)
        self.assertIn("ruth app", msg, "must name who holds her")
        self.assertIn("overwrite", msg.lower(), "must say what would be lost")

    def test_the_lock_is_released_when_the_owner_exits(self):
        _, owner, _ = self._owner_mod()
        with owner(self.tmp, label="first"):
            pass
        with owner(self.tmp, label="second"):   # must not raise
            pass

    def test_the_lock_survives_a_killed_owner(self):
        """fcntl locks are released by the kernel on death, so a crashed app
        must not leave her permanently unwritable."""
        import fcntl
        _, owner, who_owns = self._owner_mod()
        path = os.path.join(self.tmp, ".ruth-owner.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        with self.assertRaises(Exception):
            with owner(self.tmp, label="other"):
                pass
        os.close(fd)          # simulate the owner dying
        with owner(self.tmp, label="next"):
            pass

    def test_a_stale_pid_does_not_look_like_a_live_owner(self):
        _, _, who_owns = self._owner_mod()
        with open(os.path.join(self.tmp, ".ruth-owner.lock"), "w") as fh:
            fh.write("999999 dead-process\n")
        self.assertIsNone(who_owns(self.tmp),
                          "a lock file left by a dead pid must read as free")
