import os
import tempfile
import unittest

import numpy as np

from ruth import Brain, BrainConfig
from ruth.agent import ActiveInferenceAgent
from ruth.dynamics import LegendreTrace
from ruth.memory import AttractorMemory
from ruth.senses import (Cochlea, EventRetina, Voice, byte_code, byte_decode,
                          bytes_signal)


def small(**kw):
    base = dict(senses={"text": 9}, inter=32, command=24, motor=8, ssm_channels=16,
                ssm_state=8, working_capacity=128, longterm_capacity=512, replay_capacity=128)
    base.update(kw)
    return BrainConfig(**base)


class TestSenses(unittest.TestCase):
    def test_byte_code_roundtrip_all_bytes(self):
        for b in range(256):
            self.assertEqual(byte_decode(byte_code(b)), b)
        sig = bytes_signal(bytes(range(256)))
        self.assertTrue(np.allclose(sig, np.stack([byte_code(b) for b in range(256)])))

    def test_cochlea_is_frequency_selective(self):
        sr = 16000
        t = np.arange(sr // 2) / sr
        ear = Cochlea(sr, channels=16)
        feats = [f for f, _ in ear.process(np.sin(2 * np.pi * 440 * t))]
        band = int(np.argmax(np.mean(feats[10:], axis=0)))
        nearest = int(np.argmin(np.abs(np.log(ear.freqs / 440))))
        self.assertLessEqual(abs(band - nearest), 1)

    def test_retina_fires_only_on_change(self):
        eye = EventRetina(grid=4)
        frame = np.full((32, 32), 0.5)
        eye.observe(frame, 0.0)
        ev, dt = eye.observe(frame, 0.033)
        self.assertEqual(ev.sum(), 0.0)
        moved = frame.copy()
        moved[:, 10:14] = 1.0
        ev, dt = eye.observe(moved, 0.066)
        self.assertGreater(ev[:16].sum(), 0.0)      # ON events
        self.assertAlmostEqual(dt, 0.033, places=6)

    def test_voice_renders_bounded_audio(self):
        curves = np.array([Voice.params_from_latent(np.random.randn(4)) for _ in range(20)])
        wav = Voice(8000).render(curves, 0.01)
        self.assertEqual(wav.shape[0], 20 * 80)
        self.assertLessEqual(np.abs(wav).max(), 0.9 + 1e-9)


class TestDynamics(unittest.TestCase):
    def test_liquid_state_bounded_under_irregular_time(self):
        b = Brain(small())
        rng = np.random.default_rng(0)
        for _ in range(300):
            b.step({"text": rng.normal(0, 3, 9)}, dt=float(rng.uniform(0.01, 20.0)))
            self.assertTrue(np.all(np.abs(b.core.x) <= 1.0))
            self.assertTrue(np.all(np.isfinite(b.ssm.h)))

    def test_ltc_cell_runs(self):
        b = Brain(small(cell="ltc"))
        for c in b"hello world":
            b.step({"text": byte_code(c)})
        self.assertTrue(np.all(np.isfinite(b.core.x)))

    def test_state_size_independent_of_stream_length(self):
        b = Brain(small())
        size = sum(a.size for a in (b.core.x, b.ssm.h, b.trace.m))
        b.learn_bytes(b"x" * 500)
        self.assertEqual(size, sum(a.size for a in (b.core.x, b.ssm.h, b.trace.m)))

    def test_legendre_trace_remembers_window(self):
        tr = LegendreTrace(1, order=6, window=4.0)
        for v in (0, 0, 0, 1, 0, 0):
            tr.step(np.array([float(v)]), 1.0)
        a = tr.m.ravel().copy()
        tr.reset()
        for v in (0, 0, 0, 0, 1, 0):
            tr.step(np.array([float(v)]), 1.0)
        b = tr.m.ravel()
        cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        self.assertLess(cos, 0.95)  # different positions -> distinguishable


class TestAttractorMemory(unittest.TestCase):
    def test_noisy_recall_and_energy_descent(self):
        rng = np.random.default_rng(1)
        mem = AttractorMemory(64, 3, 200, 10, beta=16.0, iters=3)
        pats = rng.normal(size=(50, 64))
        pats /= np.linalg.norm(pats, axis=1, keepdims=True)
        for i, p in enumerate(pats):
            mem.store(p, np.array([i, 0.0, 0.0]))
        q = pats[17] + 0.3 * rng.normal(size=64) / 8
        e0 = mem.energy(q / np.linalg.norm(q))
        val, xi, conf, _ = mem.recall(q)
        self.assertAlmostEqual(val[0], 17, delta=0.05)
        self.assertGreater(xi @ pats[17] / np.linalg.norm(xi), 0.99)
        self.assertLess(mem.energy(xi), e0)

    def test_consolidation_merges_and_keeps_recall(self):
        mem = AttractorMemory(8, 1, 10, 10, beta=50.0)
        k = np.eye(8)[0]
        for _ in range(5):
            mem.store(k, np.array([1.0]))
        mem.store(np.eye(8)[1], np.array([2.0]))
        stats = mem.consolidate()
        self.assertEqual(stats["longterm"], 2)
        self.assertEqual(mem.w_n, 0)
        self.assertEqual(mem.l_count[0], 5)
        self.assertAlmostEqual(mem.recall(k)[0][0], 1.0, places=3)


class TestLearning(unittest.TestCase):
    def test_learns_repeating_phrase_and_speaks_it(self):
        b = Brain(small())
        phrase = b"I am Ruth. "
        r = b.learn_bytes(phrase * 30)
        tail = b.learn_bytes(phrase * 3)
        self.assertGreater(tail["byte_accuracy"], 0.9)
        self.assertIn(b"Ruth", b.generate(b"I am ", 12))
        self.assertGreater(tail["byte_accuracy"], r["byte_accuracy"])

    def test_sleep_consolidates_and_replays(self):
        b = Brain(small(auto_sleep=False))
        b.learn_bytes(b"the quick brown fox jumps over the lazy dog. " * 3)
        self.assertGreater(b.memory.w_n, 0)
        stats = b.sleep()
        self.assertEqual(b.memory.w_n, 0)
        self.assertGreater(stats["longterm"], 0)
        self.assertGreater(stats["replayed"], 0)

    def test_save_load_resumes_identically(self):
        b = Brain(small())
        b.learn_bytes(b"continuity of mind. " * 5)
        with tempfile.TemporaryDirectory() as d:
            path = b.save(os.path.join(d, "s.npz"))
            c = Brain.load(path)
        for ch in b"continuity":
            pa = b.step({"text": byte_code(ch)})
            pb = c.step({"text": byte_code(ch)})
            self.assertTrue(np.allclose(pa, pb))

    def test_multimodal_save_load_keeps_sense_layout(self):
        b = Brain(small(senses={"text": 9, "ears": 4, "eyes": 2}))
        for i in range(40):
            b.step({"text": byte_code(97 + i % 5), "ears": np.full(4, np.sin(i)), "eyes": [i % 2, 0]})
        with tempfile.TemporaryDirectory() as d:
            c = Brain.load(b.save(os.path.join(d, "s.npz")))
        self.assertEqual(list(c.cfg.senses), ["text", "ears", "eyes"])
        pa = b.step({"text": byte_code(97), "ears": np.ones(4)})
        pb = c.step({"text": byte_code(97), "ears": np.ones(4)})
        self.assertTrue(np.allclose(pa, pb))

    def test_multimodal_senses_fuse(self):
        b = Brain(small(senses={"text": 9, "ears": 4}))
        for i in range(50):
            b.step({"text": byte_code(65 + i % 3), "ears": np.full(4, np.sin(i))})
            b.step({"ears": np.full(4, np.cos(i))}, dt=0.5)
        self.assertEqual(b.prediction("ears").shape, (4,))


class TestActiveInference(unittest.TestCase):
    def test_agent_learns_world_and_reaches_goal(self):
        cfg = small(senses={"obs": 1, "act": 1}, targets=["obs"], trace_senses=[])
        brain = Brain(cfg)
        agent = ActiveInferenceAgent(brain, "obs", "act", preferred=[0.6],
                                     actions=[-1.0, -0.5, 0.0, 0.5, 1.0], seed=3)
        p = -0.6
        rng = np.random.default_rng(3)
        for t in range(300):
            obs = np.array([p])
            a = agent.choose(obs, explore=1.0 if t < 150 else 0.0)
            agent.live(obs, a)
            p = float(np.clip(p + 0.1 * a[0] + 0.005 * rng.normal(), -1, 1))
            if t == 150:
                p = -0.6  # restart from far away once the model is learned
        self.assertLess(abs(p - 0.6), 0.15)


if __name__ == "__main__":
    unittest.main()
