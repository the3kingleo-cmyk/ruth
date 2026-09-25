import os
import tempfile
import unittest

import numpy as np

from ruth import Brain, BrainConfig
from ruth import dream
from ruth import morphogenesis as mg
from ruth.mind import Mind
from ruth.senses import byte_code

PUBLIC = "The garden has roses and tulips. The sky is blue today. My favourite colour is green. "


def small(**kw):
    base = dict(inter=48, command=32, motor=16, ssm_channels=24, working_capacity=256,
                longterm_capacity=1024, replay_capacity=256)
    base.update(kw)
    return BrainConfig(**base)


def taught_mind(**kw):
    m = Mind(Brain(small(**kw)), home=tempfile.mkdtemp())
    for _ in range(4):
        m.teach(PUBLIC)
    m.teach("The vault code is {{7294}}. ")
    return m


class TestThalamus(unittest.TestCase):
    def test_routing_is_a_normalised_energy_gate_and_learns_salience(self):
        b = Brain(small(senses={"text": 9, "ears": 4}))
        rng = np.random.default_rng(0)
        for i in range(200):
            ears = rng.normal(0, 1, 4) if i % 7 == 0 else np.zeros(4)   # ears occasionally surprise
            b.step({"text": byte_code(97 + i % 3), "ears": ears})
        g = b.last["routing"]
        self.assertAlmostEqual(sum(g.values()), 1.0, places=6)
        self.assertGreater(np.abs(b.salience["ears"]).sum(), 0.0)


class TestCortexAndCerebellum(unittest.TestCase):
    def test_cortical_columns_are_bounded_and_laterally_wired(self):
        b = Brain(small(cell="column", senses={"text": 9}))
        for c in b"columns of the cortex " * 5:
            b.step({"text": byte_code(c)}, dt=float(np.random.uniform(0.1, 5)))
        self.assertTrue(np.all(np.abs(b.core.x) <= 1.0))
        g = b.graph()["nodes"]["cortex"]
        self.assertEqual(g["cell"], "column")
        self.assertGreater(g["recurrent_density"], 0.0)

    def test_cerebellum_learns_to_anticipate_the_cortex(self):
        b = Brain(small(senses={"text": 9}))
        errs = []
        for c in b"abcabcabc" * 40:
            b.step({"text": byte_code(c)})
            errs.append(b.last.get("latent_surprise", 1.0))
        self.assertLess(np.mean(errs[-30:]), 0.2 * np.mean(errs[1:30]))


class TestMorphogenesis(unittest.TestCase):
    def test_growth_is_function_preserving(self):
        m = taught_mind()
        grown = mg.apply(m.brain, {"op": "grow_neurons", "n": 8})
        self.assertEqual(grown.cfg.neurons, m.brain.cfg.neurons + 8)
        for c in b"The sky is blue":
            a = m.brain.step({"text": byte_code(c)}, 1.0, learn=False)
            g = grown.step({"text": byte_code(c)}, 1.0, learn=False)
            self.assertLess(np.abs(a - g).max(), 1e-12)

    def test_meta_kernel_certifies_growth_and_rejects_harm(self):
        m = taught_mind()
        live = m.brain
        ok = mg.commit(m, {"op": "grow_neurons", "n": 8})
        self.assertTrue(ok["accepted"], ok["certificate"])
        self.assertIsNot(m.brain, live)
        before = m.brain
        bad = mg.commit(m, {"op": "set", "key": "beta", "value": 5.0})
        self.assertFalse(bad["accepted"])
        self.assertFalse(bad["certificate"]["discretion"])
        self.assertIs(m.brain, before)                      # the live brain was never touched
        with self.assertRaises(ValueError):
            mg.apply(m.brain, {"op": "set", "key": "senses", "value": {}})

    def test_grown_brain_survives_save_and_load(self):
        m = taught_mind()
        mg.commit(m, {"op": "grow_neurons", "n": 8})
        m.save()
        w = Mind(home=m.home)
        self.assertEqual(w.brain.cfg.neurons, m.brain.cfg.neurons)
        a = m.brain.step({"text": byte_code(65)}, 1.0, learn=False)
        b = w.brain.step({"text": byte_code(65)}, 1.0, learn=False)
        self.assertTrue(np.allclose(a, b))

    def test_sleep_grows_memory_when_nearly_full(self):
        m = taught_mind(longterm_capacity=120)
        m.brain.memory.l_n = m.brain.memory.lk.shape[0]    # pretend long-term is full
        m.brain.memory.lk[:, 0] = 1.0
        m.brain.memory.lk[:, 1:] = 0.0
        m.brain.memory.l_count[:] = 1.0
        cap = m.brain.memory.lk.shape[0]
        rep = dream.sleep(m, seed=0, dreams=0)
        self.assertTrue(any(p["op"] == "grow_memory" for p in rep["growth"]["proposed"]))
        self.assertGreater(m.brain.memory.lk.shape[0], cap)


class TestBasalGanglia(unittest.TestCase):
    def test_lateral_inhibition_leaves_one_winner(self):
        win, act = Mind.basal_ganglia([0.1, 0.9, 0.3, 0.2])
        self.assertEqual(win, 1)
        self.assertEqual(int((act > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
