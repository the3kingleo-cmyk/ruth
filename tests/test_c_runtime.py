import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from ruth import Brain, BrainConfig
from ruth.senses import bytes_signal

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(os.path.dirname(HERE), "ruth", "csrc")


@unittest.skipUnless(shutil.which("cc") or shutil.which("gcc"), "C compiler required")
class TestCRuntimeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "ruth-core")
        cc = shutil.which("cc") or shutil.which("gcc")
        subprocess.run([cc, "-O2", "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-o", cls.exe,
                        os.path.join(CSRC, "ruth_core.c"), "-lm"], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _parity(self, grow=0, **kw):
        cfg = BrainConfig(auto_sleep=False, inter=48, command=32, motor=16, ssm_channels=24,
                          working_capacity=2048, longterm_capacity=256, **kw)
        text = b"Ruth is one mind in many bodies. She learns while she lives. " * 4
        b = Brain(cfg)
        b.learn_bytes(text[:100])
        if grow:
            b.grow_neurons(grow)
            b.learn_bytes(text[:40])
        model = os.path.join(self.tmp, "m.bin")
        b.export_core(model)
        out = subprocess.run([self.exe, model, "trace"], input=text[100:],
                             capture_output=True, check=True).stdout.decode()
        c = np.array([[float(v) for v in line.split()] for line in out.splitlines() if line])
        py = []
        for u in bytes_signal(text[100:]):
            b.step({"text": u})
            py.append(b.prediction("text").copy())
        self.assertEqual(c.shape, (len(text) - 100, 9))
        self.assertLess(np.abs(c - np.array(py)).max(), 1e-8)

    def test_parity_cfc_text_only(self):
        self._parity(senses={"text": 9})

    def test_parity_ltc_multimodal_layout(self):
        self._parity(cell="ltc")

    def test_parity_cortical_columns(self):
        self._parity(senses={"text": 9}, cell="column")

    def test_parity_after_growth(self):
        self._parity(senses={"text": 9}, grow=8)

    def test_learn_and_generate(self):
        b = Brain(BrainConfig(senses={"text": 9}, inter=32, command=24, motor=8))
        model = os.path.join(self.tmp, "g.bin")
        b.export_core(model)
        learn = subprocess.run([self.exe, model, "learn"], input=b"I am Ruth. " * 40,
                               capture_output=True)
        self.assertEqual(learn.returncode, 0, learn.stderr.decode(errors="replace"))
        gen = subprocess.run([self.exe, model, "gen", "10", "--no-save"], input=b"I am ",
                             capture_output=True)
        self.assertEqual(gen.returncode, 0, gen.stderr.decode(errors="replace"))
        out = gen.stdout
        self.assertIn(b"Ruth", out)


if __name__ == "__main__":
    unittest.main()
