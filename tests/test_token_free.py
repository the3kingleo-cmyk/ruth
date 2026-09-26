"""Proof, re-checked on every build, that Ruth is token-free and self-contained."""
import ast
import os
import re
import sys
import tempfile
import unittest

import numpy as np

from ruth import Brain, BrainConfig
from ruth.mind import Mind, Temperament
from ruth.senses import byte_code

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ruth")
STDLIB = set(getattr(sys, "stdlib_module_names", ())) or {
    "argparse", "ast", "collections", "contextlib", "copy", "dataclasses", "errno", "fcntl",
    "glob", "hashlib", "http", "json", "msvcrt", "os", "traceback",
    "re", "shutil", "struct", "subprocess", "sys", "tempfile", "threading", "time", "wave",
    "webbrowser", "__future__"}


def sources(exts):
    for d, _, files in os.walk(PKG):
        for f in files:
            if f.endswith(exts):
                yield os.path.join(d, f)


class TestTokenFree(unittest.TestCase):
    def test_no_tokenizer_vocabulary_or_ai_library_anywhere(self):
        banned = re.compile(r"tokeni[sz]er|sentencepiece|tiktoken|vocab_size|from_pretrained|"
                            r"transformers|openai|anthropic|api[_-]?key|hf_hub", re.I)
        for path in sources((".py", ".c", ".js", ".html")):
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    self.assertIsNone(banned.search(line), f"{path}:{n}: {line.strip()}")

    def test_only_numpy_and_the_standard_library(self):
        for path in sources((".py",)):
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    mods = [node.module]
                else:
                    continue
                for m in mods:
                    top = m.split(".")[0]
                    self.assertTrue(top in STDLIB or top in ("numpy", "ruth"), f"{path}: imports {m}")
                    self.assertNotIn(top, ("urllib", "socket", "requests", "ssl"),
                                     f"{path}: network client {m}")

    def test_brain_takes_continuous_signals_not_ids(self):
        b = Brain(BrainConfig(senses={"text": 9}))
        a, z = byte_code(ord("a")), byte_code(ord("z"))
        states = []
        for w in np.linspace(0.0, 1.0, 21):       # points *between* letters: no token exists there
            snap = b.snapshot()
            b.step({"text": (1 - w) * a + w * z}, dt=0.37, learn=False)
            states.append(b.core.x.copy())
            b.restore(snap)
        jumps = [np.abs(states[i + 1] - states[i]).max() for i in range(20)]
        self.assertGreater(np.abs(states[-1] - states[0]).max(), 0.05)   # it tells a from z
        self.assertTrue(all(j > 0 for j in jumps))                        # every in-between differs
        self.assertLess(max(jumps), 0.5 * np.abs(states[-1] - states[0]).max() + 1e-9)  # smoothly

    def test_text_pathway_is_nine_continuous_lines_not_a_symbol_table(self):
        b = Brain(BrainConfig())
        enc = b.encoders["text"]
        self.assertEqual(enc.omega.shape[1], 9)
        self.assertEqual(enc.w1.shape[1], 9 + 2 * b.cfg.fourier)
        self.assertEqual(b.cfg.senses["text"], 9)


class TestBlankSlate(unittest.TestCase):
    def test_a_new_ruth_is_born_empty_with_identity(self):
        home = tempfile.mkdtemp()
        m = Mind(home=home)
        self.assertEqual(m.identity["name"], "Ruth")
        self.assertIn("born", m.identity)
        self.assertEqual(m.brain.steps, 0)
        self.assertEqual(m.brain.memory.l_n + m.brain.memory.w_n, 0)
        self.assertEqual(m.cues, [])
        self.assertEqual(m.lived_history(), b"")
        self.assertEqual(m.temperament.snapshot(), Temperament().snapshot())
        m.save()
        w = Mind(home=home)
        self.assertEqual(w.identity, m.identity)          # the same being wakes up


if __name__ == "__main__":
    unittest.main()
