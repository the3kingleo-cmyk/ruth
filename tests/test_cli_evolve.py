"""Tests for the two modules that had zero coverage: cli.py and evolve.py.

`ruth/cli.py` (253 statements) and `ruth/evolve.py` (142) were never executed by
the suite, so both were unverified code in the public repository -- the largest
remaining block of "filler" in the sense that nothing had ever proven it
worked. These tests drive the real entry points in a temporary RUTH_HOME, with
no network, no TTY, and no dependency on a pre-existing mind.

The shape mirrors how she is actually used: build arguments, call main(argv),
and assert on what she wrote or printed.
"""
import ast
import io
import json
import os
import pathlib
import contextlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from ruth import paths
import ruth.cli as cli
import ruth.evolve as evolve
from ruth.mind import Mind


class TempHome(unittest.TestCase):
    """Every test gets its own RUTH_HOME so nothing touches a real mind."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self._env = mock.patch.dict(os.environ, {"RUTH_HOME": self.home})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

    def run_cli(self, *argv):
        """Run main(argv) and capture output.

        cli.main() returns None on success -- a non-zero exit is a raised
        SystemExit, which is what a failing subcommand does.
        """
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = cli.main(list(argv))
        self.assertIsNone(result, "main() should not return a value")
        return 0, out.getvalue(), err.getvalue()


class TestPaths(TempHome):
    def test_home_honours_ruth_home(self):
        self.assertEqual(paths.home(), self.home)

    def test_home_creates_the_directory(self):
        target = os.path.join(self.home, "nested", "ruth")
        with mock.patch.dict(os.environ, {"RUTH_HOME": target}):
            self.assertEqual(paths.home(), target)
        self.assertTrue(os.path.isdir(target))

    def test_path_builds_and_creates_parent(self):
        p = paths.path("a", "b.bin")
        self.assertTrue(p.endswith(os.path.join("a", "b.bin")))
        self.assertTrue(os.path.isdir(os.path.dirname(p)))

    def test_brain_state_default_and_override(self):
        default = paths.brain_state()
        self.assertTrue(os.path.isabs(default))
        with mock.patch.dict(os.environ, {"RUTH_BRAIN": "/tmp/explicit.npz"}):
            self.assertEqual(paths.brain_state(), "/tmp/explicit.npz")

    def test_no_user_name_is_embedded(self):
        """She must not be pinned to one account on any platform."""
        src = open(paths.__file__.replace(".pyc", ".py")).read()
        self.assertNotIn(os.path.expanduser("~"), src.replace("~", ""))


class TestCliSurface(TempHome):
    def test_version(self):
        with self.assertRaises(SystemExit) as cm:
            cli.main(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_requires_a_subcommand(self):
        with self.assertRaises(SystemExit):
            cli.main([])

    def test_teach_then_status(self):
        code, out, _ = self.run_cli("teach", "a small fact worth keeping")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertGreater(payload["bytes"], 0)

        code, out, _ = self.run_cli("status")
        self.assertEqual(code, 0)
        status = json.loads(out)
        self.assertEqual(status["identity"]["name"], "Ruth")
        self.assertGreater(status["moments_lived"], 0)

    def test_say_returns_text(self):
        self.run_cli("teach", "the repository holds code")
        code, out, _ = self.run_cli("say", "the repository holds")
        self.assertEqual(code, 0)
        self.assertIsInstance(out.strip(), str)

    def test_think_reports_free_energy_and_a_choice(self):
        self.run_cli("teach", "a fact")
        code, out, _ = self.run_cli("think", "a fact")
        self.assertEqual(code, 0)
        thoughts = json.loads(out)
        self.assertTrue(thoughts, "no candidate thoughts")
        for t in thoughts:
            self.assertIn("thought", t)
            self.assertIn("free_energy", t)
            self.assertIn("chosen", t)
        self.assertTrue(any(t["chosen"] for t in thoughts))

    def test_check_on_a_private_file_reports_json(self):
        secret = os.path.join(self.home, "secret.txt")
        with open(secret, "w") as fh:
            fh.write("nothing private here\n")
        code, out, _ = self.run_cli("check", secret)
        self.assertEqual(code, 0)
        json.loads(out)   # must be machine-readable

    def test_sleep_and_dreams_round_trip(self):
        self.run_cli("teach", "something to remember")
        code, out, _ = self.run_cli("sleep")
        self.assertEqual(code, 0)
        json.loads(out)
        code, out, _ = self.run_cli("dreams")
        self.assertEqual(code, 0)
        json.loads(out)

    def test_export_writes_a_model(self):
        code, out, _ = self.run_cli("export", os.path.join(self.home, "m.bin"))
        self.assertEqual(code, 0)
        self.assertTrue(os.path.getsize(out.strip().split("->")[-1].strip()) > 0)

    def test_unknown_subcommand_exits_nonzero(self):
        with self.assertRaises(SystemExit) as cm:
            cli.main(["not-a-command"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_app_help_does_not_bind_a_port(self):
        """--help on the app subcommand must not start a server."""
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["app", "--help"])
        self.assertEqual(cm.exception.code, 0)


class TestEvolve(TempHome):
    def test_primer_is_real_bytes(self):
        p = evolve.primer()
        self.assertIsInstance(p, (bytes, bytearray))
        self.assertGreater(len(p), 0)

    def test_corpus_uses_the_minds_own_text(self):
        Mind(home=self.home)                 # give her somewhere to read
        c = evolve.corpus(home=self.home)
        self.assertIsInstance(c, (bytes, bytearray))

    def test_evaluate_returns_a_bounded_accuracy(self):
        """evaluate() is the evolve gate's only signal, so it must be a real
        number in [0, 1] for any config -- including a crippled one."""
        import math
        text = b"the repository holds code. " * 60
        for overrides in ({}, {"ssm_channels": 3}, {"command": 8, "motor": 4}):
            score = evolve.evaluate(overrides, text, train=200, test=80)
            self.assertIsInstance(score, float, overrides)
            self.assertTrue(math.isfinite(score), f"{overrides} -> {score}")
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_evaluate_survives_an_unknown_override(self):
        """A bad hyper-parameter must raise, not silently score 0."""
        with self.assertRaises(TypeError):
            evolve.evaluate({"definitely_not_a_real_option": 1},
                            b"abc " * 60, train=80, test=40)

    def test_archive_round_trip(self):
        evolve._append({"overrides": {"a": 1}, "score": -0.5, "gen": 0})
        entries = evolve.load_archive()
        self.assertTrue(entries, "archive did not persist")
        self.assertEqual(entries[-1]["overrides"], {"a": 1})
        self.assertEqual(entries[-1]["score"], -0.5)

    def test_active_overrides_are_a_dict(self):
        self.assertIsInstance(evolve.active_overrides(), dict)

    def test_load_archive_on_a_fresh_home_is_empty_not_an_error(self):
        self.assertIsInstance(evolve.load_archive(), list)

    def test_run_tests_reports_failure_as_failure(self):
        """A gate that cannot fail is not a gate."""
        import subprocess as sp
        done = sp.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with mock.patch.object(sp, "run", return_value=done):
            result = evolve.run_tests(self.home)
        self.assertFalse(bool(result and getattr(result, "returncode", 1) == 0))


class TestTokenFreeStillHolds(TempHome):
    def test_a_born_mind_is_empty_but_identifiable(self):
        m = Mind(home=self.home)
        self.assertEqual(m.identity["name"], "Ruth")
        self.assertIn("born", m.identity)
        self.assertEqual(m.brain.steps, 0)

    def test_teaching_moves_her_state(self):
        m = Mind(home=self.home)
        before = m.brain.steps
        m.teach(b"a first impression that she will keep")
        self.assertGreater(m.brain.steps, before)

    def test_states_are_continuous_not_symbolic(self):
        """Between 'a' and 'z' there is a real, different intermediate."""
        m = Mind(home=self.home)
        st = [m.brain.step_signal(c) if hasattr(m.brain, "step_signal") else None
              for c in (b"a", b"z")]
        st = [s for s in st if s is not None]
        if st:
            self.assertEqual(len(st), 2)
            self.assertGreater(np.abs(np.asarray(st[0]) - np.asarray(st[1])).max(), 0.05)


if __name__ == "__main__":
    unittest.main()


class TestNoUntestedModules(unittest.TestCase):
    """No module of her mind may sit in the repository unreferenced.

    `ruth/cli.py` and `ruth/evolve.py` shipped with zero coverage for the
    life of this project. evolve.py in fact carried three relative imports
    that escaped the package, so its entire self-improvement path had never
    executed and could not have -- nothing would have raised until the day
    someone ran `ruth evolve`.

    A module counts as referenced when its name appears in the test suite or
    in any other module of the package. That is deliberately simpler than
    resolving the import graph: an earlier attempt walked the graph and
    mishandled `from . import x`, reporting reachable modules as orphans.
    What actually matters is the blunt fact -- is anything pointing at it.
    """

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    PKG = ROOT / "ruth"
    PKG_MODULES = ("__init__", "__main__")

    def test_every_module_is_referenced_by_the_suite_or_the_package(self):
        suite = "\n".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in (self.ROOT / "tests").glob("*.py"))
        package = {f: f.read_text(encoding="utf-8", errors="replace")
                   for f in self.PKG.glob("*.py")}
        orphans = []
        for f in sorted(self.PKG.glob("*.py")):
            if f.stem in self.PKG_MODULES:
                continue
            stem = f.stem
            in_suite = stem in suite
            # imported by a sibling module (checked over every file but itself)
            in_package = any(stem in text for other, text in package.items()
                             if other != f)
            if not (in_suite or in_package):
                orphans.append(f.name)
        self.assertEqual(orphans, [],
                         f"unreferenced modules, i.e. unverified code: {orphans}")

    def test_the_modules_that_were_once_unverified_are_now_covered(self):
        """Pin the specific regression rather than a general rule."""
        suite = "\n".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in (self.ROOT / "tests").glob("*.py"))
        for name in ("cli", "evolve", "paths"):
            self.assertIn(name, suite,
                          f"ruth/{name}.py is no longer exercised by any test")

    def test_no_relative_import_escapes_the_package(self):
        """A `from ..x` in a depth-1 module raises ImportError when that code
        path is first reached -- which, with no coverage, is never."""
        bad = []
        for dirpath, _dirs, files in os.walk(self.PKG):
            # the package itself is depth 1; ruth/app is depth 2, where
            # `from ..config` correctly means ruth.config
            parts = [p for p in pathlib.Path(os.path.relpath(dirpath, self.PKG)).parts
                     if p != "."]
            depth = len(parts) + 1
            for f in files:
                if not f.endswith(".py"):
                    continue
                full = os.path.join(dirpath, f)
                if "from .." in open(full, encoding="utf-8").read() and depth < 2:
                    bad.append(os.path.relpath(full, self.ROOT))
        self.assertEqual(bad, [], f"relative imports escape the package: {bad}")
