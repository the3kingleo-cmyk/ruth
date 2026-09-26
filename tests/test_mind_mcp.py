"""Tests for mind_mcp, the bridge between the agent and Ruth's actual mind.

Two things in this project are called "ruth" and they are not the same thing:
the token-free Python mind, and the model-backed agent that speaks her
persona. Installing both does not connect them. mind_mcp is that connection,
so the tests here are about the bridge being honest:

- every tool reaches her real CLI and returns her real state;
- a missing brain is a clear error, not a stack trace;
- **no tool guesses a CLI flag.** `mind_learned` shipped calling
  `ruth introspect --json`; introspect takes no arguments at all. The call
  silently fell back to weaker data and nothing failed. test_no_guessed_flags
  pins the real signatures.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = ROOT / "tools" / "mind_mcp.py"
_spec = importlib.util.spec_from_file_location("mind_mcp", SERVER)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _cli_help(*args):
    """Ask the real CLI what a subcommand accepts."""
    bin_ = os.environ.get("RUTH_BIN", os.path.expanduser("~/.local/bin/ruth"))
    if not os.path.exists(bin_):
        return None
    return subprocess.run([bin_, *args, "--help"], capture_output=True,
                          text=True, timeout=90).stdout


class TestToolContract(unittest.TestCase):
    def test_every_tool_has_a_handler(self):
        declared = {t["name"] for t in mm.TOOLS}
        self.assertEqual(declared, set(mm.HANDLERS),
                         "a declared tool with no handler would fail at call time")

    def test_every_handler_is_declared(self):
        self.assertEqual(set(mm.HANDLERS), {t["name"] for t in mm.TOOLS})

    def test_tools_are_documented_and_schematised(self):
        for t in mm.TOOLS:
            self.assertTrue(t.get("description"), t["name"])
            # `required` is optional: mind_teach accepts text OR files, so
            # neither key is individually required and the handler validates
            # the combination (see TestErrorPaths).
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])
            for prop, spec in (t["inputSchema"].get("properties") or {}).items():
                self.assertIn("type", spec, f"{t['name']}.{prop}")

    def test_mind_is_exposed(self):
        """The point of the whole file: her mind, reachable as tools."""
        for name in ("mind_status", "mind_think", "mind_teach", "mind_sleep",
                     "mind_introspect", "mind_patch"):
            self.assertIn(name, mm.HANDLERS, name)


class TestNoGuessedFlags(unittest.TestCase):
    """Every subcommand this server invokes must exist with that signature.

    A wrong flag does not raise -- argparse exits 2, and the tool falls back
    to weaker data while looking healthy. That is how `introspect --json`
    shipped. These tests are skipped when her CLI is not installed, but on
    any box that actually runs her, they are the real check.
    """

    def test_introspect_takes_no_arguments(self):
        """She already emits JSON; passing --json is wrong."""
        help_ = _cli_help("introspect")
        if help_ is None:
            self.skipTest("ruth CLI not installed")
        self.assertNotIn("--json", help_,
                         "mind_learned/mind_introspect must not pass --json")
        self.assertRegex(help_, r"usage:.*introspect")

    def test_subcommands_the_server_calls_all_exist(self):
        help_ = _cli_help("status")
        if help_ is None:
            self.skipTest("ruth CLI not installed")
        for sub in ("status", "think", "say", "teach", "sleep", "dreams",
                    "check", "introspect", "patch", "export"):
            r = subprocess.run(
                [os.environ.get("RUTH_BIN", os.path.expanduser("~/.local/bin/ruth")),
                 sub, "--help"], capture_output=True, text=True, timeout=90)
            self.assertEqual(r.returncode, 0,
                             f"ruth {sub} is not a subcommand, but mind_mcp calls it")


class TestErrorPaths(unittest.TestCase):
    def test_missing_brain_is_a_clear_error(self):
        with mock.patch.object(mm, "RUTH_BIN", "/nonexistent/ruth"):
            with self.assertRaises(mm.MindError) as cm:
                mm.mind_status()
        self.assertIn("no brain at", str(cm.exception))
        self.assertIn("bootstrap", str(cm.exception),
                      "the error should say how to fix it")

    def test_empty_prompt_is_refused(self):
        for bad in ("", "   ", None):
            with self.assertRaises(mm.MindError):
                mm.mind_think(bad)
            with self.assertRaises(mm.MindError):
                mm.mind_say(bad)

    def test_teach_needs_text_or_files(self):
        with self.assertRaises(mm.MindError):
            mm.mind_teach()
        with self.assertRaises(mm.MindError):
            mm.mind_teach(text="   ")

    def test_check_needs_files(self):
        with self.assertRaises(mm.MindError):
            mm.mind_check([])

    def test_patch_needs_an_op(self):
        for bad in (None, {}, "grow_neurons", [1]):
            with self.assertRaises(mm.MindError):
                mm.mind_patch(bad)

    def test_non_json_output_is_reported_not_swallowed(self):
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="not json")
        # the binary check runs first; point it at something that exists so
        # the path under test is reached on any machine
        with mock.patch.object(mm, "RUTH_BIN", sys.executable), \
             mock.patch.object(mm.subprocess, "run", return_value=done):
            with self.assertRaises(mm.MindError) as cm:
                mm._json("status")
        self.assertIn("did not return JSON", str(cm.exception))

    def test_failing_command_surfaces_stderr(self):
        done = subprocess.CompletedProcess(args=[], returncode=1, stdout="",
                                           stderr="boom")
        # the binary check runs first; point it at something that exists so
        # the path under test is reached on any machine
        with mock.patch.object(mm, "RUTH_BIN", sys.executable), \
             mock.patch.object(mm.subprocess, "run", return_value=done):
            with self.assertRaises(mm.MindError) as cm:
                mm._json("status")
        self.assertIn("boom", str(cm.exception))

    def test_timeout_is_reported(self):
        # the binary check runs first; point it at something that exists so
        # the path under test is reached on any machine
        with mock.patch.object(mm, "RUTH_BIN", sys.executable), \
             mock.patch.object(mm.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("ruth", 1)):
            with self.assertRaises(mm.MindError) as cm:
                mm._json("status")
        self.assertIn("timed out", str(cm.exception))


class TestAgainstTheRealMind(unittest.TestCase):
    """Exercises her actual state, in a throwaway RUTH_HOME."""

    def setUp(self):
        self.bin = os.environ.get("RUTH_BIN", os.path.expanduser("~/.local/bin/ruth"))
        if not os.path.exists(self.bin):
            self.skipTest("ruth is not installed on this box")
        # Her own mind, not the live one. These tests used to teach the mind
        # the box is actually running, which is both rude and now refused:
        # `ruth app` holds exclusive ownership, and a writer that cannot get
        # the lock is stopped rather than allowed to clobber it.
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = mock.patch.dict(os.environ, {"RUTH_HOME": self._tmp.name})
        self._env.start()
        self.addCleanup(self._env.stop)
        patch = mock.patch.object(mm, "RUTH_HOME", self._tmp.name)
        patch.start()
        self.addCleanup(patch.stop)

    def test_status_reports_a_lived_mind(self):
        s = mm.mind_status()
        self.assertEqual(s["name"], "Ruth")
        self.assertGreater(s["neurons"], 0)
        self.assertGreater(s["parameters"], 0)
        self.assertIn("text", s["senses"])

    def test_introspect_returns_her_graph(self):
        graph = mm.mind_introspect()
        self.assertIsInstance(graph, dict)
        self.assertIn("nodes", graph)
        self.assertIn("senses", graph["nodes"])

    def test_learned_agrees_with_status(self):
        s, learned = mm.mind_status(), mm.mind_learned()
        self.assertEqual(learned["moments_lived"], s["moments_lived"])
        self.assertEqual(learned["working_memories"], s["working_memories"])

    def test_teach_then_status_shows_more_moments(self):
        before = mm.mind_status()["moments_lived"]
        result = mm.mind_teach("a small fact for the bridge test")
        self.assertGreater(result["bytes"], 0)
        self.assertGreater(mm.mind_status()["moments_lived"], before)

    def test_think_returns_candidates_with_free_energy(self):
        mm.mind_teach("the repository holds code")
        t = mm.mind_think("the repository holds")
        self.assertTrue(t["candidates"])
        for c in t["candidates"]:
            self.assertIn("free_energy", c)
            self.assertIsInstance(c["chosen"], bool)
        self.assertTrue(any(c["chosen"] for c in t["candidates"]))

    def test_patch_requires_a_whitelisted_op(self):
        """Her meta-kernel refuses unknown ops; the bridge must not bypass it."""
        with self.assertRaises(mm.MindError):
            mm.mind_patch({"op": "definitely-not-an-op"})

    def test_export_writes_a_real_model(self):
        out = mm.mind_export()
        self.assertTrue(os.path.getsize(out["path"]) > 0)

    def test_check_reports_on_a_real_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("nothing private here\n")
            path = fh.name
        try:
            self.assertIsInstance(mm.mind_check([path]), dict)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
