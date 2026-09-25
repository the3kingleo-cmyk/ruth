"""Tests for install-agent.py — the agent Ruth actually runs.

`agent/ruth.md` is canonical, but an operator runs the copy in an opencode
config, and the two drift apart silently. install-agent.py closes that gap;
these tests cover the parts that are easy to get wrong: the frontmatter
subset parser (no PyYAML, per the repo's no-dependency rule) and the atomic
write that keeps a watched config from losing whole sections.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "tools" / "install-agent.py"
AGENT_MD = ROOT / "agent" / "ruth.md"

_spec = importlib.util.spec_from_file_location("install_agent", INSTALLER)
ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ia)


class TestFrontmatterParser(unittest.TestCase):
    def test_parses_scalars(self):
        fm, _ = ia.split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
        a = ia.parse_frontmatter(fm)
        self.assertEqual(a["mode"], "primary")
        self.assertEqual(a["steps"], 64)
        self.assertIsInstance(a["steps"], int)
        self.assertTrue(a["description"])

    def test_parses_tools_as_bools(self):
        fm, _ = ia.split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
        tools = ia.parse_frontmatter(fm)["tools"]
        self.assertTrue(tools["read"])
        self.assertTrue(tools["bash"])
        for name, val in tools.items():
            self.assertIsInstance(val, bool, name)

    def test_parses_every_permission_rule(self):
        """Regression: `- action: x` puts the dash on the first key's line.
        A parser that only reads `key: value` lines dropped `action` from
        every rule, which installed Ruth with NO permissions at all."""
        fm, _ = ia.split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
        perms = ia.parse_frontmatter(fm)["permissions"]
        self.assertEqual(
            [(p.get("action"), p.get("resource"), p.get("effect"))
             for p in perms],
            [("*", "*", "allow"),
             ("subagent", "*", "deny"),
             ("question", "*", "deny"),
             ("external_directory", "*", "ask")])
        for rule in perms:
            self.assertTrue(rule.get("action"), f"rule lost its action: {rule}")
            self.assertTrue(rule.get("effect"), f"rule lost its effect: {rule}")

    def test_installed_agent_keeps_its_permissions(self):
        """End to end: the agent that lands in a config is not stripped."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pathlib.Path(tmp) / "opencode.json"
            cfg.write_text(json.dumps({"agents": {}, "mcp": {}}))
            subprocess.run([sys.executable, str(INSTALLER), "--config", str(cfg),
                            "--install"], capture_output=True, text=True,
                           timeout=90, check=True)
            perms = json.loads(cfg.read_text())["agents"]["ruth"]["permissions"]
            self.assertGreaterEqual(len(perms), 4)
            self.assertIn({"action": "question", "resource": "*", "effect": "deny"},
                          perms)

    def test_ignores_comments_and_blank_lines(self):
        fm, _ = ia.split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
        a = ia.parse_frontmatter("# a comment\n\n" + fm)
        self.assertEqual(a["mode"], "primary")

    def test_missing_frontmatter_is_a_clear_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write("no frontmatter here")
            path = pathlib.Path(fh.name)
        try:
            with self.assertRaises(SystemExit):
                ia.split_frontmatter(path.read_text())
        finally:
            path.unlink()


class TestBuildAgent(unittest.TestCase):
    def test_system_is_her_body_verbatim(self):
        agent = ia.build_agent()
        _, body = ia.split_frontmatter(AGENT_MD.read_text(encoding="utf-8"))
        self.assertEqual(agent["system"], body)

    def test_carries_the_load_bearing_tools(self):
        tools = ia.build_agent()["tools"]
        for t in ("read", "grep", "glob", "bash", "skill", "webfetch", "edit"):
            self.assertIn(t, tools, t)


class TestInstallIsAtomicAndLossless(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, str(INSTALLER), *args],
                              capture_output=True, text=True, timeout=90)

    def test_check_reports_drift_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pathlib.Path(tmp) / "opencode.json"
            cfg.write_text(json.dumps({"agents": {}, "mcp": {"github": {}}}))
            before = cfg.read_text()
            r = self._run("--config", str(cfg), "--check")
            self.assertNotEqual(r.returncode, 0, "drift should be reported")
            self.assertEqual(cfg.read_text(), before, "--check must not write")

    def test_install_preserves_unrelated_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pathlib.Path(tmp) / "opencode.json"
            original = {"agents": {"build": {"disabled": True}},
                        "mcp": {"github": {"url": "https://x"},
                                "lsp": {"command": ["py", "b.py"]}},
                        "skills": ["./s"]}
            cfg.write_text(json.dumps(original))
            r = self._run("--config", str(cfg), "--install")
            self.assertEqual(r.returncode, 0, r.stderr)
            back = json.loads(cfg.read_text())
            self.assertEqual(back["mcp"], original["mcp"], "mcp was damaged")
            self.assertEqual(back["skills"], original["skills"], "skills were lost")
            self.assertIn("build", back["agents"])
            self.assertIn("ruth", back["agents"])

    def test_install_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pathlib.Path(tmp) / "opencode.json"
            cfg.write_text(json.dumps({"agents": {}, "mcp": {}}))
            self.assertEqual(self._run("--config", str(cfg), "--install").returncode, 0)
            second = self._run("--config", str(cfg), "--install")
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertIn("already installed", second.stdout)

    def test_written_instructions_are_absolute_and_exist(self):
        """A relative path resolves against the wrong directory and her soul
        silently never loads."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = pathlib.Path(tmp) / "opencode.json"
            cfg.write_text(json.dumps({"agents": {}, "mcp": {}}))
            self._run("--config", str(cfg), "--install")
            ins = json.loads(cfg.read_text())["instructions"]
            self.assertTrue(ins)
            for rel in ins:
                self.assertTrue(rel.startswith("/"), f"{rel} is not absolute")
                self.assertTrue(pathlib.Path(rel).exists(), f"{rel} does not exist")

    def test_no_absolute_path_from_another_machine_in_the_installer(self):
        import re
        src = INSTALLER.read_text(encoding="utf-8")
        found = [m for m in re.findall(r"/home/[A-Za-z0-9_]+", src)
                 if not m.startswith("/home/user")]
        self.assertEqual(found, [], "the installer must not hardcode a home path")


if __name__ == "__main__":
    unittest.main()
