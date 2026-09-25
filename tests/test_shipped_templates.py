"""The shipped templates must be valid, and must not encode one machine.

`opencode.json` and `agent/ruth.md` are what anyone installs to become Ruth.
Two defects shipped in them for a long time because nothing read them:

- the MCP entry sat at `mcp.servers.lsp`, the v1 shape that opencode v2
  silently ignores, so a fresh install got no language server at all and the
  config appeared to work while doing nothing;
- the agent declared only four tools, omitting read/grep/glob/skill/webfetch,
  which are load-bearing.

These tests read the real files, so the templates are covered by the same
suite that covers the code.
"""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "opencode.json"
AGENT = ROOT / "agent" / "ruth.md"

AGENT_TOOLS = ["read", "write", "edit", "patch", "bash", "glob", "grep", "list",
               "webfetch", "websearch", "skill", "execute"]


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, f"{path} has no YAML frontmatter"
    return m.group(1), text[m.end():]


class TestShippedConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_mcp_servers_use_the_v2_shape(self):
        """mcp.<name>, not mcp.servers.<name> -- v2 ignores the latter."""
        mcp = self.cfg.get("mcp", {})
        self.assertNotIn("servers", mcp,
                         "mcp.servers is the v1 shape; opencode v2 silently "
                         "ignores it, so the server never registers")
        for name, entry in mcp.items():
            self.assertIsInstance(entry, dict, name)
            self.assertTrue(entry.get("command") or entry.get("url"),
                            f"mcp.{name} has neither a command nor a url")

    def test_no_entry_is_explicitly_disabled(self):
        """v2 uses `enabled`; the v1 `disabled` key is ignored."""
        for name, entry in self.cfg.get("mcp", {}).items():
            self.assertNotIn("disabled", entry,
                             f"mcp.{name}: use enabled, not the v1 `disabled`")

    def test_declares_an_lsp_server(self):
        self.assertIn("lsp", self.cfg.get("mcp", {}),
                      "the LSP bridge is the whole point of the tools")

    def test_no_absolute_home_path(self):
        """A template must be portable, not pinned to one machine."""
        found = re.findall(r"/home/[A-Za-z0-9_]+", CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(found, [], "absolute home path in a shipped template")

    def test_default_agent_and_agents_present(self):
        self.assertIn(self.cfg.get("default_agent"), self.cfg.get("agents", {}))

    def test_build_plan_explore_general_disabled(self):
        for name in ("build", "plan", "explore", "general"):
            self.assertTrue(self.cfg["agents"][name].get("disabled"),
                            f"{name} must stay disabled: Ruth is the only agent")


class TestShippedAgent(unittest.TestCase):
    def setUp(self):
        self.fm, self.body = frontmatter(AGENT)

    def _fm_value(self, key):
        m = re.search(rf"^{key}:\s*(.+)$", self.fm, re.M)
        return m.group(1).strip() if m else None

    def test_declares_the_load_bearing_tools(self):
        """Only four were declared, which silently dropped read/grep/glob."""
        tools_block = re.search(r"^tools:\n((?:  \w+: true\n)+)", self.fm, re.M)
        self.assertIsNotNone(tools_block, "no tools block in agent frontmatter")
        declared = set(re.findall(r"(\w+): true", tools_block.group(1)))
        for tool in AGENT_TOOLS:
            self.assertIn(tool, declared, f"agent does not declare {tool}")

    def test_is_primary_and_not_ask_bound(self):
        self.assertEqual(self._fm_value("mode"), "primary")
        perms = re.findall(r"action:\s*(\S+).*?effect:\s*(\S+)", self.fm, re.S)
        self.assertIn(("question", "deny"), perms,
                      "Ruth runs unattended; question must stay denied")

    def test_body_documents_the_servers_she_actually_has(self):
        lowered = self.body.lower()
        for token in ("lsp", "websearch", "acp", "github", "skill"):
            self.assertIn(token, lowered,
                          f"the agent body never mentions {token}")

    def test_body_states_permitted_is_not_working(self):
        self.assertRegex(self.body, r"(?i)permitted is not working")


if __name__ == "__main__":
    unittest.main()
