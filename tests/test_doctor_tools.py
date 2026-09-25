"""Tests for the doctor's tool-surface reporting.

Two things this guards:

1. `permission_effects` resolves the real permission. The top-level
   `"permission"` map is only half the story -- the agent's `permissions`
   array is evaluated after it and the last matching rule wins. Reading only
   the top-level map reported tools as available that the agent cannot call.

2. `tool_surface` separates "permitted" from "works". The built-in websearch
   tool is permitted on this box and still fails on every single call, because
   it needs a provider credential. A permission of "allow" is not health.
"""
import ast
import pathlib
import unittest

DOCTOR = pathlib.Path(__file__).resolve().parent.parent / "tools" / "doctor.py"


def _load(name, extra=()):
    tree = ast.parse(DOCTOR.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {name, *extra}]
    assert fns, f"{name} not found in tools/doctor.py"
    consts = []
    for n in tree.body:
        if not (isinstance(n, ast.Assign)
                and all(isinstance(t, ast.Name) for t in n.targets)):
            continue
        try:
            ast.literal_eval(n.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        consts.append(n)
    ns = {"os": __import__("os")}
    mod = ast.Module(body=consts + fns, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(DOCTOR), "exec"), ns)
    return ns[name]


permission_effects = _load("permission_effects")
tool_surface = _load("tool_surface", extra=("websearch_status", "mcp_entry",
                                         "permission_effects"))


class TestPermissionEffects(unittest.TestCase):
    def test_top_level_map_only(self):
        cfg = {"permission": {"read": "allow", "edit": "allow"}}
        e = permission_effects(cfg)
        self.assertEqual(e["read"], "allow")
        self.assertEqual(e["edit"], "allow")
        self.assertEqual(e["question"], "unset")

    def test_agent_deny_overrides_top_level_allow(self):
        """The real box: read/edit/bash allowed globally, question denied on
        the agent. Reading only the top-level map gets this wrong."""
        cfg = {"permission": {"read": "allow", "question": "allow"},
               "agents": {"ruth": {"permissions": [
                   {"action": "*", "resource": "*", "effect": "allow"},
                   {"action": "question", "resource": "*", "effect": "deny"}]}}}
        e = permission_effects(cfg)
        self.assertEqual(e["read"], "allow")
        self.assertEqual(e["question"], "deny")

    def test_last_rule_wins(self):
        cfg = {"permissions": [
            {"action": "bash", "resource": "*", "effect": "deny"},
            {"action": "bash", "resource": "*", "effect": "allow"}]}
        self.assertEqual(permission_effects(cfg)["bash"], "allow")

    def test_resource_specific_deny_is_reported_not_hidden(self):
        """A per-resource deny (e.g. ~/.github_token) is a real protection but
        is not the action-level effect. It must be surfaced, not dropped."""
        cfg = {"permission": {"external_directory": {
            "*": "allow", "~/.github_token": "deny"}}}
        e = permission_effects(cfg, agent="nobody")
        self.assertEqual(e["external_directory"], "allow")
        self.assertIn("external_directory:~/.github_token=deny", e["_restricted"])

    def test_no_restrictions_reports_empty(self):
        self.assertEqual(permission_effects({})["_restricted"], [])

    def test_wildcard_action_grants(self):
        cfg = {"permissions": [{"action": "*", "resource": "*", "effect": "allow"}]}
        e = permission_effects(cfg)
        for action in ("read", "write", "execute", "skill"):
            self.assertEqual(e[action], "allow", action)

    def test_agent_ask_is_reported_as_ask(self):
        cfg = {"agents": {"ruth": {"permissions": [
            {"action": "*", "resource": "*", "effect": "allow"},
            {"action": "external_directory", "resource": "*", "effect": "ask"}]}}}
        self.assertEqual(permission_effects(cfg)["external_directory"], "ask")

    def test_missing_sections_do_not_raise(self):
        for cfg in ({}, {"agents": None}, {"permission": None},
                    {"permissions": None}, {"agents": {"ruth": None}}):
            e = permission_effects(cfg)
            self.assertIn("read", e)


class TestToolSurface(unittest.TestCase):
    BASE = {"permission": {"read": "allow", "edit": "allow", "bash": "allow",
                           "websearch": "allow"},
            "agents": {"ruth": {"permissions": [
                {"action": "*", "resource": "*", "effect": "allow"},
                {"action": "subagent", "resource": "*", "effect": "deny"},
                {"action": "question", "resource": "*", "effect": "deny"}]}}}

    def test_permitted_but_broken_is_visible(self):
        """websearch is allowed yet cannot work without a provider key.
        Both facts must be reported, or the tool looks healthy."""
        t = tool_surface(dict(self.BASE, websearch={"provider": "random"}), {})
        self.assertEqual(t["tools_builtin_websearch"], "allow")
        self.assertFalse(t["tools_builtin_websearch_works"])

    def test_working_builtin_search(self):
        cfg = dict(self.BASE, websearch={"provider": "tavily"})
        t = tool_surface(cfg, {"TAVILY_API_KEY": "k"})
        self.assertTrue(t["tools_builtin_websearch_works"])

    def test_mcp_search_registered_separately(self):
        cfg = dict(self.BASE,
                   mcp={"websearch": {"command": ["/usr/bin/python3", "ws.py"]}})
        t = tool_surface(cfg, {})
        self.assertTrue(t["tools_websearch_mcp"])
        self.assertFalse(t["tools_builtin_websearch_works"])

    def test_restricted_paths_are_reported(self):
        cfg = dict(self.BASE, permission={
            "external_directory": {"*": "allow", "~/.github_token": "deny"}})
        t = tool_surface(cfg, {})
        self.assertIn("external_directory:~/.github_token=deny", t["tools_restricted"])

    def test_denied_tools_are_surfaced(self):
        t = tool_surface(dict(self.BASE), {})
        self.assertEqual(t["tools_question"], "deny")
        self.assertEqual(t["tools_subagent"], "deny")
        self.assertEqual(t["tools_read"], "allow")

    def test_lists_mcp_servers_and_skill_sources(self):
        cfg = dict(self.BASE, mcp={"lsp": {"command": ["x"]}}, skills=["./s"])
        t = tool_surface(cfg, {})
        self.assertEqual(t["tools_mcp_servers"], ["lsp"])
        self.assertEqual(t["tools_skills_configured"], 1)

    def test_empty_config_does_not_raise(self):
        t = tool_surface({}, {})
        self.assertEqual(t["tools_mcp_servers"], [])
        self.assertEqual(t["tools_skills_configured"], 0)


if __name__ == "__main__":
    unittest.main()
