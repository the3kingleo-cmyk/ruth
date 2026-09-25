"""Regression tests for the doctor's MCP-server lookup.

`tools/doctor.py` is a standalone top-level script, so importing it would run
the entire health check (network calls, AGENT_STATE.md writes). These tests
lift just the `mcp_entry` helper out of the source with `ast` and exercise it
in isolation.

The bug this guards: the doctor read MCP servers only from `mcp.servers.<name>`
while opencode v2 registers them at `mcp.<name>`. A healthy, working LSP
bridge was therefore reported as unregistered, which silently dropped it from
the health gate — the same wrong assumption that had put `lsp` in the wrong
place in the live config.
"""
import ast
import pathlib
import unittest

DOCTOR = pathlib.Path(__file__).resolve().parent.parent / "tools" / "doctor.py"


def _load(name):
    """Exec a single top-level def from doctor.py in isolation.

    Module-level constants are carried along, since a lifted function may
    reference them (e.g. WS_PROVIDERS).
    """
    tree = ast.parse(DOCTOR.read_text(encoding="utf-8"))
    fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    assert fn is not None, f"{name} not found in tools/doctor.py"
    consts = []
    for n in tree.body:
        if not (isinstance(n, ast.Assign)
                and all(isinstance(t, ast.Name) for t in n.targets)):
            continue
        try:                       # keep literals only (skip e.g. HOME = os.path...)
            ast.literal_eval(n.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        consts.append(n)
    module = ast.Module(body=consts + [fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(DOCTOR), "exec"), ns)
    return ns[name]


mcp_entry = _load("mcp_entry")
websearch_status = _load("websearch_status")


class TestMcpEntry(unittest.TestCase):
    def test_v2_flat_shape(self):
        cfg = {"mcp": {"lsp": {"type": "local",
                               "command": ["/usr/bin/python3", "bridge.py"]}}}
        self.assertEqual(mcp_entry(cfg, "lsp")["command"],
                         ["/usr/bin/python3", "bridge.py"])

    def test_legacy_nested_shape(self):
        cfg = {"mcp": {"servers": {"lsp": {"command": ["bridge.py"]}}}}
        self.assertEqual(mcp_entry(cfg, "lsp")["command"], ["bridge.py"])

    def test_flat_wins_over_nested(self):
        cfg = {"mcp": {"lsp": {"command": ["flat.py"]},
                       "servers": {"lsp": {"command": ["nested.py"]}}}}
        self.assertEqual(mcp_entry(cfg, "lsp")["command"], ["flat.py"])

    def test_missing_server_is_empty(self):
        self.assertEqual(mcp_entry({"mcp": {"github": {}}}, "lsp"), {})

    def test_missing_mcp_section_is_empty(self):
        self.assertEqual(mcp_entry({}, "lsp"), {})

    def test_malformed_mcp_section_does_not_raise(self):
        # A corrupt config must not crash the doctor: one red check would
        # otherwise hide every other check.
        for bad in (None, [], "nonsense", 7):
            self.assertEqual(mcp_entry({"mcp": bad}, "lsp"), {})

    def test_malformed_servers_section_does_not_raise(self):
        cfg = {"mcp": {"servers": "nonsense", "lsp": {"command": ["x.py"]}}}
        self.assertEqual(mcp_entry(cfg, "lsp")["command"], ["x.py"])

    def test_github_remote_url_shape(self):
        cfg = {"mcp": {"github": {"type": "remote", "url": "https://example/mcp"}}}
        self.assertEqual(mcp_entry(cfg, "github")["url"], "https://example/mcp")


class TestDoctorReportsLspRegistered(unittest.TestCase):
    """The health-gate expressions must see a v2-shaped lsp entry."""

    @staticmethod
    def enabled(entry):
        registered = bool(entry.get("command"))
        return bool(registered
                    and entry.get("enabled", True) is not False
                    and entry.get("disabled", False) is not True)

    def test_registered_and_enabled_for_v2_config(self):
        cfg = {"mcp": {"lsp": {"type": "local", "command": ["bridge.py"],
                               "enabled": True}}}
        entry = mcp_entry(cfg, "lsp")
        self.assertTrue(bool(entry.get("command")))
        self.assertTrue(self.enabled(entry))

    def test_disabled_entry_is_not_enabled(self):
        cfg = {"mcp": {"lsp": {"command": ["bridge.py"], "enabled": False}}}
        self.assertFalse(self.enabled(mcp_entry(cfg, "lsp")))

    def test_legacy_disabled_flag_is_not_enabled(self):
        cfg = {"mcp": {"lsp": {"command": ["bridge.py"], "disabled": True}}}
        self.assertFalse(self.enabled(mcp_entry(cfg, "lsp")))

    def test_commandless_entry_is_not_enabled(self):
        cfg = {"mcp": {"lsp": {"type": "local"}}}
        self.assertFalse(self.enabled(mcp_entry(cfg, "lsp")))


class TestWebsearchStatus(unittest.TestCase):
    """websearch reports enabled/keyed/ready separately.

    The point of splitting these: a config naming a provider with no key set
    is the half-configured state where the tool is advertised to the model but
    every live search fails. That must not read as healthy.
    """

    KEYED = {"TAVILY_API_KEY": "t"}

    def test_named_provider_without_key_is_not_ready(self):
        r = websearch_status({"websearch": {"provider": "tavily"}}, {})
        self.assertTrue(r["websearch_enabled"])
        self.assertEqual(r["websearch_provider"], "tavily")
        self.assertFalse(r["websearch_key"])
        self.assertFalse(r["websearch_ready"])

    def test_named_provider_with_its_key_is_ready(self):
        r = websearch_status({"websearch": {"provider": "tavily"}}, self.KEYED)
        self.assertTrue(r["websearch_ready"])

    def test_named_provider_ignores_another_providers_key(self):
        # provider=exa but only TAVILY_API_KEY is set -> not ready
        r = websearch_status({"websearch": {"provider": "exa"}}, self.KEYED)
        self.assertFalse(r["websearch_key"])
        self.assertFalse(r["websearch_ready"])

    def test_random_provider_with_any_key_is_ready(self):
        r = websearch_status({"websearch": {"provider": "random"}}, self.KEYED)
        self.assertEqual(r["websearch_provider"], "random")
        self.assertTrue(r["websearch_ready"])

    def test_random_provider_without_any_key_is_not_ready(self):
        r = websearch_status({"websearch": {"provider": "random"}}, {})
        self.assertTrue(r["websearch_enabled"])
        self.assertFalse(r["websearch_ready"])

    def test_disabled_websearch(self):
        r = websearch_status({"websearch": False}, self.KEYED)
        self.assertFalse(r["websearch_enabled"])
        self.assertEqual(r["websearch_provider"], "disabled")
        # a key cannot make a disabled tool ready
        self.assertFalse(r["websearch_ready"])

    def test_absent_section_defaults_to_random(self):
        r = websearch_status({}, {})
        self.assertEqual(r["websearch_provider"], "default")
        self.assertFalse(r["websearch_ready"])

    def test_empty_provider_value_treated_as_random(self):
        r = websearch_status({"websearch": {"provider": ""}}, self.KEYED)
        self.assertEqual(r["websearch_provider"], "random")
        self.assertTrue(r["websearch_ready"])

    def test_secret_value_is_never_returned(self):
        # The doctor pushes this ledger to GitHub, so a key value must never
        # reach the returned mapping.
        secret = "tvly-SUPER-SECRET-abc123"
        r = websearch_status({"websearch": {"provider": "tavily"}},
                             {"TAVILY_API_KEY": secret})
        self.assertTrue(r["websearch_key"])
        self.assertNotIn(secret, "".join(map(str, r.values())))
        self.assertNotIn(secret, "".join(f"{k}={v}" for k, v in r.items()))

    def test_all_four_documented_providers_resolve(self):
        for prov, var in (("exa", "EXA_API_KEY"),
                          ("firecrawl", "FIRECRAWL_API_KEY"),
                          ("parallel", "PARALLEL_API_KEY"),
                          ("tavily", "TAVILY_API_KEY")):
            with self.subTest(provider=prov):
                r = websearch_status({"websearch": {"provider": prov}},
                                     {var: "k"})
                self.assertTrue(r["websearch_ready"])


if __name__ == "__main__":
    unittest.main()
