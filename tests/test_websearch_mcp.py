"""Tests for the keyless websearch MCP server.

The network-dependent paths are not exercised here (no outbound calls in the
test suite); what's covered is the MCP framing and the DuckDuckGo HTML
parsing, which is where the real bugs were: a handler that crashed on an
already-parsed frame killed the server and surfaced to the client as
"Connection closed".
"""
import ast
import importlib.util
import json
import pathlib
import sys
import unittest

SERVER = pathlib.Path(__file__).resolve().parent.parent / "tools" / "websearch_mcp.py"

_spec = importlib.util.spec_from_file_location("websearch_mcp", SERVER)
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)

DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fbun.sh%2F&amp;rut=xyz">
     Bun &mdash; JavaScript Runtime
  </a>
  <a class="result__snippet">
     Bundle, install, and run JavaScript &amp; TypeScript.
  </a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/plain">Plain Link</a>
  <div class="result__snippet">A direct href result.</div>
</div>
</body></html>
"""


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.p = ws._Results()
        self.p.feed(DDG_HTML)
        self.results = self.p.results

    def test_finds_both_results(self):
        self.assertEqual(len(self.results), 2)

    def test_title_is_unescaped_and_collapsed(self):
        self.assertEqual(self.results[0]["title"], "Bun — JavaScript Runtime")

    def test_uddg_redirect_is_unwrapped(self):
        self.assertEqual(self.results[0]["url"], "https://bun.sh/")

    def test_plain_href_is_kept(self):
        self.assertEqual(self.results[1]["url"], "https://example.com/plain")

    def test_snippet_attaches_to_its_result(self):
        self.assertIn("JavaScript", self.results[0]["snippet"])
        self.assertIn("direct href", self.results[1]["snippet"])

    def test_non_result_links_ignored(self):
        doc = '<a class="result__a" href="ftp://x">x</a><a class="nav" href="https://y">y</a>'
        p = ws._Results()
        p.feed(doc)
        self.assertEqual([r["url"] for r in p.results], [])


class TestToolContract(unittest.TestCase):
    def test_handle_accepts_parsed_dict(self):
        """Regression: passing a dict used to raise TypeError and kill the
        server, which the client reported only as 'Connection closed'."""
        out = ws.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18"}})
        self.assertIsNone(out)  # responds on stdout, returns nothing

    def test_handle_accepts_raw_string(self):
        ws.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}))

    def test_handle_ignores_non_dict(self):
        self.assertIsNone(ws.handle([1, 2, 3]))
        self.assertIsNone(ws.handle("not json at all"))

    def test_every_handler_is_declared_in_tools(self):
        declared = {t["name"] for t in ws.TOOLS}
        self.assertEqual(declared, set(ws.HANDLERS))

    def test_tools_have_descriptions_and_schemas(self):
        for t in ws.TOOLS:
            self.assertTrue(t.get("description"), t["name"])
            self.assertEqual(t["inputSchema"]["type"], "object", t["name"])

    def test_search_requires_query(self):
        for bad in ("", "   ", None):
            with self.subTest(query=bad):
                self.assertIn("error", ws.tool_search(bad))

    def test_max_results_clamping(self):
        # Regression: int(n or 8) turned an explicit 0 into 8, so a caller
        # asking for zero results silently got eight.
        self.assertEqual(ws._clamp_results(0), 1)
        self.assertEqual(ws._clamp_results(-3), 1)
        self.assertEqual(ws._clamp_results(None), 8)
        self.assertEqual(ws._clamp_results(""), 8)
        self.assertEqual(ws._clamp_results("abc"), 8)
        self.assertEqual(ws._clamp_results(5), 5)
        self.assertEqual(ws._clamp_results(999), 25)
        self.assertEqual(ws._clamp_results(26), 25)

    def test_fetch_rejects_non_http(self):
        self.assertIn("error", ws.tool_fetch("file:///etc/passwd"))
        self.assertIn("error", ws.tool_fetch("javascript:alert(1)"))


class TestSearchErrorSurface(unittest.TestCase):
    def test_all_backends_failing_reports_errors(self):
        res, err = ws._search("q", 5, [
            ("a", lambda: (_ for _ in ()).throw(OSError("down"))),
            ("b", lambda: []),
        ])
        self.assertEqual(res, [])
        self.assertIn("down", err)
        self.assertIn("no results", err)

    def test_first_successful_backend_wins(self):
        res, err = ws._search("q", 5, [
            ("a", lambda: []),
            ("b", lambda: [{"title": "t", "url": "https://u", "snippet": "s"}]),
        ])
        self.assertEqual(len(res), 1)
        self.assertIsNone(err)


class TestNoCredentialsRequired(unittest.TestCase):
    def test_module_never_reads_api_key_envs(self):
        src = SERVER.read_text(encoding="utf-8")
        for var in ("EXA_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY",
                    "PARALLEL_API_KEY", "GITHUB_TOKEN"):
            self.assertNotIn(var, src, f"{var} must not be referenced")

    def test_module_only_uses_stdlib_imports(self):
        tree = ast.parse(SERVER.read_text(encoding="utf-8"))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        self.assertTrue(mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names))


if __name__ == "__main__":
    unittest.main()
