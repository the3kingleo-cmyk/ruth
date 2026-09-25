"""Tests for the local-model layer: `tools/gemma` and its boundaries.

The point of these is the boundary. Ruth's mind is token-free and offline by
construction, and tests/test_token_free.py enforces numpy + stdlib inside the
package. A local LLM is therefore *not* part of her: it belongs to the agent
layer, lives outside ruth/, and is reached over HTTP. These tests hold that
line, so a future "just add it to her" does not quietly break her guarantee.

The model itself is a 2.49 GB download; nothing here needs it present.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEMMA = ROOT / "tools" / "gemma"
PKG = ROOT / "ruth"


class TestGemmaLauncher(unittest.TestCase):
    def setUp(self):
        self.text = GEMMA.read_text(encoding="utf-8")

    def test_is_executable_shell(self):
        self.assertTrue(os.access(GEMMA, os.X_OK), "gemma must be executable")
        done = subprocess.run(["bash", "-n", str(GEMMA)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_targets_a_model_that_fits_this_box(self):
        """6.4 GB RAM, no GPU. Measured: 1B answers in 2.6s, 4B cannot."""
        self.assertIn("gemma-3-4b-it-Q4_K_M.gguf", self.text)
        self.assertIn("gemma-3-1b-it-Q4_K_M.gguf", self.text)
        self.assertRegex(self.text, r"Q4_K_M",
                         "Q4_K_M is the quantisation that fits; Q8 does not")

    def test_never_downloads_an_unquantised_model(self):
        for bad in ("F16.gguf", "-BF16.gguf", "Q2_K", "Q3_K"):
            self.assertNotIn(bad, self.text,
                             f"{bad} will not fit in 6.4 GB of RAM")

    def test_caps_context_so_the_kv_cache_cannot_exhaust_ram(self):
        self.assertRegex(self.text, r"GEMMA_CTX",
                         "context must be bounded on a 6.4 GB box")
        m = re.search(r'GEMMA_CTX:-(\d+)', self.text)
        self.assertIsNotNone(m)
        ctx = int(m.group(1))
        # opencode's own agent prompt measured 8,105 tokens, so a smaller
        # window cannot serve as an agent at all...
        self.assertGreaterEqual(ctx, 8192,
                                "opencode's agent prompt alone is ~8,105 tokens")
        # ...and a much larger KV cache will not fit in 6.4 GB alongside the
        # opencode stack.
        self.assertLessEqual(ctx, 16384, "KV cache must not exhaust 6.4 GB")

    def test_serves_openai_compatible_http(self):
        """opencode and most harnesses speak OpenAI-shaped HTTP; that is how
        the agent layer reaches the local model."""
        self.assertIn("llama-server", self.text)
        self.assertIn("/v1/models", self.text)
        self.assertIn("--port", self.text)

    def test_models_live_outside_the_package(self):
        self.assertIn(".local/share/models", self.text,
                      "a 2.49 GB model must not sit in the source repo")

    def test_has_a_fallback_when_no_model_is_downloaded(self):
        """Running `gemma serve` with nothing downloaded must say so, not
        fail obscurely."""
        self.assertRegex(self.text, r"gemma fetch")


class TestTheBoundaryHolds(unittest.TestCase):
    """No model runtime may enter ruth/ -- that is her guarantee."""

    def test_nothing_in_the_package_imports_a_model_runtime(self):
        banned = ("llama_cpp", "llama", "ggml", "transformers", "torch",
                  "onnxruntime", "ctransformers", "vllm", "gemma", "sentencepiece")
        for path in PKG.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for mod in banned:
                self.assertNotRegex(
                    text, rf"^\s*(import|from)\s+{mod}\b",
                    f"{path} imports {mod}: her runtime is numpy + stdlib only")

    def test_the_launcher_is_not_inside_the_package(self):
        self.assertTrue(str(GEMMA).startswith(str(ROOT / "tools")),
                        "the launcher belongs in tools/, not inside ruth/")

    def test_her_package_never_shells_out_to_a_model(self):
        for path in PKG.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in ("llama-server", "llama-server ", "localhost:8077", "gemma"):
                self.assertNotIn(token, text,
                                 f"{path} references the local model: her mind "
                                 f"must not depend on one")

    def test_the_token_free_suite_still_passes(self):
        """The real guard, not a paraphrase of it."""
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_token_free.py", "-q"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout[-2000:])


if __name__ == "__main__":
    unittest.main()


class TestTheMeasuredChoice(unittest.TestCase):
    """The default must be the model that was measured to work here.

    Numbers taken from this box (4 vCPU i3-10110U, no GPU, 6.4 GB):
      gemma-3-1b-it Q4 (0.81 GB) -> 50 tokens in 2.6 s
      gemma-3-4b-it Q4 (2.49 GB) -> 0.09-0.59 tok/s; a 60-token answer
                                   exceeded four minutes
    Shipping the 4B as the default would mean a default that cannot answer.
    """

    def setUp(self):
        self.text = (ROOT / "tools" / "gemma").read_text(encoding="utf-8")

    def test_default_is_the_one_that_measured_usable(self):
        m = re.search(r'^DEFAULT_MODEL="(.+)"$', self.text, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "gemma-3-1b-it-Q4_K_M.gguf",
                         "the 4B is downloaded but cannot answer on this box")

    def test_the_4b_is_still_available_opt_in(self):
        self.assertIn('BIG_MODEL="gemma-3-4b-it-Q4_K_M.gguf"', self.text)
        self.assertIn('"$MODEL_DIR/$BIG_MODEL"; return', self.text)
        self.assertIn('[ "${GEMMA_MODEL:-}" = "4b" ]', self.text,
                         "the 4B must remain reachable as an opt-in")

    def test_both_are_fetched_so_both_are_available(self):
        self.assertRegex(self.text, r'for f in "\$DEFAULT_MODEL" "\$BIG_MODEL"')

    def test_the_measured_numbers_are_recorded_in_the_file(self):
        # a future change to the default should have to confront the measurement
        self.assertIn("2.6 s", self.text)
        self.assertIn("0.09", self.text)


class TestTheConnection(unittest.TestCase):
    """A downloaded model that nothing can reach is a brain in a box.

    This is the third time in this project something got installed and
    reported done while being unreachable: Ruth's mind (installed, never
    connected to the agent), the MCP servers, and then Gemma itself -- which
    served perfectly on 127.0.0.1:8077 while `opencode models` showed zero of
    it and the config had no `provider` key at all.

    So the provider registration is pinned. If someone removes it, the model
    goes back to being decorative.
    """

    def setUp(self):
        cfg_path = pathlib.Path(os.path.expanduser(
            "~/.config/opencode/opencode.json"))
        if not cfg_path.exists():
            self.skipTest("no opencode config on this box")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    def test_a_local_provider_is_registered(self):
        self.assertIn("provider", self.cfg,
                      "the local model is unreachable without a provider entry")
        self.assertIn("local", self.cfg["provider"])

    def test_the_provider_points_at_the_running_endpoint(self):
        opts = self.cfg["provider"]["local"].get("options", {})
        self.assertIn("127.0.0.1:8077", opts.get("baseURL", ""),
                      "baseURL must be where llama-server actually listens")
        self.assertTrue(opts.get("baseURL", "").endswith("/v1"),
                        "the OpenAI-compatible baseURL ends in /v1")

    def test_the_models_are_advertised_with_tool_calling(self):
        models = self.cfg["provider"]["local"].get("models", {})
        self.assertIn("gemma-3-1b", models)
        self.assertTrue(models["gemma-3-1b"].get("tool_call"),
                        "an agent model that cannot call tools is a chat toy")

    def test_context_is_large_enough_for_an_agent_prompt(self):
        """opencode's own prompt (system + tool schemas) measured 8,105
        tokens. A smaller context cannot serve as an agent at all."""
        limit = self.cfg["provider"]["local"]["models"]["gemma-3-1b"]["limit"]
        self.assertGreaterEqual(limit["context"], 8192,
                                "opencode's agent prompt alone is ~8,105 tokens")

    def test_registering_the_provider_did_not_break_anything(self):
        for section in ("mcp", "agents", "permission"):
            self.assertIn(section, self.cfg,
                          f"adding a provider must not drop {section}")
        for server in ("github", "lsp", "websearch", "acp", "mind"):
            self.assertIn(server, self.cfg["mcp"],
                          f"mcp.{server} was lost while adding the provider")
