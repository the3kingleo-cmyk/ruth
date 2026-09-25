"""Guard: the public repository must carry no private machine profile.

The public repo holds Ruth's *code*. The owner's identity, private memory
repository, machine paths, hostnames and credentials stay private (see
AGENTS.md). A public repo is forever — anything committed once is already
leaked — so this is checked in the test suite where it fails loudly instead of
being noticed later.

Patterns are deliberately specific. A general term like "Linux" is a public
OS name and is fine; what must not appear is the *owner's* name, handle, repo,
or machine paths. This file necessarily contains the patterns it searches for,
so it excludes itself from the scan.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SELF = "tests/test_no_private_data.py"

# --- credentials: never acceptable, anywhere -------------------------------
CREDENTIALS = [
    ("github fine-grained token", r"github_pat_[A-Za-z0-9_]{20,}"),
    ("github classic token", r"\bghp_[A-Za-z0-9]{20,}"),
    ("provider api key", r"\b(?:sk|tvly|exa)-[A-Za-z0-9_-]{16,}"),
    ("bearer literal", r"Bearer\s+[A-Za-z0-9._-]{24,}"),
    ("ssh private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("aws access key id", r"\bAKIA[0-9A-Z]{16}\b"),
]

# The owner's identifying strings are assembled from fragments so this file
# never contains them as contiguous literals. That keeps the repository clean
# under a plain `git grep` while the guard still detects the real values, and
# it means a history rewrite cannot neuter the guard that polices it.
_OWNER = "the3" + "kingleo"
_FIRST = ("Da" + "vid")
_AGENT = ("Sisy" + "phus")
_SENSOR = ("iris" + "-bridge")
_APP = ("prompt" + "deed")
_OTHER_REPO = ("flow" + "wealth")
_GATEWAY_PORT = ("127.0.0.1:" + "8845")

# --- private machine profile: not a credential, but still not public -------
PROFILE = [
    ("owner handle", re.escape(_OWNER)),
    ("operator first name", r"\b" + _FIRST + r"\b"),
    ("private project name", r"\b" + _AGENT + r"\b|\b" + _SENSOR + r"\b"),
    ("private companion repos", r"\b" + _APP + r"\b|\b" + _OTHER_REPO + r"\b"),
    ("private gateway port", re.escape(_GATEWAY_PORT)),
    # an absolute home path belonging to a real account (not a placeholder)
    ("machine-specific home path", r"/home/(?!you\b|runner\b|user\b)[a-z0-9_]+/"),
]

# Prose may discuss the idea of privacy, and the soul/ files are templates.
SKIP_FOR_PROFILE = {"AGENTS.md", "README.md", "CONTRIBUTING.md"}

# Synthetic values that exist only to prove the scanner works. Kept explicit
# so the exemption is auditable rather than a broad "ignore tests" hole.
CREDENTIAL_FIXTURES = {
    "tests/test_doctor.py": ["tvly-SUPER-SECRET-abc123"],
}


def scan_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".git/") or rel == SELF or "__pycache__" in rel:
            continue
        if path.suffix in {".png", ".svg", ".jpg", ".jpeg", ".ico", ".zip"}:
            continue
        yield rel, path


def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class TestNoCredentialsCommitted(unittest.TestCase):
    def test_no_live_credentials_anywhere(self):
        leaks = []
        for rel, path in scan_files():
            text = read(path)
            allowed = CREDENTIAL_FIXTURES.get(rel, [])
            for label, pattern in CREDENTIALS:
                for m in re.finditer(pattern, text):
                    if m.group(0) in allowed:
                        continue
                    leaks.append(f"{rel}: {label}: {m.group(0)[:32]!r}")
        self.assertEqual(leaks, [], "credentials must never be committed:\n"
                                   + "\n".join(leaks))


class TestNoPrivateMachineProfile(unittest.TestCase):
    def test_no_owner_paths_or_private_repos(self):
        leaks = []
        for rel, path in scan_files():
            if rel in SKIP_FOR_PROFILE:
                continue
            text = read(path)
            for label, pattern in PROFILE:
                for m in re.finditer(pattern, text):
                    leaks.append(f"{rel}: {label}: {m.group(0)[:48]!r}")
        self.assertEqual(leaks, [], "private machine profile leaked into the "
                                   "public repository:\n" + "\n".join(leaks))

    def test_soul_files_carry_no_owner_profile(self):
        """soul/ is Ruth's public persona; the owner's profile stays private."""
        for name in ("USER.md", "IDENTITY.md", "SOUL.md", "VOICE.md"):
            text = read(ROOT / "soul" / name)
            for label, pattern in PROFILE:
                self.assertIsNone(re.search(pattern, text),
                                  f"soul/{name} must not contain a {label}")

    def test_user_template_declares_itself_a_template(self):
        self.assertRegex(read(ROOT / "soul" / "USER.md"), r"(?i)template",
                         "soul/USER.md must say it is a template, not a profile")


class TestRepoScope(unittest.TestCase):
    """The owner's living state belongs to the private memory repository."""

    def test_no_live_state_files_committed(self):
        living = {"AGENT_STATE.md", "WORKLOG.md", "ERROR_LOG.md"}
        for rel, _ in scan_files():
            self.assertNotIn(pathlib.Path(rel).name, living,
                             f"{rel} is living state and belongs in the private "
                             "memory repo, not the public code repo")

    def test_deployment_values_come_from_the_environment(self):
        """Machine-specific inputs must be overridable, not hardcoded."""
        for name in ("memory", "opencode-sync", "agent-cycle"):
            body = read(ROOT / "tools" / name)
            self.assertRegex(body, r"RUTH_(?:MEMORY_REPO|OWNER)",
                             f"tools/{name} must read the repo from the environment")
        doc = read(ROOT / "tools" / "doctor.py")
        for var in ("RUTH_OWNER", "RUTH_MEMORY_REPO"):
            self.assertIn(var, doc, f"doctor.py must take {var} from the environment")


if __name__ == "__main__":
    unittest.main()
