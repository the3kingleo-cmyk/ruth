#!/usr/bin/env python3
"""opencode-doctor: verify API keys, MCP, config, LSP, timers, corrupt files; update AGENT_STATE.md in place."""
import json, os, platform, re, subprocess, sys, datetime, base64, time

HOME = os.path.expanduser("~")
CFG_ROOT = os.environ.get("OPENCODE_CONFIG_ROOT", f"{HOME}/.config/opencode")
# opencode has used both extensions across v1/v2; accept whichever exists.
CFG = next((p for p in (f"{CFG_ROOT}/opencode.jsonc", f"{CFG_ROOT}/opencode.json")
            if os.path.exists(p)), f"{CFG_ROOT}/opencode.jsonc")
STATE = f"{CFG_ROOT}/AGENT_STATE.md"
AUTH = f"{HOME}/.local/share/opencode/auth.json"
GH_MCP = "https://api.githubcopilot.com/mcp/"
# Owner/repo are deployment-specific; override with RUTH_OWNER/RUTH_MEMORY_REPO.
OWNER = os.environ.get("RUTH_OWNER", "OWNER")
MEMORY_REPO = os.environ.get("RUTH_MEMORY_REPO", f"{OWNER}/memory")
MEMORY_PATH = "state"
MEMORY_FILES = ["AGENT_STATE.md"]
PULL_FILES = ["WORKLOG.md", "ERROR_LOG.md"]

# auth.json is opencode-owned and absent on a fresh/rebuilt box. A health check
# must report a missing file, never crash on it — otherwise one red check hides
# every other check. Read it defensively.
try:
    _auth = json.load(open(AUTH, encoding="utf-8"))
except FileNotFoundError:
    _auth = {}
except (ValueError, OSError):
    _auth = {}
results = {}
results["auth_json_present"] = os.path.exists(AUTH)
OR_KEY = _auth.get("openrouter", {}).get("key", "")
# GitHub key: ~/.github_token is the single source (AGENTS.md). auth.json is
# owned by opencode, which rewrites it and strips non-provider entries.
GH_TOKEN_FILE = f"{HOME}/.github_token"
GH_KEY = open(GH_TOKEN_FILE, encoding="utf-8").read().strip() if os.path.exists(GH_TOKEN_FILE) else ""
if not GH_KEY:
    GH_KEY = _auth.get("github", {}).get("token", "")
HF_KEY = _auth.get("huggingface", {}).get("key", "")
OC_KEY = _auth.get("opencode", {}).get("key", "")
GH_API = f"https://api.github.com/repos/{MEMORY_REPO}/{MEMORY_PATH}"

def curl_json(url, headers, body=None, method=None):
    cmd = ["curl", "-s", "--max-time", "20"]
    if method: cmd += ["-X", method]
    for h in headers: cmd += ["-H", h]
    if body:
        cmd += ["-H", "Content-Type: application/json",
                "-H", "Accept: application/json, text/event-stream",
                "-d", json.dumps(body)]
    cmd += [url]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("data: "):
            try: return json.loads(line[6:])
            except Exception: return None
    try: return json.loads(out) if out.strip() else None
    except Exception: return None

results["machine_info"] = platform.node() or platform.machine()
results["home_dir"] = HOME
results["config_root"] = CFG
# 1. API keys
r = curl_json("https://openrouter.ai/api/v1/auth/key", [f"Authorization: Bearer {OR_KEY}"])
results["openrouter_key"] = bool(r and "data" in r)
r = curl_json("https://api.github.com/user", [f"Authorization: Bearer {GH_KEY}"])
results["github_key"] = bool(r and r.get("login"))
results["huggingface_key"] = bool(HF_KEY and HF_KEY.startswith("hf_"))
r = curl_json("https://opencode.ai/zen/v1/models", [f"Authorization: Bearer {OC_KEY}"]) if OC_KEY else None
results["opencode_zen_key"] = bool(r and "data" in r)
# Free brain door: cc-bridge on 127.0.0.1:0 (opencode zen + big-pickle).
# This is the box's actual model source since 2026-09-23 (no OpenRouter key);
# gate on it so "has a working model source" stays a real check.
try:
    _h = subprocess.run(["curl", "-s", "--max-time", "5", "-o", "/dev/null",
                         "-w", "%{http_code}", "http://127.0.0.1:0/health"],
                        capture_output=True, text=True).stdout.strip()
    results["zen_door"] = _h == "200"
except Exception:
    results["zen_door"] = False
# gumroad: env file locked down; token optional until operator generates it
_genv = f"{HOME}/.config/opencode/secrets/gumroad.env"
results["gumroad_env_secure"] = os.path.exists(_genv) and (os.stat(_genv).st_mode & 0o077) == 0
_gtok = ""
if os.path.exists(_genv):
    for _l in open(_genv):
        if _l.startswith("GUMROAD_ACCESS_TOKEN=") and _l.strip().split("=",1)[1]: _gtok = _l.strip().split("=",1)[1]
if _gtok:
    r = curl_json(f"https://api.gumroad.com/v2/user?access_token={_gtok}", [])
    results["gumroad_token"] = bool(r and r.get("success"))
else:
    results["gumroad_token"] = "pending (generate on app page)"
# 2. MCP
r = curl_json(GH_MCP, [f"Authorization: Bearer {GH_KEY}"], body={"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"doctor","version":"1"}}})
results["mcp_initialize"] = bool(r and "result" in r)
r = curl_json(GH_MCP, [f"Authorization: Bearer {GH_KEY}"], body={"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
results["mcp_tools"] = len(r.get("result", {}).get("tools", [])) if r else 0
# 3. config
raw = open(CFG, encoding="utf-8").read()
s = re.sub(r"(?m)^\s*//.*$", "", raw)          # full-line comments
s = re.sub(r"(?m)(?<=\s)//\s.*$", "", s)       # inline comments (never "://")
s = re.sub(r",(?=\s*[}\]])", "", s)           # trailing commas
cfg = json.loads(s)


def mcp_entry(config, name):
    """Return an MCP server definition from either schema shape.

    opencode v2 registers servers directly under `mcp.<name>`; an older
    nested form used `mcp.servers.<name>`. Reading only one of the two made
    this doctor report a healthy bridge as missing (and vice versa), so
    accept whichever is present.
    """
    mcp = config.get("mcp") or {}
    if not isinstance(mcp, dict):
        return {}
    flat = mcp.get(name)
    if isinstance(flat, dict):
        return flat
    nested = mcp.get("servers")
    if isinstance(nested, dict) and isinstance(nested.get(name), dict):
        return nested[name]
    return {}


# websearch provider id -> the env var holding its key (opencode's four
# built-in providers; each needs a key, none work anonymously).
WS_PROVIDERS = {"exa": "EXA_API_KEY", "firecrawl": "FIRECRAWL_API_KEY",
                "parallel": "PARALLEL_API_KEY", "tavily": "TAVILY_API_KEY"}


def websearch_status(config, env):
    """Describe websearch readiness without performing a search.

    `enabled` is the config switch, `key` is whether a usable provider
    credential exists, and `ready` is the only one that predicts whether a
    search will actually succeed. Keys are read from `env` and never
    returned — only their presence.
    """
    ws = config.get("websearch")
    if ws is False:
        return {"websearch_enabled": False, "websearch_provider": "disabled",
                "websearch_key": False, "websearch_ready": False}
    if isinstance(ws, dict):
        enabled = True
        provider = ws.get("provider") or "random"
    else:
        enabled = bool(ws)
        provider = "default"
    if provider in WS_PROVIDERS:
        # A named provider needs that provider's own key.
        keyed = bool(env.get(WS_PROVIDERS[provider]))
    else:
        # "random" / unset needs at least one provider key to choose from.
        keyed = any(env.get(v) for v in WS_PROVIDERS.values())
    return {"websearch_enabled": enabled, "websearch_provider": provider,
            "websearch_key": keyed,
            "websearch_ready": bool(enabled and keyed)}

results["instructions_loaded"] = all(os.path.exists(p) for p in cfg.get("instructions", [])) and len(cfg.get("instructions", [])) >= 6
results["compaction_on"] = bool(cfg.get("compaction", {}).get("auto"))
results["permission_allow"] = cfg.get("permission") == "allow" or (
    isinstance(cfg.get("permission"), dict)
    and cfg.get("permission", {}).get("read") == "allow"
    and cfg.get("permission", {}).get("edit") == "allow"
    and cfg.get("permission", {}).get("bash") == "allow"
)
# opencode v2 schema: "permissions": [...] array. The box runs allow-all actions
# + question:deny (zero prompts) + secret-file read denials — semantics-equivalent
# to the legacy "permission": allow posture the check below was built for.
_perms = cfg.get("permissions")
if isinstance(_perms, list):
    _eff = {p.get("action"): p.get("effect") for p in _perms if isinstance(p, dict)}
    if _eff.get("*") == "allow" and _eff.get("question") == "deny":
        results["permission_allow"] = True
# opencode v2 has no built-in LSP client. Real language feedback comes from the
# `lsp` MCP bridge, so "enabled" means: bridge registered and its servers present.
# The bridge must also not be explicitly disabled.
_bridge_srv = mcp_entry(cfg, "lsp")
results["lsp_bridge_registered"] = bool(_bridge_srv.get("command"))
results["lsp_enabled"] = bool(
    results["lsp_bridge_registered"]
    and _bridge_srv.get("enabled", True) is not False
    and _bridge_srv.get("disabled", False) is not True
)
results["mcp_configured"] = bool(
    mcp_entry(cfg, "github").get("url")
    or mcp_entry(cfg, "github").get("command")
)
# 3a. Websearch: enabled in config, and a provider key actually present.
# A provider name without a key is the common half-configured state — the tool
# is advertised to the model but every search fails at call time, so report
# the missing key rather than a bare True.
results.update(websearch_status(cfg, os.environ))
# 3a-2. ACP: opencode is the Agent Client Protocol server (`opencode acp`).
# Health = a real `initialize` handshake over stdio returns a result.
results["acp_ready"] = False
results["acp_protocol"] = "n/a"
try:
    _acp = subprocess.Popen(["opencode", "acp"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    try:
        _acp.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": 1,
                       "clientCapabilities": {"fs": {"readTextFile": True,
                                                     "writeTextFile": True},
                                              "terminal": True}}}) + "\n")
        _acp.stdin.flush()
        _deadline = time.time() + 25
        while time.time() < _deadline:
            _line = _acp.stdout.readline()
            if not _line:
                break
            if '"id"' not in _line:
                continue          # skip notifications/other traffic
            _msg = json.loads(_line)
            if _msg.get("id") == 1 and "result" in _msg:
                results["acp_ready"] = True
                results["acp_protocol"] = _msg["result"].get(
                    "protocolVersion", "unknown")
            break
    finally:
        _acp.terminate()
        try:
            _acp.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _acp.kill()
except (OSError, ValueError, json.JSONDecodeError):
    # No `opencode` on PATH or it spoke something unexpected: leave acp_ready
    # False rather than crashing the whole health check.
    pass
# 3b. LSP runtime readiness: node (for spawned server shims) + at least one server bin installed
try:
    semver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    semver = ""
results["node"] = semver or "n/a (LSP off)"
# Servers are seeded on PATH under ~/.local/bin (not ~/.cache/opencode/packages).
# NOTE: opencode v2 accepts `lsp` but does not run language servers, so this is
# reported readiness of the binaries only and stays outside the health gate.
_LSP_BINS = ["typescript-language-server", "bash-language-server", "pyright-langserver"]
_lsp_found = [b for b in _LSP_BINS
              if any(os.path.exists(os.path.join(d, b))
                     for d in os.environ.get("PATH", "").split(os.pathsep))]
results["lsp_servers_installed"] = _lsp_found or "none"
results["lsp_server_ready"] = bool(_lsp_found)
# 3c. Diagnostics toolchain — the real LSP replacement on v2 (binary has no
# LSP client: zero textDocument/* strings; see skills/diagnostics/SKILL.md).
_DIAG_BINS = ["tsc", "ruff", "shellcheck"]
results["diagnostics_tools"] = [b for b in _DIAG_BINS
                                if any(os.path.exists(os.path.join(d, b))
                                       for d in os.environ.get("PATH", "").split(os.pathsep))] or "none"
# 4. mojibake / corrupt scan (private-app index) — only if the clone exists locally
pd_path = f"{HOME}/gitwork/private-app-pwa/index.html"
if os.path.exists(pd_path):
    pd = open(pd_path, encoding="utf-8").read()
    results["mojibake_remaining"] = sum(1 for c in pd if 0xE0 <= ord(c) <= 0xEF)
else:
    results["mojibake_remaining"] = "n/a (private-app not cloned)"
# 5. version
for _bin in (f"{HOME}/.opencode/bin/opencode", "opencode"):
    try:
        results["version"] = subprocess.run([_bin, "--version"], capture_output=True, text=True).stdout.strip()
        if results["version"]: break
    except FileNotFoundError:
        continue
else:
    results["version"] = "n/a"
# 6. timers
timers = subprocess.run(["systemctl", "--user", "list-timers", "--all"],
                        capture_output=True, text=True).stdout
results["doctor_timer"] = "opencode-doctor.timer" in timers
results["logrotate_timer"] = "opencode-logrotate.timer" in timers
# 7. RUTH scheduled workflows green?
def gh_jobs(repo, wf_sha=None, n=6):
    r = curl_json(f"https://api.github.com/repos/{OWNER}/{repo}/actions/runs?per_page={n}",
                  [f"Authorization: Bearer {GH_KEY}"])
    if not r or "workflow_runs" not in r: return {}
    out = {}
    for run in r["workflow_runs"]:
        name = run["name"]
        if name in out: continue
        out[name] = run.get("conclusion")
    return out
fw = gh_jobs("private-repo")
results["ruth_maintenance"] = fw.get("Ruth Foundation Maintenance", "n/a")
results["ruth_verification"] = fw.get("Ruth Foundation Verification", "n/a")
# 8. secrets on private-app (bridge gating)
r = curl_json("https://api.github.com/repos/{OWNER}/private-app-pwa/actions/secrets",
              [f"Authorization: Bearer {GH_KEY}",
               "Accept: application/vnd.github+json",
               "X-GitHub-Api-Version: 2022-11-28"])
secrets = sorted(s["name"] for s in r.get("secrets", [])) if r and "secrets" in r else []
bridge_needed = ["SHOPIFY_STORE_DOMAIN", "SHOPIFY_ADMIN_TOKEN", "GEMINI_API_KEY"]
results["bridge_secrets"] = [s for s in bridge_needed if s not in secrets]

# 9. push memory to GitHub (canonical store). Keys are read from auth.json,
# never hardcoded, so nothing sensitive touches the repo.
# INVARIANT: doctor only owns AGENT_STATE.md. WORKLOG/ERROR_LOG are written by
# the agent/memory CLI/agent-cycle directly to GitHub; doctor must NEVER push a
# stale local mirror over them — it pulls them into the local mirror instead.
def pull_memory(local_path, repo_path):
    auth_h = [f"Authorization: Bearer {GH_KEY}", "User-Agent: opencode-agent", "Accept: application/vnd.github+json", "X-GitHub-Api-Version: 2022-11-28"]
    cmd = ["curl", "-s", "--max-time", "25", "-H", "Authorization: Bearer " + GH_KEY,
           "-H", "Accept: application/vnd.github.raw+json", "-H", "User-Agent: opencode-agent",
           f"https://api.github.com/repos/{MEMORY_REPO}/contents/{MEMORY_PATH}/{repo_path}"]
    body = subprocess.run(cmd, capture_output=True, text=True).stdout
    if body.strip() and not body.strip().startswith("{"):
        open(local_path, "w", encoding="utf-8").write(body)

def push_memory():
    def raw(url, method="GET", body=None, headers=None):
        cmd = ["curl", "-s", "--max-time", "25", "-X", method]
        for h in (headers or []): cmd += ["-H", h]
        if body: cmd += ["-d", json.dumps(body)]
        cmd += [url]
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    base = f"https://api.github.com/repos/{MEMORY_REPO}/contents/{MEMORY_PATH}"
    auth_h = [f"Authorization: Bearer {GH_KEY}", "User-Agent: opencode-agent", "Accept: application/vnd.github+json", "X-GitHub-Api-Version: 2022-11-28"]
    shas = {}
    idx = raw(base, headers=auth_h)
    try:
        for e in json.loads(idx):
            if isinstance(e, dict): shas[e["name"]] = e["sha"]
    except Exception:
        return False, []
    pushed, failed = [], []
    for name in MEMORY_FILES:
        p = f"{HOME}/.config/opencode/{name}"
        if not os.path.exists(p): continue
        content = open(p, encoding="utf-8").read()
        msg = {"message": f"sync {name} (opencode-doctor)", "branch": "main", "content": base64.b64encode(content.encode("utf-8")).decode("utf-8")}
        if name in shas: msg["sha"] = shas[name]
        r = raw(f"{base}/{name}", method="PUT", body=msg, headers=auth_h)
        did_push = '"content"' in r
        (pushed if did_push else failed).append(name)
    return (not failed), pushed

# 10. update ledger in place, with GitHub-first persistence (push memory first)
# First push memory to GitHub (canonical store), then read latest status
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for _name in PULL_FILES:  # doctor never overwrites the agent's ledgers
    pull_memory(f"{HOME}/.config/opencode/{_name}", _name)
m = push_memory()
results["memory_on_github"], results["memory_pushed"] = m

# health gate (includes persistence: GitHub store is the first place we look)
# NOTE: LSP + node are excluded from the gate — the box is small; LSP is
# enabled (typescript/bash/pyright seeded under ~/.cache/opencode/packages)
# and its readiness is reported, but a failed server must not block autonomy.
ok = all([
    results["openrouter_key"] or results.get("zen_door"),
    results["github_key"], results["mcp_initialize"],
    results["mcp_tools"] > 0, results["permission_allow"], results["mcp_configured"],
    results["doctor_timer"], results["logrotate_timer"],
    results["memory_on_github"],
])

# update ledger in place (GitHub is now current via push_memory)
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

ledger = f"""# opencode Identity — Soul & Memory Layer

The persistent identity and memory root for the opencode agent. The
canonical copy lives on GitHub and the local machine is the mirror. It
survives context resets and is consulted at the start of every session.

## Identity
- Name: Ruth (the agent you talk to in your terminal)
- Owner: {os.environ.get('RUTH_OWNER_DISPLAY', OWNER)} (GitHub: {OWNER})
- Machine: {results.get('machine_info', 'Unknown')}
- Home: {results.get('home_dir', f"{HOME}")}
- Config root: {results.get('config_root', f"{CFG_ROOT}")}
- Repos (cloned, shallow): gitwork/private-repo, gitwork/private-app-pwa; opencode fork NOT cloned (514 MB, disk rule)

## Memory architecture (GitHub first, computer second)
- GitHub (canonical): {OWNER}/opencode-identity -> memory/{'{IDENTITY,AGENT_STATE,WORKLOG,ERROR_LOG}'}.md
- Computer (mirror):    ~/.config/opencode/{'{IDENTITY,AGENT_STATE,WORKLOG,ERROR_LOG}'}.md
- Flow: opencode-doctor (hourly) pushes GitHub = source of truth.
  opencode-sync (pull GitHub -> local) runs at every session start.
  Local changes are always pushed; tables are append-only.

## Memory map (local mirror paths)
- AGENT_STATE.md  — live health ledger (keys, MCP, permissions, LSP, workflows, persistence) — auto-updated by opencode-doctor, pushed to GitHub
- IDENTITY.md     — this file
- WORKLOG.md      — detailed log of every change made (timestamp + commit SHA where relevant)
- ERROR_LOG.md    — accumulated app/agent error log (append-only)

## Rules
1. On every session start, FIRST: run opencode-sync (pull GitHub -> local),
   THEN read AGENT_STATE.md, then WORKLOG.md tail. GitHub is the first place,
   the computer is the second.
2. After every completed task: append to WORKLOG.md with timestamp + SHA.
3. On any error: append to ERROR_LOG.md with timestamp, command, root cause.
4. Never delete entries; the log IS the memory. Rotate only the huge
   opencode.log (systemd timer, daily 03:15), not these markdown memories.
5. Keep the doctor green (ALL_OK=True). If a check is red, fix it before
   proceeding — that is the standing order from the owner.

## Current Status
- GitHub persistence: {results["memory_on_github"]}
- Files pushed: {', '.join(results.get('memory_pushed', []))}
- OpenRouter key: {results["openrouter_key"]}
- Zen door (free brain): {results.get("zen_door", False)}
- GitHub key: {results["github_key"]}
- LSP enabled: {results["lsp_enabled"]}
- LSP server ready: {results["lsp_server_ready"]}
- GitHub MCP: {results["mcp_initialize"]} tools ({results["mcp_tools"]})
- Websearch: {results["websearch_enabled"]} provider={results["websearch_provider"]} key={results["websearch_key"]} ready={results["websearch_ready"]}
- ACP: {results["acp_ready"]} (protocol {results["acp_protocol"]})
- Permission allow: {results["permission_allow"]}
- MCP configured: {results["mcp_configured"]}
- Mojibake remaining: {results["mojibake_remaining"]}
- Node version: {results["node"]}
- Diagnostics tools (LSP replacement): {results["diagnostics_tools"]}
- Doctor timer: {results["doctor_timer"]}
- Logrotate timer: {results["logrotate_timer"]}
- the agent maintenance: {results.get('ruth_maintenance', 'unknown')}
- the agent verification: {results.get('ruth_verification', 'unknown')}

## Source Information
- Generated by opencode-doctor at {ts}
- GitHub API status: {results["memory_on_github"]}
- All checks: {len([k for k, v in results.items() if v])}/{len(results)}
- ALL_OK={"True" if ok else "False"}
"""

open(STATE, "w", encoding="utf-8").write(ledger)

# second pass: push the ledger to GitHub (now that GitHub is current via push_memory)
push_memory()

print(f"doctor: ALL_OK={ok}")
for k, v in results.items(): print(f"  {k}: {v}")
print(f"  ledger updated: {STATE}")
sys.exit(0 if ok else 1)


