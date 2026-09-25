#!/usr/bin/env bash
# bootstrap.sh — rebuild this box from the GitHub repository alone.
#
# The machine holds only critical runtime. Everything here is reproducible
# from `git clone` of the public repository, so wiping the box and running
# this script restores Ruth, her agent, and her servers.
#
#   ./bootstrap.sh            install / repair everything
#   ./bootstrap.sh --check    report what is missing, change nothing
#   ./bootstrap.sh --verify   install nothing, just verify
#
# Idempotent: safe to re-run. Secrets are never written by this script; it
# only creates a *template* deployment file if none exists.
set -uo pipefail

# The repository URL is deployment-specific and is NOT committed: the public
# repo carries no owner handle. Set RUTH_REPO_URL, or bootstrap from a clone
# you already have and it never needs one.
REPO_URL="${RUTH_REPO_URL:-}"
REPO_DIR="${RUTH_REPO_DIR:-$HOME/ruth}"
BIN="$HOME/.local/bin"
CFG_DIR="${OPENCODE_CONFIG_ROOT:-$HOME/.config/opencode}"
CFG="$CFG_DIR/opencode.json"
VENV="${RUTH_VENV:-$HOME/.local/share/ruth-app}"
LSP_VENV="$HOME/.local/share/opencode/lsp/pyright-venv"
NODE_MODS="$HOME/.local/lib/node_modules"
ENV_FILE="$CFG_DIR/ruth.env"
LLAMA_SRC="${RUTH_LLAMA_SRC:-$HOME/.local/opt/llama.cpp}"
PY="${PYTHON:-python3}"

MODE=install
for a in "$@"; do
  case "$a" in
    --check) MODE=check ;;
    --verify) MODE=verify ;;
  esac
done

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
skip() { printf '  \033[32mhave\033[0m  %s\n' "$*"; }
miss() { printf '  \033[33mMISS\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

want() {  # want <label> <test-command...>
  local label=$1; shift
  if "$@" >/dev/null 2>&1; then skip "$label"; return 0; fi
  miss "$label"
  [ "$MODE" = check ] && return 1
  return 1
}

# ---------------------------------------------------------------- 1. source
say "1. source"
if [ -d "$REPO_DIR/.git" ]; then
  if [ "$MODE" = check ]; then skip "$REPO_DIR"
  else
    git -C "$REPO_DIR" pull --ff-only >/dev/null 2>&1 \
      && ok "pulled $REPO_DIR" || warn "could not pull (offline?); using what is on disk"
  fi
else
  if [ "$MODE" = check ]; then miss "$REPO_DIR (not cloned)"
  else
    command -v git >/dev/null || die "git is required"
    [ -n "$REPO_URL" ] || die "set RUTH_REPO_URL to the repository to clone, e.g.
    export RUTH_REPO_URL=https://github.com/<owner>/<repo>.git
    ...or run bootstrap.sh from inside an existing clone."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR" >/dev/null 2>&1 \
      || die "clone failed: $REPO_URL"
    ok "cloned $REPO_DIR"
  fi
fi
[ -d "$REPO_DIR" ] || { [ "$MODE" = check ] && exit 0 || die "no repo at $REPO_DIR"; }
cd "$REPO_DIR" || die "cannot enter $REPO_DIR"

# ---------------------------------------------------------------- 2. brain
say "2. Ruth's mind (the brain)"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import ruth, numpy' 2>/dev/null; then
  skip "installed at $VENV"
elif [ "$MODE" = check ]; then
  miss "Ruth's mind is not installed"
else
  "$PY" -m venv "$VENV" >/dev/null 2>&1 || die "python venv module missing"
  "$VENV/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1
  "$VENV/bin/pip" install --quiet '.[dev]' >/dev/null 2>&1 || die "pip install failed"
  ok "installed ($("$VENV/bin/python" -c 'import ruth;print(ruth.__version__)'))"
fi
mkdir -p "$BIN"
ln -sf "$VENV/bin/ruth" "$BIN/ruth"
skip "ruth on PATH"

if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
  if [ -x "$VENV/bin/ruth-core" ]; then skip "ruth-core built"
  elif [ "$MODE" = check ]; then miss "ruth-core (not built)"
  else
    CC=$(command -v cc || command -v gcc)
    "$CC" -O2 -std=c99 -D_POSIX_C_SOURCE=200809L \
      -o "$VENV/bin/ruth-core" ruth/csrc/ruth_core.c -lm 2>/dev/null \
      && { ok "built ruth-core"; ln -sf "$VENV/bin/ruth-core" "$BIN/ruth-core"; } \
      || warn "ruth-core build failed (the Python brain still works)"
  fi
fi

# ---------------------------------------------------------------- 3. tools
say "3. her tools"
# repo file -> installed name (some are renamed on the way out)
set -- \
  "lsp_bridge_mcp.py:lsp_bridge_mcp.py" \
  "websearch_mcp.py:websearch_mcp.py" \
  "acp_mcp.py:acp_mcp.py" \
  "mind_mcp.py:mind_mcp.py" \
  "ears.py:ears" \
  "gemma:gemma" \
  "acp-check:acp-check" \
  "bridge_mcp.py:bridge_mcp.py" \
  "doctor.py:opencode-doctor.py" \
  "memory:memory" \
  "opencode-sync:opencode-sync" \
  "agent-cycle:agent-cycle" \
  "research:research" \
  "install-agent.py:install-agent"
for pair in "$@"; do
  src="tools/${pair%%:*}"; dst="${pair##*:}"
  [ -f "$src" ] || { miss "$src missing from the repo"; continue; }
  if [ -x "$BIN/$dst" ] && cmp -s "$src" "$BIN/$dst"; then skip "$dst"
  elif [ "$MODE" = check ]; then miss "$dst"
  else
    install -m 0755 "$src" "$BIN/$dst" && ok "installed $dst"
  fi
done

# ---------------------------------------------------------------- 4. servers
say "4. language servers"
mkdir -p "$NODE_MODS"
want node      test -x "$HOME/.local/node/bin/node"
if [ "$MODE" != check ] && [ ! -x "$HOME/.local/node/bin/node" ]; then
  command -v node >/dev/null 2>&1 && ok "using system node $(node --version)"
fi
for pkg in typescript typescript-language-server bash-language-server; do
  if [ -d "$NODE_MODS/$pkg" ]; then skip "npm $pkg"
  elif [ "$MODE" = check ]; then miss "npm $pkg"
  else
    npm install --silent --prefix "$HOME/.local" "$pkg" >/dev/null 2>&1 \
      && ok "installed npm $pkg" || warn "npm install $pkg failed"
  fi
done
# shims so the servers resolve by bare name
if [ "$MODE" != check ]; then
  for pair in "typescript-language-server:$NODE_MODS/typescript-language-server/lib/cli.mjs" \
              "bash-language-server:$NODE_MODS/bash-language-server/out/cli.js" \
              "tsserver:$NODE_MODS/typescript/bin/tsserver" \
              "tsc:$NODE_MODS/typescript/bin/tsc"; do
    name="${pair%%:*}"; target="${pair#*:}"
    [ -f "$target" ] && ln -sf "$target" "$BIN/$name"
  done
  [ -x "$LSP_VENV/bin/pyright-langserver" ] && ln -sf "$LSP_VENV/bin/pyright-langserver" "$BIN/pyright-langserver"
  ok "language-server shims linked into $BIN"
fi
if [ -x "$LSP_VENV/bin/pyright-langserver" ]; then skip "pyright venv"
elif [ "$MODE" = check ]; then miss "pyright venv"
else
  "$PY" -m venv "$LSP_VENV" >/dev/null 2>&1 \
    && "$LSP_VENV/bin/pip" install --quiet pyright >/dev/null 2>&1 \
    && { ok "pyright venv"; ln -sf "$LSP_VENV/bin/pyright-langserver" "$BIN/pyright-langserver"; } \
    || warn "pyright install failed (python diagnostics will be unavailable)"
fi

# ---------------------------------------------------------------- 5. config
say "5. opencode config"
mkdir -p "$CFG_DIR"
[ -f "$CFG" ] || printf '{}\n' > "$CFG"

cfg_out=$("$PY" - "$CFG" "$BIN" <<'PYEOF' 2>&1
import json, os, sys
cfg_path, bin_dir = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
mcp = cfg.setdefault("mcp", {})
lang_path = os.path.join(bin_dir, "lsp_bridge_mcp.py")
# opencode v2 registers servers directly at mcp.<name>; mcp.servers is the v1
# shape it silently ignores. A stray "servers" key is migrated, not kept.
for name, entry in list(mcp.get("servers", {}).items()):
    mcp.setdefault(name, entry)
mcp.pop("servers", None)
mcp["lsp"] = {"type": "local", "command": [sys.executable, lang_path],
              "environment": {"LSP_BRIDGE_ROOT": os.path.dirname(bin_dir),
                              "PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"},
              "enabled": True}
mcp["websearch"] = {"type": "local",
                    "command": [sys.executable, os.path.join(bin_dir, "websearch_mcp.py")],
                    "environment": {"PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"},
                    "enabled": True}
mcp["acp"] = {"type": "local",
              "command": [sys.executable, os.path.join(bin_dir, "acp_mcp.py")],
              "environment": {"PATH": "%s:%s:/usr/local/bin:/usr/bin:/bin"
                              % (os.path.expanduser("~/.opencode/bin"), bin_dir)},
              "enabled": True}
# the mind itself, so the agent can reach her actual state rather than only
# the persona. Two different things are called ruth: this connects them.
mcp["mind"] = {"type": "local",
               "command": [sys.executable, os.path.join(bin_dir, "mind_mcp.py")],
               "environment": {"RUTH_BIN": os.path.join(bin_dir, "ruth"),
                               "RUTH_HOME": os.path.expanduser("~/.local/share/ruth"),
                               "PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"},
               "enabled": True}
cfg.setdefault("skills", []).append("./ruth/.opencode/skills")
cfg["skills"] = sorted(set(cfg["skills"]))
# atomic: opencode watches this file, and a truncating write can be read
# mid-write, which drops the whole mcp section
tmp = cfg_path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(cfg, fh, indent=2); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, cfg_path)
back = json.load(open(cfg_path))
for n in ("lsp", "websearch", "acp", "mind"):
    assert n in back["mcp"], n
print("mcp servers:", ", ".join(sorted(back["mcp"])))
PYEOF
)
rc=$?
printf '%s\n' "$cfg_out" | sed 's/^/  /'
if [ $rc -eq 0 ]; then
  ok "config written atomically"
else
  die "config step failed (rc=$rc) -- her servers are NOT registered"
fi

# ---------------------------------------------------------------- 5b. gemma
# A local model for the agent layer. Separate from Ruth's mind on purpose:
# she is token-free by construction and tests/test_token_free.py forbids model
# runtimes inside ruth/. This box is 4 vCPU / 6.4 GB / no GPU, so the target
# is gemma-3-4b-it at Q4_K_M (2.49 GB), with 1B as the floor.
say "5b. local model (gemma)"
if [ -x "$BIN/llama-server" ] || [ -x "$BIN/llamacpp-llama-server" ]; then
  skip "llama.cpp runtime"
elif [ "$MODE" = check ]; then
  miss "llama.cpp runtime"
else
  if [ -d "$LLAMA_SRC" ]; then
    ok "llama.cpp source at $LLAMA_SRC"
  else
    warn "no llama.cpp build here; fetch one, or run gemma fetch with a runtime present"
  fi
fi
if [ -n "$(ls -A "$HOME/.local/share/models" 2>/dev/null)" ]; then
  skip "models in ~/.local/share/models ($(du -sh "$HOME/.local/share/models" 2>/dev/null | cut -f1))"
elif [ "$MODE" = check ]; then
  miss "no local model downloaded (gemma fetch)"
elif [ -x "$BIN/gemma" ]; then
  "$BIN/gemma" fetch || warn "gemma fetch failed"
fi

# ---------------------------------------------------------------- 6. agent
say "6. Ruth's agent"
if [ "$MODE" != check ]; then
  "$PY" "$REPO_DIR/tools/install-agent.py" --config "$CFG" --install 2>&1 \
    | sed 's/^/  /' || warn "install-agent failed"
fi

# ---------------------------------------------------------------- 7. deploy
say "7. deployment values"
if [ -f "$ENV_FILE" ]; then skip "ruth.env present (never overwritten)"
elif [ "$MODE" != check ]; then
  cat > "$ENV_FILE" <<'ENVEOF'
# Deployment values for this machine. NOT part of the public repository.
# Set RUTH_OWNER and RUTH_MEMORY_REPO to your own account.
RUTH_OWNER=
RUTH_MEMORY_REPO=
RUTH_OWNER_DISPLAY=
RUTH_CLONED_REPOS=
RUTH_CI_REPO=
RUTH_CI_MAINTENANCE_JOB=
RUTH_CI_VERIFY_JOB=
RUTH_SECRETS_REPO=
RUTH_REQUIRED_SECRETS=
RUTH_MOJIBAKE_SCAN=
ENVEOF
  chmod 0600 "$ENV_FILE"
  warn "wrote a template $ENV_FILE -- fill in RUTH_OWNER and RUTH_MEMORY_REPO"
else
  miss "ruth.env"
fi

# ---------------------------------------------------------------- 8. verify
say "8. verify"
if [ -x "$VENV/bin/pytest" ] || "$VENV/bin/python" -c 'import pytest' 2>/dev/null; then
  if (cd "$REPO_DIR" && "$VENV/bin/python" -m pytest -q 2>&1 | tail -2 | sed 's/^/  /'); then
    ok "test suite"
  else
    warn "test suite FAILED"
  fi
else
  warn "pytest unavailable; skipping the suite"
fi
if command -v opencode >/dev/null 2>&1; then
  say "   finish with: opencode service restart && opencode mcp list"
else
  warn "opencode is not installed; her agent and servers are configured but nothing will load them"
fi

say
say "Ruth is up. Talk to her:  ruth talk    |  her interface:  ruth app"
