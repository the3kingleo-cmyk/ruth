#!/bin/sh
# Install Ruth on Linux, macOS, Linux, WSL or BSD.
# Usage:  sh install.sh          (from an unpacked download or a git clone)
# Installs into its own virtual environment; nothing system-wide is changed.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "Ruth needs Python 3.9+ (https://www.python.org/downloads/)"; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' || { echo "Ruth needs Python 3.9+"; exit 1; }

dest=${RUTH_INSTALL_DIR:-"$HOME/.local/share/ruth-app"}
echo "Installing Ruth into $dest"
"$PY" -m venv "$dest" 2>/dev/null || {
  echo "Python's venv module is missing. On Debian/Ubuntu/Crostini: sudo apt install python3-venv"; exit 1; }
"$dest/bin/python" -m pip install --quiet --upgrade pip
"$dest/bin/python" -m pip install --quiet "$here"

bin="$HOME/.local/bin"
mkdir -p "$bin"
ln -sf "$dest/bin/ruth" "$bin/ruth"

# optional: the dependency-free C brain, if a compiler is present
if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
  cc_=$(command -v cc || command -v gcc)
  "$cc_" -O2 -std=c99 -D_POSIX_C_SOURCE=200809L -o "$dest/bin/ruth-core" "$here/ruth/csrc/ruth_core.c" -lm \
    && ln -sf "$dest/bin/ruth-core" "$bin/ruth-core" && echo "Built ruth-core (C runtime)."
fi

case ":$PATH:" in *":$bin:"*) ;; *) echo "Add $bin to your PATH to run 'ruth' from any terminal." ;; esac
echo "Done. Start her with:  ruth app     (or: ruth talk)"
