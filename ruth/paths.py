"""Where Ruth keeps her mind on whatever machine she wakes up on.

Nothing is tied to a user name, host name or terminal. Resolution order:
1. ``$RUTH_HOME`` if set;
2. Windows: ``%APPDATA%\\Ruth``; macOS: ``~/Library/Application Support/Ruth``;
3. otherwise ``$XDG_DATA_HOME/ruth`` (default ``~/.local/share/ruth``).
"""
from __future__ import annotations

import os
import sys


def home() -> str:
    env = os.environ.get("RUTH_HOME")
    if env:
        base = env
    elif sys.platform.startswith("win"):
        base = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Ruth")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/Ruth")
    else:
        xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        base = os.path.join(xdg, "ruth")
    os.makedirs(base, exist_ok=True)
    return base


def path(*parts: str) -> str:
    p = os.path.join(home(), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def brain_state() -> str:
    return os.environ.get("RUTH_BRAIN") or path("brain.npz")
