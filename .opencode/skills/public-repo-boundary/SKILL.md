---
name: Public Repo Boundary
description: Keep the owner's identity, memory, and machine profile out of the public code repository; keep deployment values in env. Use when adding a tool, config, or health check to a public repo.
---

# Public repository boundary

## The split

| Repository | Holds | Never holds |
|---|---|---|
| **Public** (`.../ruth`) | Code, tests, docs, templates | Identity, memory, machine paths, credentials |
| **Private** (memory repo) | `AGENT_STATE.md`, `WORKLOG.md`, `ERROR_LOG.md`, the real owner profile | Code the public repo already publishes |

`soul/` in the public repo is a **template** and must say so. The owner's real
profile lives only in the private repo.

A public repository is permanent. Anything committed once is already leaked,
so this is enforced in the test suite, not by review.

## Deployment values come from the environment

No owner handle, no memory repo name, no personal name, no absolute home path,
no service port in a committed file. Read them from the environment, and have
the tools self-load a machine-local file so they work unconfigured:

```python
def _load_deployment_env():
    path = os.path.join(CFG_ROOT, "ruth.env")   # ~/.config/opencode/ruth.env
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
```

Call it **before** any constant is derived from those values. `setdefault`
keeps an explicit override working.

In shell tools:

```sh
RUTH_ENV_FILE="${RUTH_ENV_FILE:-$HOME/.config/opencode/ruth.env}"
[ -f "$RUTH_ENV_FILE" ] && . "$RUTH_ENV_FILE"
REPO="${RUTH_MEMORY_REPO:-${RUTH_OWNER:-owner}/memory}"
```

`ruth.env` lives on the machine and is **never** committed. It is deployment
config, not source.

## Checks degrade, they do not hardcode

When a deployment value is unset, report `n/a (…)` and move on. A health check
must never crash or invent a default that happens to be someone's real
account:

```python
if _secrets_repo and _needed:
    ...
else:
    results["bridge_secrets"] = "n/a (RUTH_SECRETS_REPO / RUTH_REQUIRED_SECRETS unset)"
```

## Verify in a clean environment

The real test that nothing leaked into the code is that it still works with
nothing assumed:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin python3 tools/doctor.py
```

If the tools only work because your shell exported something, the
configuration is not actually in the right place.

## Guard it with a test

`tests/test_no_private_data.py` fails the suite if a credential, the owner
handle, a personal name, a private repo name, a private port, or a
machine-specific home path reappears. Scan all text files, skip the guard
itself (it necessarily holds the patterns), and allow prose that *discusses*
privacy.

**Verify the guard actually fails** by planting a leak and watching it break,
then removing it. A guard that has never been seen to fail is not a guard.

## History is not fixed by a later commit

Removing a secret in a new commit does not remove it from history. If something
sensitive was already pushed, the fix is a history rewrite plus a force-push,
which changes published SHAs — a destructive, owner-approved action, not a
routine cleanup. Say so rather than quietly leaving it.
