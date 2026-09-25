---
name: Tool Surface
description: Audit what an agent can actually call — resolve real permissions across config layers, and separate "permitted" from "works". Use when a tool appears available but fails, or before enabling/disabling a tool.
---

# The tool surface

A tool being in the catalog does not mean the agent may call it, and being
permitted does not mean it works. Those are three separate facts and a health
check that conflates them will report green over a broken tool.

## Layered permissions, last match wins

OpenCode evaluates permission rules **in order, last match wins**, across:

1. a top-level `permissions` array
2. the top-level `permission` map (`"read": "allow"`, or a per-resource map)
3. `agents.<id>.permissions` — the agent's own rules, applied **last**

So an agent can carry `{"action":"*","effect":"allow"}` and still be denied a
single tool by a later rule. Reading only the top-level `permission` map gets
this wrong and reports tools as available that the agent cannot call.

```python
def resolve(action):
    effect = None
    for r in rules:                      # last match wins
        if r.get("action") in (action, "*") and r.get("resource", "*") in ("*", action):
            effect = r.get("effect", "allow")
    return effect or "unset"
```

## Per-resource entries are protections, not effects

`{"external_directory": {"*": "allow", "~/.github_token": "deny"}}` means the
action is allowed **except** for that path. A resolver that only tracks the
action-level effect will report `allow` and quietly hide the deny that is
protecting a credential file. Surface the specific entries too:

```python
out["_restricted"] = [f"{a}:{res}={eff}" for rules where res != "*" and eff in ("deny","ask")]
```

## Permitted ≠ working

The built-in `websearch` tool on this box is permitted and **permitted-and-
broken**: every provider (Exa, Firecrawl, Parallel, Tavily) needs its own API
key, and with none set every call fails at invocation time. A `permission:
allow` check passes while the tool is useless.

Report both, separately:

```
Tool websearch (builtin): permitted=allow works=False | mcp=True
```

`mcp=True` means a keyless local server covers the capability. That is the
difference between "search is unavailable" and "search works by another route".

## Denied tools are a design decision, not a bug

`question: deny` means the agent can never surface a choice to the user;
`subagent: deny` means no child sessions. Both are legitimate for an
autonomous agent, but both should be **visible** in the health ledger so the
posture is a recorded choice rather than an accident. `external_directory:
ask` on the other hand *does* prompt, so "zero prompts" is only true if no
prompting rule is left.

## Auditing

- Enumerate the live catalog, not the docs.
- Resolve each action through all permission layers.
- Probe each permitted tool for a real result, not a config key.
- Put permitted-but-broken in the ledger as its own line.
- Remember `question: deny` also means the agent cannot ask for a decision —
  surface the trade-off in the report instead, as a recommendation.
