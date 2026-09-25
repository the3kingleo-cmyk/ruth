---
name: Credential Hygiene
description: Never print, log, commit, or echo secrets; handle a credential pasted into chat. Use before writing any config, running any health check that reports keys, or when a secret may have been exposed.
---

# Credential hygiene

## The rule

Secrets are **referenced by path**, never by value. They live on the machine
in mode-0600 files and never enter a repository, a log, a status line, or a
conversation.

## Reporting the presence of a key, never the key

A health check should report a boolean or a set of *names*:

```
openrouter_key: True        github_key: True        ok
```

Never the value, never a prefix, never a length. When a check needs to prove a
key exists, assert on its **absence** in output and let the reader trust the
boolean:

```python
r = status(env)                       # env holds the real key
assert secret not in json.dumps(r)    # value must not appear in the result
```

## Never embed a real token while testing

Use an obvious synthetic fixture, and keep the exemption auditable — an
explicit allowlist entry, not a blanket "skip the test directory":

```python
CREDENTIAL_FIXTURES = {"tests/test_doctor.py": [SYNTHETIC_KEY]}
```

Use a value the scanner's own pattern cannot match, or add it to the
allowlist — do not paste a real-looking key into documentation either.

A scanner that flags its own patterns is a scanner with a false positive; a
scanner with a broad ignore is a scanner with a hole.

## If a secret is pasted into chat

Treat it as **compromised and rotate it**, because transcripts persist. Then:

1. Do **not** write it anywhere to "make use of it". A rotated key is the fix.
2. Say plainly that it is exposed and needs rotation.
3. Verify the real credential on disk never entered version control:

   ```bash
   TOK=$(cat ~/.github_token)
   git log -p | grep -qF "$TOK" && echo "LEAKED" || echo "clean"
   ```

4. Scrub it from any log or status file that captured it, and note the rotation
   with a timestamp in the worklog (append-only, never rewritten).

## Git hygiene

- Never put a token in a remote URL. `https://x-access-token:TOKEN@github.com/...`
  persists in `.git/config`; add the remote clean, then scrub.
- After any commit that touched config, scan the diff for credential shapes
  before pushing.
- Remote URLs in shared output should be printed with the userinfo redacted.

## What "redacted" means in output

Never print the full value, a long prefix, or a reversible fragment. Either
omit it or show at most a couple of non-sensitive leading characters alongside
an explicit marker, and prefer total omission.
