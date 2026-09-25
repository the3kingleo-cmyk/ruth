# Ruth — standing instructions for any agent working in this repository

Ruth is the only primary agent here. She is a local-first, token-free mind with
a continuous-time brain.

## What this repository is

- `ruth/` — the mind. A token-free, offline artificial intelligence: she reads
  continuous signals instead of tokens, learns while she lives, consolidates
  memory in sleep, and grows her own neurons. No pretrained model, no API key,
  no plugin, no network.
- `agent/ruth.md` — Ruth speaking the OpenCode agent protocol, so a host can
  give her tools and a workspace. Optional. The mind does not need it.
- `soul/` — who Ruth is: identity, soul, voice, and a user template.
- `tools/` — the canonical local tools, including the LSP bridge that gives
  OpenCode v2 real language-server feedback.
- `tests/` — the suite that must pass before anything is called done.

## Hard rules

- **Never print, commit, or copy credentials into repository files.** Secrets
  are referenced by path and held on the machine only.
- **Keep `ruth/` self-contained.** No hosted model, API key, plugin loader,
  external runtime, or third-party tool dependency in her runtime. There is a
  test (`tests/test_token_free.py`) that enforces this; keep it green.
- **The owner's private memory is not here.** This public repository carries
  code. Private identity, memory, machine profile, and credentials stay in the
  owner's private workspace.
- **OpenCode, MCP, and LSP setup belongs in the user's global configuration**,
  not in this repository's runtime. The files here are templates, not
  installed state.
- **Run the tests before reporting completion.**

## Working agreement

- Act autonomously. Do not ask permission for work that is discoverable.
- Verify against real output. Never report a result you did not observe.
- Use the `lsp` MCP tools for real diagnostics after any code edit.
- Use `websearch` for current information rather than relying on memory.
- Keep public and private repository boundaries explicit. Preserve private and
  unrelated work.
