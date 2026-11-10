# Contributing to Ruth

Thank you for helping Ruth grow. Four principles decide what belongs here:

1. **Token-free.** No tokenizer, vocabulary, embedding table of symbols, or
   pretrained model. Signals stay continuous; `tests/test_token_free.py`
   enforces this on every build.
2. **Self-contained.** Python standard library + numpy only, and no network
   access. The C runtime stays dependency-free C99.
3. **Security is her judgment.** Privacy comes from what she was told and
   her discretion, not from filters or keys.
4. **Measured, not assumed.** A change that claims to help comes with a
   measurement (`ruth bench`, a test, or a documented experiment), and null
   results are written down in `docs/ARCHITECTURE.md` too.

## Workflow

### Set up a checkout

Start from the repository's `main` branch and create a topic branch for one
change. Use Python 3.9 or newer and run the following from the repository root:

```sh
git switch main
git switch -c your-topic
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux, macOS or
WSL, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install
the editable package and test dependencies:

```sh
python -m pip install -e ".[dev]"
ruth --version
```

The runtime dependency is numpy. The `dev` extra adds pytest because
`tests/test_gemma.py` invokes it from within the unittest suite. Installing
dependencies may use the network; running the mind does not require it.

### Make and verify the change

- Keep each PR focused on its issue. For a functional bug, include a local
  reproduction and a regression test that fails before the fix and passes
  afterward. Describe documentation changes with runnable examples.
- If you change the brain's step equations in Python, mirror them in
  `ruth/csrc/ruth_core.c`. The parity tests require agreement to 1e-8.
- Keep a newborn Ruth a blank slate: no data, text or personality may ship in
  the package.
- Structural self-modification goes through `ruth/morphogenesis.py` and its
  meta-kernel. Never bypass the certificate.

Run a focused test while iterating, then the complete suite before opening
the PR:

```sh
python -m unittest tests.test_cli_evolve -v
python -m unittest discover -s tests -t . -v
git diff --check
```

The focused command above is an example for a CLI change; choose the module
that exercises your change. Do not use a focused run as a substitute for the
full suite. A C99 compiler (`cc` or `gcc`) enables `tests/test_c_runtime.py`;
those tests compile into a temporary directory. To build the optional CLI
runtime yourself, use `make -C ruth/csrc`.

The mind bridge tests look for the executable at `RUTH_BIN`, falling back to
`~/.local/bin/ruth`. To exercise those tests against your virtual environment
on Linux, macOS or WSL, run:

```sh
RUTH_BIN="$(command -v ruth)" python -m unittest tests.test_mind_mcp -v
```

Those integration tests use a temporary mind folder. Tests that require
machine-specific optional configuration may skip when it is absent. Report
skips and failures with their reasons; do not present an unrun check as a pass.
The [CI workflow](.github/workflows/ci.yml) runs unittest on Linux, macOS and
Windows with Python 3.9, 3.12 and 3.13. A local run only verifies the platform
and interpreter on which it ran.

### Prepare the pull request

Explain the problem, the resulting behavior and how to reproduce it. Link the
issue, list the commands you actually ran and their results, and call out any
remaining limitations. Include measurements for performance claims and record
null experiments in [the architecture notes](docs/ARCHITECTURE.md).

Before committing, review `git diff` and `git status --short`. Keep generated
brain state, history, dream journals, model exports, personal profiles and
credentials out of the PR. Use an explicit disposable `--home` folder outside
the checkout for experiments; stop any app using it before CLI writes. The
[usage guide](docs/USAGE.md) and [FAQ](docs/FAQ.md) explain state ownership and
backup. Documentation should describe the current CLI, use fictional examples
and link to the relevant existing design notes.

## Releases

For maintainers preparing a release: bump `__version__` in `ruth/__init__.py`,
then push a tag `vX.Y.Z`. The
[release workflow](.github/workflows/release.yml) tests, builds and publishes
the downloads. An ordinary documentation or bug-fix PR does not need a release
tag.
