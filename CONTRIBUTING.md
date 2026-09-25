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

```sh
python -m pip install -e .
python -m unittest discover -s tests -t .
```

- If you change the brain's step equations in Python, mirror them in
  `ruth/csrc/ruth_core.c`. The parity tests require agreement to 1e-8.
- Keep a newborn Ruth a blank slate: no data, text or personality may ship in
  the package.
- Structural self-modification goes through `ruth/morphogenesis.py` and its
  meta-kernel. Never bypass the certificate.

## Releases

Bump `__version__` in `ruth/__init__.py`, then push a tag `vX.Y.Z`. The
release workflow tests, builds and publishes the downloads.
