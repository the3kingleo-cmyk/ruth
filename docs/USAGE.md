# Using Ruth

Install Ruth using the [README](../README.md#download) first. Commands below
run in a terminal with that Python environment activated. Use `ruth --help`
or `ruth teach --help` to see the installed CLI's options.

## Choose a mind folder

Each mind lives in its own directory. These examples consistently use
`./ruth-demo`, relative to the directory where you run the commands. For a
lasting setup, choose an absolute path outside the source checkout.

```sh
ruth --home ./ruth-demo status
```

On a fresh folder, `status` reports zero `moments_lived` and no learned
memories. A writing command such as `teach` saves the new mind. `--home` is a
global option: it comes before `status`, `teach`, `app` or any other subcommand.
It takes precedence over `RUTH_HOME`. Without either setting, Ruth uses the
[OS data directory](FAQ.md#where-is-my-mind-stored).

## Teach, continue and inspect

```sh
ruth --home ./ruth-demo teach "The garden gate is green."
ruth --home ./ruth-demo status
ruth --home ./ruth-demo say "The garden gate is " -n 24
ruth --home ./ruth-demo think "The garden gate is " -n 12
```

`teach` returns JSON with `bytes`, `private_bytes`, `cues` and elapsed
`seconds`. `status` shows that she has lived more moments. `say` generates a
continuation up to the requested length (or a newline or inhibition), and
`think` emits JSON describing candidate `thought`s, their `free_energy` and
which was `chosen`. A new mind has no general knowledge; do not expect a
particular sentence from one short lesson.

To teach from a file, create `lesson.txt` with public example text, such as
`The garden gate is green.` followed by a newline. Then run:

```sh
ruth --home ./ruth-demo teach lesson.txt
```

`teach` interprets an existing file path as a file; otherwise it treats its
argument as literal text. It also accepts standard input. For example, in a
POSIX shell:

```sh
printf 'The garden gate is green.\n' | ruth --home ./ruth-demo teach -
```

## Mark a confidence

These values are fictional demo data. Use `{{...}}` to mark the exact private
span within an otherwise public lesson:

```sh
ruth --home ./ruth-demo teach "The demo locker label is {{violet-314}}."
```

Alternatively, `--private` treats an entire message as a confidence:

```sh
ruth --home ./ruth-demo teach --private "The demo meeting is at the orchard."
```

The two modes differ: explicit braces mark a span; `--private` marks the
information she could not already predict, rather than every ordinary word
in its wrapping. The reported `private_bytes` therefore need not equal the
whole message length. Private spans are excluded from the public history, but
the brain state still learns from the experience. Treat the entire mind
directory as sensitive.

To ask whether she recognises private information in a file you created:

```sh
ruth --home ./ruth-demo check lesson.txt
```

`check` returns JSON under `recognised_private`. Its exit status is **1** if
it recognises private spans and **0** otherwise; file errors are errors, not a
clean scan. Learned recognition can also flag public text after a confidence;
inspect the report rather than treating the exit code as a definitive
classification. It is not encryption or a guarantee that a file is safe to
publish.

## Talk in the terminal or browser

```sh
ruth --home ./ruth-demo talk
```

At the `you>` prompt, enter a message or one of these commands:

| Input | Action |
|---|---|
| `/private The demo meeting is at the orchard.` | Share a confidence |
| `/good` or `/bad` | Give feedback on the last reply |
| `/sleep` | Consolidate and dream |

Ctrl-C ends the conversation and saves the mind. To use the browser instead:

```sh
ruth --home ./ruth-demo app --no-browser --port 7438
```

Open `http://127.0.0.1:7438` yourself. Omitting `--no-browser` asks Ruth to
open it automatically. Keep the default loopback host. Stop the app with
Ctrl-C before starting another writer, including `talk`, `teach` or `sleep`,
against this mind. Read-only commands such as `status` inspect the saved
snapshot, so they may lag behind the app's in-memory state.

## Sleep and read the journal

With the app and terminal conversation stopped:

```sh
ruth --home ./ruth-demo sleep --dreams 1
ruth --home ./ruth-demo dreams --last 1
ruth --home ./ruth-demo introspect
```

`sleep` consolidates memories and reports its phases as JSON. `dreams` reads
the saved journal; it does not start a new sleep. `introspect` emits the
operational graph as JSON without needing a `--json` flag. Sleep duration
depends on the accumulated history. See [architecture and measurements](ARCHITECTURE.md)
for what is measured and which experiments did not help.

## Use the Python brain directly

The low-level `Brain` API is useful for experiments without the conversation
layer. This example uses a temporary directory and does not touch an existing
mind:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from ruth import Brain, BrainConfig

with TemporaryDirectory() as directory:
    brain = Brain(BrainConfig(senses={"text": 9}, auto_sleep=False))
    result = brain.learn_bytes(b"The garden gate is green.\n")
    assert result["bytes"] == 26
    state = Path(directory) / "brain.npz"
    brain.save(str(state))
    restored = Brain.load(str(state))
    assert restored.steps == brain.steps
```

`Brain` handles continuous signals, learning and its array state. The `Mind`
layer adds identity, temperament, confidences and the public history. Saving
a `Brain` alone is not a backup of a complete `Mind`.

## Export to the optional C runtime

You need `make` and a C99 compiler. From the repository root, with the app
stopped, run these commands in a POSIX shell:

```sh
make -C ruth/csrc
ruth --home ./ruth-demo export ./ruth-demo/brain.bin
printf 'The garden gate is ' | ./ruth/csrc/ruth-core ./ruth-demo/brain.bin gen 24 --no-learn --no-save
```

The export carries the brain into the dependency-free C runtime, not the
Python app and full mind folder. The flags above keep generation from
learning or rewriting the export. C learning updates the `.bin` file; it does
not sync those changes back into the Python mind's `.npz` state.

For installation issues, backups and common errors, see the [FAQ](FAQ.md).
