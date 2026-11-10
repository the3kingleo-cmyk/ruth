# Frequently asked questions

## What do I need to run Ruth?

Python 3.9 or newer and numpy 1.22 or newer. The Python package installs the
`ruth` command. A C99 compiler is needed only for the optional C runtime and
its parity tests. Contributors should also install the `dev` extra; see
[CONTRIBUTING.md](../CONTRIBUTING.md).

The mind does not need a model download, account or API key. Installing Python
dependencies may need internet access; normal mind operation is offline.

## Is the mind the same as the optional agent setup?

No. `ruth/` is the numpy-and-standard-library mind and local app. The repository
also contains optional agent definitions and host tools under `agent/` and
`tools/`. Their external integrations and local-model setup are separate from
the Python mind; you do not need them to run the
[quick start](../README.md#quick-start).

## Why does a new Ruth give short or unfamiliar answers?

A newborn has no pretrained knowledge or lived memories. Teach a small public
lesson first and inspect `ruth status` to confirm that learning happened.
`say` continues a prompt, whereas `talk` is the interactive conversation
interface. Exact output depends on her history. The
[honest status](../README.md#honest-status) and
[architecture notes](ARCHITECTURE.md) describe the current limitations.

## Where is my mind stored?

The selection order is:

1. The global CLI option `ruth --home PATH ...`.
2. The `RUTH_HOME` environment variable.
3. The operating system's data directory:

| System | Default directory |
|---|---|
| Linux and other Unix systems | `$XDG_DATA_HOME/ruth`, or `~/.local/share/ruth` if unset |
| macOS | `~/Library/Application Support/Ruth` |
| Windows | `%APPDATA%\Ruth` (home-directory fallback if `APPDATA` is absent) |

`ruth --home PATH status` includes the selected `home` in its JSON output.
Relative paths are relative to the current working directory, so use an
absolute path when switching between terminals or checkouts.

## How do I back up, move or start a separate mind?

Stop the app and any terminal conversation first so they save and release
ownership. Copy the **entire mind directory**, including `brain.npz`,
`mind.json`, public history, dream journal and any evolved configuration.
Restore it to a separate folder and use `ruth --home PATH status` to inspect
the copy. The Python version and dependencies must still be installed on the
destination computer.

To experiment with a blank slate, point `--home` at a new, empty folder. Keep
your original as a backup. A C `brain.bin` export or a standalone `.npz` file
does not contain the complete Python mind state.

## Does marking something private encrypt the files?

No. `{{...}}` and `--private` teach learned discretion; they do not provide
file encryption or an access-control boundary. Private spans are omitted from
the public history, but the learned brain state still contains experience.
Protect the whole mind folder and its backups using your operating system's
permissions and storage protections. Use fictional data when sharing examples
or bug reports. See [confidence examples](USAGE.md#mark-a-confidence).

## Why does a command say another process owns the mind?

The app holds exclusive ownership while it runs. Another writer could
overwrite its live experience, so commands such as `teach`, `talk` and `sleep`
refuse to start against that folder. Stop the owner with Ctrl-C and retry, or
use a separate `--home` for an independent experiment.

Read-only commands such as `status`, `introspect` and `check` inspect the
saved state. They may not reflect the app's most recent unsaved moments. The
lock error mentions `--read-only`, but the current CLI has no such global
flag; choose one of those read-only commands instead. Do not delete an active
owner's lock file to force a second writer.

## Why is `ruth` not found, or numpy missing?

Activate the same virtual environment where you installed the package, then
run `python -m pip show ruth-mind numpy`. If either is missing, run
`python -m pip install .` from the repository root in that environment.
`python -m ruth.cli --help` is an alternative way to invoke the installed
CLI when diagnosing command lookup. If `python` is unavailable but `python3`
exists, use that interpreter consistently when creating the environment.

## What if native Windows reports a missing `fcntl` module?

The current `ruth/owner.py` uses the POSIX `fcntl` module for ownership locks,
which is unavailable in native Windows Python. Use WSL with Linux Python for
this checkout if you hit that error. The CI matrix includes Windows, but its
presence alone does not prove that every current command works there. Include
the interpreter, OS and traceback when reporting the problem.

## Why did `--home` or `--json` fail?

Place `--home` before the subcommand, for example
`ruth --home ./ruth-demo status`. `status`, `introspect`, `think`, `sleep`
and `dreams` already emit JSON; they do not take a `--json` option. Check
`ruth COMMAND --help` for the accepted flags.

## How do I open the app without launching a browser?

Run `ruth --home ./ruth-demo app --no-browser` and open
`http://127.0.0.1:7438` manually. If another local service uses that port, pass
`--port 7439` and open the corresponding URL. Keep the default loopback host;
the app is intended for local use.

## Why is the dream journal empty, or sleep taking time?

`dreams` reads journal entries that already exist. With other writers stopped,
run `ruth --home ./ruth-demo sleep --dreams 1` to create an entry. Consolidation,
rehearsal and growth checks can take longer as history grows. If the interface
stops responding while she sleeps, capture the command, history size and
observed duration for a bug report; do not start a second writer on that mind.

## What should a useful bug report include?

Include `ruth --version`, the Python version, operating system, exact command,
expected behavior, observed output and a minimal example using a disposable
mind. Remove personal paths and use fictional lesson text. Do not attach
your real brain state, history, journal or credentials. For a contribution,
include the tests you actually ran and any failures or skips; see the
[contribution workflow](../CONTRIBUTING.md).
