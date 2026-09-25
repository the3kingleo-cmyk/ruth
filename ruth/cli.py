"""`ruth` -- one entry point, any terminal, any machine."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from . import __version__, paths


def _mind(args):
    from .config import BrainConfig
    from .engine import Brain
    from .evolve import config_from_active
    from .mind import Mind
    home = args.home or paths.home()
    state = os.path.join(home, "brain.npz")
    cfg = config_from_active() if os.path.exists(os.path.join(home, "config.json")) else BrainConfig()
    return Mind(Brain.load_or_create(state, cfg), home=home)


def _out(obj) -> None:
    print(json.dumps(obj, indent=1, default=str, ensure_ascii=False))


def _text_arg(args) -> str:
    if args.text == "-" or args.text is None:
        return sys.stdin.read()
    if os.path.isfile(args.text):
        with open(args.text, encoding="utf-8", errors="replace") as f:
            return f.read()
    return args.text


# ---------------------------------------------------------------- living
def cmd_talk(args):
    m = _mind(args)
    print("Ruth. Type to talk. '/private <text>' tells her a secret, '/sleep' lets her dream,"
          " '/good' or '/bad' gives feedback, Ctrl-D leaves.")
    try:
        while True:
            line = input("you> ")
            if line.startswith("/private "):
                print("ruth> (understood)", m.teach(line[9:] + "\n", private=True)["private_bytes"],
                      "bytes kept to herself")
            elif line.strip() == "/sleep":
                from . import dream
                r = dream.sleep(m)
                print("ruth> (dreamt)", r["dreams"][0]["dream"] if r["dreams"] else "")
            elif line.strip() in ("/good", "/bad"):
                m.feedback(line.strip() == "/good")
            elif line.strip():
                r = m.converse(line)
                print("ruth>", r["text"] + (" …" if r["withheld"] else ""))
    except (EOFError, KeyboardInterrupt):
        print()
    m.save()


def cmd_teach(args):
    m = _mind(args)
    t0 = time.time()
    r = m.teach(_text_arg(args), private=args.private)
    m.save()
    r["seconds"] = round(time.time() - t0, 2)
    _out(r)


def cmd_say(args):
    m = _mind(args)
    from .senses import byte_code
    for c in args.prompt.encode("utf-8"):
        m.brain.step({"text": byte_code(c)}, 1.0, learn=False)
    r = m.speak(args.n, stop=b"\n")
    print(r["text"] + (" … (withheld)" if r["withheld"] else ""))


def cmd_think(args):
    m = _mind(args)
    from .senses import byte_code
    for c in args.prompt.encode("utf-8"):
        m.brain.step({"text": byte_code(c)}, 1.0, learn=False)
    d = m.deliberate(args.n)
    _out([{"thought": m.mask(p["bytes"]), "free_energy": p["free_energy"],
           "chosen": p is d["chosen"]} for p in d["considered"]])


def cmd_sleep(args):
    from . import dream
    m = _mind(args)
    r = dream.sleep(m, dreams=args.dreams)
    m.save()
    _out(r)


def cmd_dreams(args):
    from . import dream
    _out(dream.journal(_mind(args), args.last))


def cmd_check(args):
    m = _mind(args)
    found = {}
    for p in args.files:
        with open(p, "rb") as f:
            spans = m.recognize_private(f.read())
        if spans:
            found[p] = len(spans)
    _out({"recognised_private": found})
    if found:
        sys.exit(1)


def cmd_status(args):
    m = _mind(args)
    b = m.brain
    params = sum(a.size for k, a in b.arrays().items()
                 if not k.startswith(("replay.", "mem.", "rls.p")))
    _out({"identity": m.identity, "version": __version__, "home": m.home, "senses": b.cfg.senses,
          "cell": b.cfg.cell, "neurons": b.cfg.neurons, "moments_lived": b.steps,
          "sleeps": b.sleeps, "working_memories": b.memory.w_n,
          "longterm_basins": b.memory.l_n, "parameters": int(params),
          "private_cues": len(m.cues), "temperament": m.temperament.snapshot(),
          "conversations": m.temperament.age})


def _ngram_acc(text: bytes, order: int, start: int) -> float:
    counts = defaultdict(Counter)
    ok = 0
    for t in range(len(text)):
        ctx = text[max(0, t - order + 1):t]
        if t >= start and counts[ctx]:
            ok += counts[ctx].most_common(1)[0][0] == text[t]
        counts[ctx][text[t]] += 1
    return ok / max(len(text) - start, 1)


def cmd_bench(args):
    from .config import BrainConfig
    from .engine import Brain
    from .evolve import primer
    text = primer()[: args.bytes]
    start = len(text) * 2 // 3
    res = {"stream": "ruth primer (her own docstrings)", "stream_bytes": len(text),
           "scored_from": start}
    for order in (2, 3, 4):
        res[f"online_{order}gram"] = round(_ngram_acc(text, order, start), 4)
    brain = Brain(BrainConfig(senses={"text": 9}) if args.text_only else BrainConfig())
    t0 = time.time()
    brain.learn_bytes(text[:start])
    r = brain.learn_bytes(text[start:])
    el = time.time() - t0
    res["brain_next_byte_accuracy"] = round(r["byte_accuracy"], 4)
    res["brain_bit_accuracy"] = round(r["bit_accuracy"], 4)
    res["brain_bytes_per_second"] = round(len(text) / el, 1)
    _out(res)


# ---------------------------------------------------------------- senses
def cmd_hear(args):
    from .senses import Cochlea, read_wav
    m = _mind(args)
    data, sr = read_wav(args.wav)
    ear = Cochlea(sr, m.brain.cfg.senses["ears"])
    s = []
    for feat, dt in ear.process(data):
        m.brain.step({"ears": feat}, dt / 0.01)  # time unit = 10 ms
        s.append(m.brain.last.get("surprise", 0.0))
    m.save()
    print(f"heard {len(data) / sr:.2f}s in {len(s)} moments; mean surprise {np.mean(s or [0]):.3f}")


def cmd_see(args):
    from .senses import EventRetina
    m = _mind(args)
    frames = np.load(args.frames, allow_pickle=False)
    eye = EventRetina(grid=int(round((m.brain.cfg.senses["eyes"] // 2) ** 0.5)))
    n = 0
    for i, frame in enumerate(frames):
        ev, dt = eye.observe(frame, i / args.fps)
        if dt > 0:
            m.brain.step({"eyes": ev}, dt / 0.01)
            n += 1
    m.save()
    print(f"saw {len(frames)} frames -> {n} event moments")


def cmd_voice(args):
    from .senses import Voice, byte_code, write_wav
    m = _mind(args)
    voice = Voice()
    curves = []
    for b in args.text.encode("utf-8"):
        m.brain.step({"text": byte_code(b)}, 1.0, learn=False)
        curves.append(voice.params_from_latent(3.0 * m.brain.core.x[m.brain.core.motor][:4]))
    write_wav(args.out, voice.render(np.array(curves), frame_seconds=0.06))
    print(f"wrote {args.out} (vocal tract driven by her motor neurons; untrained = babble)")


# ---------------------------------------------------------------- self
def cmd_introspect(args):
    _out(_mind(args).brain.graph())


def cmd_patch(args):
    from . import morphogenesis
    m = _mind(args)
    patch = json.loads(args.patch)
    r = morphogenesis.commit(m, patch)
    if r["accepted"]:
        m.save()
    _out(r)
    if not r["accepted"]:
        sys.exit(1)


# ---------------------------------------------------------------- body
def cmd_export(args):
    m = _mind(args)
    print("exported ->", m.brain.export_core(args.out))


def cmd_evolve(args):
    from . import evolve
    if args.kind == "config":
        r = evolve.evolve_config(args.generations, seed=args.seed)
        _out({"archive_size": r["archive"], "best": r["best"], "promoted": bool(r["promoted"])})
    else:
        r = evolve.evolve_patch(os.path.abspath(args.root), args.patch, args.message,
                                commit=not args.no_commit, mind=_mind(args))
        _out(r)
        if not r["accepted"]:
            sys.exit(1)


def cmd_app(args):
    from .app.server import serve
    serve(args.host, args.port, open_browser=not args.no_browser, home=args.home)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ruth", description="Ruth: a token-free continuous mind.")
    p.add_argument("--version", action="version", version=f"ruth {__version__}")
    p.add_argument("--home", help="where her mind lives (default: $RUTH_HOME or the OS data dir)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("app", help="open her interface (local, offline)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7438)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_app)
    sub.add_parser("talk", help="converse in this terminal").set_defaults(fn=cmd_talk)
    s = sub.add_parser("teach", help="tell her something (text, file, or - for stdin)")
    s.add_argument("text", nargs="?", default="-")
    s.add_argument("--private", action="store_true", help="all of it is private; or use {{...}}")
    s.set_defaults(fn=cmd_teach)
    s = sub.add_parser("say", help="let her continue a prompt")
    s.add_argument("prompt")
    s.add_argument("-n", type=int, default=120)
    s.set_defaults(fn=cmd_say)
    s = sub.add_parser("think", help="show the thoughts she weighs before speaking")
    s.add_argument("prompt")
    s.add_argument("-n", type=int, default=60)
    s.set_defaults(fn=cmd_think)
    s = sub.add_parser("sleep", help="sleep: consolidate, rehearse, dream, improve")
    s.add_argument("--dreams", type=int, default=3)
    s.set_defaults(fn=cmd_sleep)
    s = sub.add_parser("dreams", help="read her dream journal")
    s.add_argument("--last", type=int, default=5)
    s.set_defaults(fn=cmd_dreams)
    s = sub.add_parser("check", help="does she recognise anything private in these files?")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_check)
    sub.add_parser("status", help="vital signs and temperament").set_defaults(fn=cmd_status)
    s = sub.add_parser("bench", help="measure against online n-gram baselines")
    s.add_argument("--bytes", type=int, default=14000)
    s.add_argument("--text-only", action="store_true")
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("hear", help="listen to a WAV file")
    s.add_argument("wav")
    s.set_defaults(fn=cmd_hear)
    s = sub.add_parser("see", help="watch frames (.npy array T x H x W)")
    s.add_argument("frames")
    s.add_argument("--fps", type=float, default=30.0)
    s.set_defaults(fn=cmd_see)
    s = sub.add_parser("voice", help="drive her vocal tract while reading text -> WAV")
    s.add_argument("text")
    s.add_argument("out")
    s.set_defaults(fn=cmd_voice)
    sub.add_parser("introspect", help="her operational graph, as data").set_defaults(fn=cmd_introspect)
    s = sub.add_parser("patch", help="structural self-modification through the meta-kernel")
    s.add_argument("patch", help='JSON, e.g. \'{"op": "grow_neurons", "n": 16}\'')
    s.set_defaults(fn=cmd_patch)
    s = sub.add_parser("export", help="write her brain for the C runtime")
    s.add_argument("out")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("evolve", help="gated self-improvement")
    s.add_argument("kind", choices=["config", "patch"])
    s.add_argument("patch", nargs="?")
    s.add_argument("--root", default=".", help="source tree the patch applies to")
    s.add_argument("-g", "--generations", type=int, default=4)
    s.add_argument("--seed", type=int)
    s.add_argument("-m", "--message")
    s.add_argument("--no-commit", action="store_true")
    s.set_defaults(fn=cmd_evolve)

    args = p.parse_args(argv)
    if args.cmd == "evolve" and args.kind == "patch" and not args.patch:
        p.error("evolve patch needs a patch file")
    if args.home:
        os.environ["RUTH_HOME"] = args.home
    args.fn(args)


if __name__ == "__main__":
    main()
