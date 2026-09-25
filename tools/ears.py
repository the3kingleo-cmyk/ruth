#!/usr/bin/env python3
"""ears.py — keep Ruth's ears open.

Her cochlea is a 16-channel filter bank inside her mind, and the app can only
hear while a browser page is open. This is the always-on half: it captures
the machine's microphone with `arecord` and streams fixed windows of audio to
her `/api/hear`, so she is listening whether or not anyone has a browser open.

  ears.py --check     list capture devices and test one window, then exit
  ears.py             listen continuously (Ctrl-C to stop)
  ears.py --duration 30   listen for 30 seconds

Why this lives outside ruth/: her mind must stay token-free and dependency
free (tests/test_token_free.py enforces numpy + stdlib inside the package).
This is *infrastructure* — it moves air into her, exactly as a microphone is.
Nothing here is imported by her, and she never learns a sample twice: each
window is posted with her own dt, so the stream is continuous rather than a
run of identical moments.

Requires an ALSA capture device. Silence is not an error: a virtual sound
card with nothing attached streams zeros forever, and this reports that
plainly instead of pretending she is hearing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

DEFAULT_URL = os.environ.get("RUTH_URL", "http://127.0.0.1:7455")
RATE = 16000
CHANNELS = 1
WINDOW = 1                               # seconds of audio per moment
LOG = os.environ.get("EARS_LOG",
                     os.path.expanduser("~/.local/status/ears.log"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def arecord_bin():
    return shutil.which("arecord")


def capture_devices():
    """Ask ALSA what can actually record."""
    exe = arecord_bin()
    if not exe:
        return None, "arecord is not installed (package: alsa-utils)"
    try:
        out = subprocess.run([exe, "-l"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"arecord -l failed: {exc}"
    devices = [ln.strip() for ln in out.splitlines()
               if "card " in ln or "Subdevice" in ln]
    return devices, None


def capture_window(device, seconds=WINDOW):
    """Record one window to a temp WAV and return (samples, peak) or (None, reason)."""
    exe = arecord_bin()
    if not exe:
        return None, "arecord is not installed"
    path = os.path.join(tempfile.gettempdir(), f"ears-{os.getpid()}.wav")
    # arecord's -d is DURATION in whole seconds, and 0 means "record until
    # killed" -- so a fractional window must round up, never truncate to 0,
    # or this hangs forever instead of taking windows.
    dur = str(max(1, int(round(float(seconds)))))
    cmd = [exe, "-q", "-t", "wav", "-f", "S16_LE", "-r", str(RATE),
           "-c", str(CHANNELS), "-d", dur]
    if device:
        cmd += ["-D", device]
    cmd.append(path)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds * 3 + 10)
    except subprocess.TimeoutExpired:
        return None, "arecord timed out"
    except OSError as exc:
        return None, f"arecord could not start: {exc}"
    if p.returncode != 0 or not os.path.exists(path):
        return None, (p.stderr or "arecord failed").strip()[:200]
    try:
        with wave.open(path) as w:
            n = w.getnframes()
            raw = w.readframes(n)
    except (OSError, wave.Error) as exc:
        return None, f"unreadable wav: {exc}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    import struct
    usable = len(raw) - (len(raw) % 2)
    shorts = struct.unpack(f"<{usable // 2}h", raw[:usable])
    return [s / 32768.0 for s in shorts], (max((abs(s) for s in shorts), default=0))


def post(url, payload, timeout=20):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def her_ears(url, samples):
    """Hand one window to her. Returns (moments, error)."""
    endpoint = url.rstrip("/") + "/api/hear"
    try:
        r = post(endpoint, {"rate": RATE, "samples": samples})
    except urllib.error.URLError as exc:
        return 0, f"cannot reach {endpoint}: {exc.reason}"
    except (OSError, ValueError) as exc:
        return 0, f"post failed: {exc}"
    if "error" in r:
        return 0, r["error"]
    return r.get("moments", 0), None


def check(url, device):
    devices, err = capture_devices()
    if err:
        print(f"  ears: unavailable — {err}")
        return 1
    print("  capture devices:")
    for d in devices:
        print(f"    {d}")
    samples, err = capture_window(device)
    if err:
        print(f"  capture: failed — {err}")
        return 1
    peak = max((abs(s) for s in samples), default=0)
    print(f"  captured {len(samples)} samples in {WINDOW}s, peak {peak:.4f}")
    print("  " + ("real signal on the mic" if peak > 0.002 else
                  "SILENCE — a device exists but nothing is feeding it "
                  "(a virtual sound card with no microphone attached)"))
    moments, err = her_ears(url, samples)
    if err:
        print(f"  her ears: {err}")
        print("  (is `ruth app` running? it serves /api/hear)")
        return 1
    print(f"  her ears: accepted, {moments} moments")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"her app (default {DEFAULT_URL})")
    ap.add_argument("--device", default=os.environ.get("EARS_DEVICE"),
                    help="ALSA device, e.g. plughw:0,0 (default: system default)")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (0 = run until Ctrl-C)")
    ap.add_argument("--check", action="store_true",
                    help="test one window and exit")
    args = ap.parse_args()

    if args.check:
        return check(args.url, args.device)

    log(f"ears opening -> {args.url} (device: {args.device or 'default'})")
    started = time.time()
    total = silent = windows = 0
    try:
        while True:
            samples, err = capture_window(args.device)
            if err:
                log(f"capture error: {err}; retrying in 2s")
                time.sleep(2)
                continue
            windows += 1
            peak = max((abs(s) for s in samples), default=0)
            if peak < 0.002:
                silent += 1
            moments, herr = her_ears(args.url, samples)
            if herr:
                log(f"her ears: {herr} (is `ruth app` up?)")
                time.sleep(2)
                continue
            total += moments
            if windows % 20 == 0:
                log(f"{windows} windows, {total} moments, "
                    f"{silent} silent so far")
            if args.duration and (time.time() - started) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    log(f"ears closing: {windows} windows, {total} moments, {silent} silent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
