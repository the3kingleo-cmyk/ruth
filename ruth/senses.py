"""Senses and actuators: continuous signals in, continuous signals out.

Nothing here produces tokens. Each sense turns a physical stream into a dense
float vector per moment in time (plus the elapsed time ``dt``), and each
actuator turns continuous parameter curves back into a physical stream.

* Text  -> ``byte_code``: a byte is a point in R^9 (8 bipolar bit levels plus
  one amplitude), like a voltage on a parallel bus. No vocabulary or lookup
  table exists anywhere. Quantisation only happens at the output device, the
  way a DAC or a keyboard is discrete.
* Ears  -> ``Cochlea``: a bank of resonant band-pass filters on the raw
  waveform, half-wave rectified, log-compressed and leakily integrated.
* Eyes  -> ``EventRetina``: event-camera emulation. Only log-intensity *changes*
  beyond a contrast threshold fire (ON/OFF), pooled into a coarse grid. Static
  scenes produce silence; there are no frames downstream of the retina.
* Mouth -> ``Voice``: a source-filter synthesiser driven by continuous
  pitch/loudness/formant curves.
"""
from __future__ import annotations

import wave

import numpy as np

BYTE_DIM = 9
_BITS = (1 << np.arange(8)).astype(np.int64)


def byte_code(b: int) -> np.ndarray:
    """Continuous 9-d signal for one byte value."""
    bits = (int(b) & _BITS) > 0
    out = np.empty(BYTE_DIM)
    out[:8] = np.where(bits, 1.0, -1.0)
    out[8] = int(b) / 127.5 - 1.0
    return out


def byte_decode(v: np.ndarray) -> int:
    """Actuator boundary: the continuous bit levels drive an 8-line bus."""
    return int(np.dot(np.asarray(v[:8]) > 0.0, _BITS))


def bytes_signal(data: bytes) -> np.ndarray:
    """Whole byte stream as a (T, 9) continuous trajectory."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int64)
    out = np.empty((arr.size, BYTE_DIM))
    out[:, :8] = np.where((arr[:, None] & _BITS[None, :]) > 0, 1.0, -1.0)
    out[:, 8] = arr / 127.5 - 1.0
    return out


class Cochlea:
    """Cochlea-inspired filterbank: raw pressure -> continuous band energies."""

    def __init__(self, sample_rate: int = 16000, channels: int = 16,
                 fmin: float = 80.0, fmax: float | None = None,
                 hop: int = 160, q: float = 4.0):
        self.sr = sample_rate
        self.hop = hop
        fmax = fmax or 0.45 * sample_rate
        self.freqs = np.geomspace(fmin, fmax, channels)
        # Two-pole resonators (RBJ band-pass, constant 0 dB peak gain).
        w0 = 2 * np.pi * self.freqs / sample_rate
        alpha = np.sin(w0) / (2 * q)
        a0 = 1 + alpha
        self.b0 = alpha / a0
        self.b2 = -alpha / a0
        self.a1 = -2 * np.cos(w0) / a0
        self.a2 = (1 - alpha) / a0
        self.reset()

    @property
    def dim(self) -> int:
        return self.freqs.size

    def reset(self) -> None:
        n = self.freqs.size
        self._x1 = self._x2 = 0.0
        self._y1 = np.zeros(n)
        self._y2 = np.zeros(n)
        self._env = np.zeros(n)

    def process(self, samples: np.ndarray):
        """Yield (features, dt) every ``hop`` samples."""
        leak = np.exp(-1.0 / (0.010 * self.sr))  # 10 ms envelope
        acc = np.zeros(self.freqs.size)
        count = 0
        for s in np.asarray(samples, dtype=float):
            y = self.b0 * s + self.b2 * self._x2 - self.a1 * self._y1 - self.a2 * self._y2
            self._x2, self._x1 = self._x1, s
            self._y2, self._y1 = self._y1, y
            self._env = leak * self._env + (1 - leak) * np.maximum(y, 0.0)
            acc += self._env
            count += 1
            if count == self.hop:
                yield np.log1p(100.0 * acc / count) / np.log(101.0), self.hop / self.sr
                acc[:] = 0.0
                count = 0


class EventRetina:
    """Event-based vision: per-pixel log-intensity change detectors."""

    def __init__(self, grid: int = 8, threshold: float = 0.15):
        self.grid = grid
        self.threshold = threshold
        self._ref = None
        self._t = None

    @property
    def dim(self) -> int:
        return 2 * self.grid * self.grid

    def reset(self) -> None:
        self._ref = None
        self._t = None

    def observe(self, frame: np.ndarray, t: float):
        """Feed one intensity image (H, W) captured at time ``t`` seconds.

        Returns (event_rates, dt): ON and OFF event counts per grid cell,
        normalised by elapsed time, i.e. a continuous flow signal.
        """
        logi = np.log(np.asarray(frame, dtype=float) + 1e-3)
        if self._ref is None:
            self._ref, self._t = logi, t
            return np.zeros(self.dim), 0.0
        dt = max(t - self._t, 1e-6)
        diff = logi - self._ref
        on = diff > self.threshold
        off = diff < -self.threshold
        fired = on | off
        self._ref = np.where(fired, logi, self._ref)  # pixels reset only when they fire
        self._t = t
        h, w = logi.shape
        g = self.grid
        ys = (np.arange(h) * g) // h
        xs = (np.arange(w) * g) // w
        cell = ys[:, None] * g + xs[None, :]
        per_cell = np.bincount(cell.ravel(), minlength=g * g).astype(float)
        on_c = np.bincount(cell.ravel(), weights=on.ravel(), minlength=g * g) / per_cell
        off_c = np.bincount(cell.ravel(), weights=off.ravel(), minlength=g * g) / per_cell
        return np.concatenate([on_c, off_c]), dt


class Voice:
    """Source-filter vocal tract driven by continuous parameter curves."""

    PARAMS = ("f0", "loudness", "formant1", "formant2")

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    @staticmethod
    def params_from_latent(v: np.ndarray) -> np.ndarray:
        """Map a bounded 4-d latent command (any reals) to physical ranges."""
        s = 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=float)))
        return np.array([80 + 220 * s[0], s[1], 300 + 700 * s[2], 900 + 1600 * s[3]])

    def render(self, curves: np.ndarray, frame_seconds: float = 0.01) -> np.ndarray:
        """curves: (T, 4) rows of (f0 Hz, loudness 0..1, F1 Hz, F2 Hz)."""
        curves = np.asarray(curves, dtype=float)
        per = max(1, int(round(frame_seconds * self.sr)))
        n = curves.shape[0] * per
        t_frames = np.arange(curves.shape[0]) * per
        t = np.arange(n)
        interp = np.stack([np.interp(t, t_frames, curves[:, i]) for i in range(4)], axis=1)
        phase = np.cumsum(interp[:, 0] / self.sr)
        source = 2.0 * (phase - np.floor(phase)) - 1.0  # glottal sawtooth
        out = source
        for k in (2, 3):  # two formant resonators with time-varying centres
            y1 = y2 = 0.0
            x1 = x2 = 0.0
            res = np.empty(n)
            for i in range(n):
                w0 = 2 * np.pi * interp[i, k] / self.sr
                alpha = np.sin(w0) / (2 * 5.0)
                a0 = 1 + alpha
                y = (alpha * out[i] - alpha * x2 + 2 * np.cos(w0) * y1 - (1 - alpha) * y2) / a0
                x2, x1 = x1, out[i]
                y2, y1 = y1, y
                res[i] = y
            out = res
        out = out * interp[:, 1]
        peak = np.max(np.abs(out)) or 1.0
        return 0.9 * out / peak


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(raw, dtype=dtype).astype(float)
    if width == 1:
        data = data - 128.0
    data /= float(2 ** (8 * width - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path: str, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
