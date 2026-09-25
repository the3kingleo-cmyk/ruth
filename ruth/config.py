"""Brain configuration.

Every number here is sized for the owner's machine: an Intel Core i3 Chromebook
(2 vCPU, 6.4 GiB RAM, no GPU, no swap). The default brain holds ~0.3 M floats
(a few MiB) and steps in well under a millisecond on that CPU.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class BrainConfig:
    # Named continuous input channels -> dimensionality. Text arrives as a
    # 9-d continuous byte signal (see senses.byte_code); audio/vision/sensors
    # add their own channels. No vocabulary, no embedding table.
    senses: dict = field(default_factory=lambda: {"text": 9, "valence": 1, "ears": 16, "eyes": 32})
    # Teaching signals are *told*, never perceived: they are learning targets
    # only, so the brain must infer them from content. "valence" < 0 marks
    # what she was told is private (see mind.py: discretion).
    teaching_signals: list = field(default_factory=lambda: ["valence"])
    # Channels the brain predicts (active inference target). None = all senses.
    targets: list | None = None

    # Perceiver-style field encoder (shared latent manifold)
    latent: int = 64
    fourier: int = 16
    encoder_hidden: int = 96

    # Neural Circuit Policy wiring (sparse, layered like C. elegans)
    inter: int = 96
    command: int = 64
    motor: int = 32
    sensory_fanin: int = 24      # inputs per inter neuron
    inter_fanin: int = 16        # inter -> command
    command_recurrent: int = 12  # command -> command
    motor_fanin: int = 16        # command -> motor
    cell: str = "cfc"            # "cfc" (closed-form), "ltc" (ODE) or "column" (cortical CTRNN)
    column_size: int = 16        # cortical columns: neurons per column
    column_intra: float = 0.5    # connection probability inside a column
    column_lateral: float = 0.15  # ... to the neighbouring columns (ring)
    grown: int = 0               # neurons added by morphogenesis after birth
    ltc_unfolds: int = 3
    input_scale: float = 1.0
    recurrent_scale: float = 1.2
    tau_min: float = 0.25
    tau_max: float = 32.0
    homeostasis: float = 1e-3    # liquid time-constant adaptation rate
    activity_target: float = 0.5

    # Selective state-space memory field (Mamba-style, diagonal, ZOH)
    ssm_channels: int = 48
    ssm_state: int = 16
    ssm_rate_min: float = 1e-3   # slowest decay rate (~1000 time units memory)
    ssm_rate_max: float = 1.0
    dt_scale: float = 1.0

    # Legendre sensory trace (echoic/iconic memory): continuous sliding window
    # of the raw input, order-q Legendre coefficients over ``lmu_window`` time.
    trace_senses: list = field(default_factory=lambda: ["text"])
    lmu_order: int = 6
    lmu_window: float = 3.0
    # A second, slower Legendre window gives the memory key sentence-scale
    # context (which subject an "is" belongs to), weighted below the fast one.
    context_order: int = 6
    context_window: float = 12.0
    context_weight: float = 0.5
    # A third, sentence-scale window: which question is being answered.
    episode_order: int = 12
    episode_window: float = 24.0
    episode_weight: float = 0.5      # 0 disables it
    thought_weight: float = 0.3  # share of liquid/SSM state in the memory key
    trace_in_decoder: bool = True

    # Associative attractor memory (modern continuous Hopfield)
    key_dim: int = 64            # projected liquid/SSM part of the key
    beta: float = 300.0
    hopfield_iters: int = 1
    working_capacity: int = 512
    longterm_capacity: int = 8192
    merge_threshold: float = 0.97
    surprise_k: float = -1.0     # store when surprise > mean + k * std
    warmup: int = 32
    auto_sleep: bool = True

    # Thalamic routing: g_k = softmax(-beta E_k), E_k = 1/2 z.z - b_k.z
    thalamic_beta: float = 4.0
    thalamic_rate: float = 0.01

    # Hippocampal fast weights (dual-trace Hebbian), decay per moment
    fast_decay: float = 0.99

    # Cerebellum: forward model of the cortex (recursive pseudo-inverse)
    cerebellum: bool = True
    cerebellum_lambda: float = 0.999
    cerebellar_gain: float = 0.0     # drift damping in imagination (measured: hurts; off)

    # Continuous decoder, trained online by recursive least squares
    rls_lambda: float = 0.9995
    rls_delta: float = 1.0
    replay_capacity: int = 1024
    replay_passes: int = 1

    seed: int = 7

    @property
    def neurons(self) -> int:
        return self.inter + self.command + self.motor + self.grown

    @property
    def input_dim(self) -> int:
        return sum(self.senses.values())

    @property
    def target_names(self) -> list:
        return list(self.targets) if self.targets else list(self.senses)

    @property
    def output_dim(self) -> int:
        return sum(self.senses[n] for n in self.target_names)

    def to_json(self) -> str:
        # key order is meaningful: sense order defines the signal layout
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "BrainConfig":
        return cls(**json.loads(text))
