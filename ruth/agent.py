"""Active inference: act to make the world match preferences, and to learn.

For each candidate action the brain *imagines* the next moment (a
counterfactual rollout on a snapshot of its own dynamic state, no learning)
and scores the expected free energy

    G(a) = risk - epistemic
    risk      = 1/2 sum_i pi_i (predicted_obs_i - preferred_obs_i)^2
    epistemic = w * 1/2 log(1 + phi' P phi)     (RLS posterior variance:
                how much this situation would teach the decoder)

The action with the lowest G is taken. Its outcome is then perceived with
learning on, closing the perception-action loop.
"""
from __future__ import annotations

import numpy as np

from .engine import Brain


class ActiveInferenceAgent:
    def __init__(self, brain: Brain, obs: str, act: str, preferred: np.ndarray,
                 actions: list, pref_precision: np.ndarray | float = 1.0,
                 epistemic_weight: float = 0.1, horizon: int = 1, seed: int = 0):
        self.brain = brain
        self.obs = obs
        self.act_sense = act
        self.preferred = np.asarray(preferred, dtype=float)
        self.actions = [np.atleast_1d(np.asarray(a, dtype=float)) for a in actions]
        self.pref_precision = pref_precision
        self.epistemic_weight = epistemic_weight
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.last_scores = None

    def expected_free_energy(self, observation: np.ndarray, action: np.ndarray, dt: float) -> float:
        b = self.brain
        snap = b.snapshot()
        try:
            g = 0.0
            obs = observation
            for _ in range(self.horizon):
                b.step({self.obs: obs, self.act_sense: action}, dt, learn=False)
                pred = b.prediction(self.obs)
                risk = 0.5 * float(np.sum(self.pref_precision * (pred - self.preferred) ** 2))
                epistemic = 0.5 * np.log1p(b.novelty())
                g += risk - self.epistemic_weight * epistemic
                obs = pred
            return g
        finally:
            b.restore(snap)

    def choose(self, observation: np.ndarray, dt: float = 1.0, explore: float = 0.0) -> np.ndarray:
        if explore > 0 and self.rng.random() < explore:
            return self.actions[int(self.rng.integers(len(self.actions)))]
        scores = [self.expected_free_energy(observation, a, dt) for a in self.actions]
        self.last_scores = scores
        return self.actions[int(np.argmin(scores))]

    def live(self, observation: np.ndarray, action: np.ndarray, dt: float = 1.0) -> None:
        """Perceive what actually happened (with learning) after acting."""
        self.brain.step({self.obs: observation, self.act_sense: action}, dt, learn=True)
