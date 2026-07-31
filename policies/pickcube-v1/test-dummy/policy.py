"""Dummy random policy for testing the arena pipeline.

This policy takes random actions. It will score low but should
successfully load, produce valid actions, and complete evaluation.
"""

import numpy as np


class Policy:
    def __init__(self, task_id: str, obs_space, action_space):
        self.action_space = action_space
        self.rng = np.random.default_rng(42)

    def predict(self, obs) -> np.ndarray:
        action = self.rng.uniform(
            low=self.action_space.low,
            high=self.action_space.high,
            size=self.action_space.shape,
        ).astype(np.float32)
        return action
