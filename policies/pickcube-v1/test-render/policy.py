"""Random policy with seed 137 for testing the full pipeline."""

import numpy as np


class Policy:
    def __init__(self, task_id, obs_space, action_space):
        self.action_space = action_space
        self.rng = np.random.default_rng(137)

    def predict(self, obs):
        return self.rng.uniform(
            low=self.action_space.low,
            high=self.action_space.high,
            size=self.action_space.shape,
        ).astype(np.float32)
