"""Machina Arena — Validator neuron.

The validator queries miners with task observations, runs their actions
through ManiSkill simulation, and scores them. Weights are set on chain.

Usage:
    python -m machina_arena.subnet.validator --netuid 42 --wallet.name validator --wallet.hotkey default

The validator:
  1. Loads the current task in ManiSkill
  2. For each miner, runs N episodes:
     - Reset env with random seed
     - Send observations to miner, receive actions
     - Step the sim with actions
     - Accumulate reward
  3. Scores each miner (avg normalized reward)
  4. Sets weights on chain via Yuma Consensus
"""

import argparse
import os
import sys
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import bittensor as bt
except ImportError:
    bt = None

import gymnasium as gym
import mani_skill.envs

# Render patch for headless evaluation
try:
    import render_patch  # noqa: F401
except ImportError:
    pass

from machina_arena.subnet.protocol import ActionRequest, ActionResponse
from machina_arena.subnet.scoring import score_miner


# Task registry — same as the PR-based competition
TASK_ENVS = {
    "pickcube-v1": "PickCubeSO100-v1",
}

TASK_CONFIG = {
    "pickcube-v1": {
        "env_name": "PickCubeSO100-v1",
        "max_steps": 50,
        "n_episodes": 100,
        "render_backend": "none",
    },
}


def get_seeds(task_id: str, n_episodes: int, salt: int = 0) -> list[int]:
    """Generate deterministic seeds for evaluation."""
    seeds = []
    for i in range(n_episodes):
        h = hashlib.sha256(f"{task_id}:{salt}:{i}".encode()).hexdigest()
        seeds.append(int(h[:8], 16))
    return seeds


class ArenaValidator:
    """Validates miners by running their policies in ManiSkill simulation."""

    def __init__(self, task_id: str, n_episodes: int = 100):
        self.task_id = task_id
        self.n_episodes = n_episodes
        config = TASK_CONFIG.get(task_id, {})
        self.env_name = config.get("env_name", TASK_ENVS.get(task_id, task_id))
        self.max_steps = config.get("max_steps", 50)
        self.render_backend = config.get("render_backend", "none")

    def evaluate_miner(self, miner_endpoint: str, salt: int = 0) -> dict:
        """Evaluate a miner by querying its axon endpoint.

        For each episode:
        1. Reset env with a random seed
        2. Send observations to miner, receive actions
        3. Step the sim
        4. Accumulate reward

        Returns: {score, success_rate, n_episodes}
        """
        seeds = get_seeds(self.task_id, self.n_episodes, salt)
        scores = []
        successes = []

        for ep_idx, seed in enumerate(seeds):
            env = gym.make(
                self.env_name,
                obs_mode="state",
                num_envs=1,
                render_backend=self.render_backend,
            )
            obs, info = env.reset(seed=seed)

            # Convert obs to numpy
            if hasattr(obs, "numpy"):
                obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs.numpy()
            elif isinstance(obs, dict):
                obs = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in obs.items()}

            total_reward = 0.0
            done = False
            success = False

            for step in range(self.max_steps):
                # Query miner for action
                action = self._query_miner(miner_endpoint, obs, step, ep_idx)

                # Step the simulation
                obs, reward, terminated, truncated, info = env.step(action)
                if hasattr(obs, "numpy"):
                    obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs.numpy()
                elif isinstance(obs, dict):
                    obs = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in obs.items()}

                total_reward += float(reward)
                done = bool(terminated or truncated)
                success = bool(info.get("success", False))
                if done:
                    break

            # Normalized score
            try:
                norm_reward = float(env.compute_normalized_dense_reward(obs, action=action, info=info))
            except Exception:
                norm_reward = total_reward / self.max_steps

            scores.append(norm_reward)
            successes.append(float(success))
            env.close()

        return {
            "score": float(np.mean(scores)),
            "success_rate": float(np.mean(successes)),
            "n_episodes": self.n_episodes,
        }

    def _query_miner(self, endpoint: str, obs: np.ndarray, step: int, episode: int) -> np.ndarray:
        """Query a miner's axon endpoint for an action.

        TODO: Implement using Bittensor v11 signed requests.
        For now, this is a placeholder that returns random actions.
        """
        # Placeholder: return random action
        # Real implementation sends observation to miner's axon
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        return action_space.sample()


def main():
    parser = argparse.ArgumentParser(description="Machina Arena Validator")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--wallet.name", dest="wallet_name", default="validator")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--task-id", default="pickcube-v1")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    if bt is None:
        print("bittensor not installed. Run: pip install bittensor")
        sys.exit(1)

    validator = ArenaValidator(args.task_id, args.episodes)
    print(f"Validator ready")
    print(f"Subnet: {args.netuid}")
    print(f"Task: {args.task_id}")
    print(f"Episodes per miner: {args.episodes}")

    # TODO: Main loop
    # 1. Get metagraph (list of miners)
    # 2. For each miner, evaluate
    # 3. Set weights on chain
    # 4. Wait for next tempo
    # 5. Repeat


if __name__ == "__main__":
    main()
