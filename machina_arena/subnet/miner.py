"""Machina Arena — Miner neuron.

The miner trains a policy on the current task and serves it via an axon.
Validators query the miner with observations; the miner returns actions.

Usage:
    python -m machina_arena.subnet.miner --netuid 42 --wallet.name miner --wallet.hotkey default

The miner loads a policy (trained offline) and serves inference.
"""

import argparse
import os
import sys
import numpy as np

# Render patch for headless rendering (if needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import bittensor as bt
except ImportError:
    bt = None

from machina_arena.subnet.protocol import (
    ActionRequest,
    ActionResponse,
    serialize_array,
    deserialize_array,
)


class PolicyMiner:
    """Serves policy inference to validators.

    The miner loads a trained policy and responds to action requests.
    The policy interface is the same as the PR-based competition:
        class Policy:
            def __init__(self, task_id, obs_space, action_space): ...
            def predict(self, obs) -> np.ndarray: ...
    """

    def __init__(self, policy_path: str, task_id: str):
        self.task_id = task_id
        self.policy = self._load_policy(policy_path, task_id)

    def _load_policy(self, policy_path: str, task_id: str):
        """Load a policy from a directory."""
        import importlib.util

        policy_file = os.path.join(policy_path, "policy.py")
        spec = importlib.util.spec_from_file_location("policy_module", policy_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # We need obs/action spaces — build a sample env
        import gymnasium as gym
        import mani_skill.envs

        env = gym.make("PickCubeSO100-v1", obs_mode="state", num_envs=1, render_backend="none")
        policy = module.Policy(
            task_id=task_id,
            obs_space=env.observation_space,
            action_space=env.action_space,
        )
        env.close()
        return policy

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Run the policy on an observation and return an action."""
        action = self.policy.predict(observation)
        if hasattr(action, "numpy"):
            action = action.numpy()
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        return action.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Machina Arena Miner")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--wallet.name", dest="wallet_name", default="miner")
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    parser.add_argument("--network", default="finney")
    parser.add_argument("--policy-path", required=True, help="Path to policy directory")
    parser.add_argument("--task-id", default="pickcube-v1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    if bt is None:
        print("bittensor not installed. Run: pip install bittensor")
        sys.exit(1)

    # Load the policy
    miner = PolicyMiner(args.policy_path, args.task_id)
    print(f"Policy loaded from {args.policy_path}")

    # TODO: Set up axon and serve
    # In Bittensor v11, this uses the signed requests API
    # The miner serves an HTTP endpoint that accepts observations
    # and returns actions

    print(f"Miner ready on port {args.port}")
    print(f"Subnet: {args.netuid}")
    print(f"Task: {args.task_id}")

    # For now, this is a skeleton. The actual axon serving requires
    # the Bittensor v11 SDK's request handling, which replaces the
    # old Axon/Synapse pattern with signed HTTP requests.


if __name__ == "__main__":
    main()
