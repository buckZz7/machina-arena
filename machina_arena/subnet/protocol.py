"""Machina Arena — Bittensor subnet protocol.

Defines the wire protocol between miners and validators.

Validator sends an observation (sim state), miner returns an action.
This repeats for each step of an episode. The validator drives the
simulation and scores the miner based on total reward.

Protocol flow:
  1. Validator sends task_id + observation
  2. Miner runs policy.predict(obs) and returns action
  3. Validator steps the sim with the action
  4. Repeat until episode ends
  5. Validator scores the episode (normalized reward)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import bittensor as bt
    BittensorSynapse = bt.Synapse if hasattr(bt, "Synapse") else None
except ImportError:
    bt = None
    BittensorSynapse = None


@dataclass
class ActionRequest:
    """Validator -> Miner: here's an observation, give me an action."""
    task_id: str
    observation: np.ndarray
    step: int
    episode: int
    done: bool = False


@dataclass
class ActionResponse:
    """Miner -> Validator: here's the action for that observation."""
    action: np.ndarray
    step: int


# For Bittensor v11, the protocol uses signed requests.
# The miner serves an HTTP endpoint that accepts ActionRequest
# and returns ActionResponse. The validator signs requests with
# its hotkey.

# If using legacy Synapse pattern (for reference):
if BittensorSynapse is not None:
    class PolicySynapse(BittensorSynapse):
        """Synapse for policy inference requests."""
        task_id: str = ""
        observation: bytes = b""  # serialized numpy array
        step: int = 0
        episode: int = 0
        action: bytes = b""  # serialized numpy array

        def deserialize(self) -> "PolicySynapse":
            return self

        def to_request(self) -> dict:
            return {
                "task_id": self.task_id,
                "observation": self.observation,
                "step": self.step,
                "episode": self.episode,
            }

        def to_response(self) -> dict:
            return {
                "action": self.action,
                "step": self.step,
            }


def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize a numpy array for transport."""
    return arr.tobytes()


def deserialize_array(data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    """Deserialize a numpy array from transport."""
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
