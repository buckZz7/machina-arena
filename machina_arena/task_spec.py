"""Machina Arena — Task Spec Interface

The platform-agnostic interface for defining robotics tasks.
A task defines how to build an environment, what success looks like,
and how to score a policy. The physics backend is pluggable.

Currently supported backends:
- maniskill (SAPIEN/PhysX, GPU-parallel, 2k+ objects)

Planned backends:
- genesis (multi-physics: cloth, liquid, FEM)
- mujoco (MJX, JAX-native)
- isaac_lab (PhysX 5, NVIDIA)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskInfo:
    """Metadata about a task."""
    task_id: str           # e.g. "pickcube-v1"
    name: str              # e.g. "Pick Cube"
    description: str       # Human-readable task description
    robot: str             # e.g. "so100", "panda", "g1"
    backend: str           # e.g. "maniskill", "genesis"
    max_episode_steps: int # e.g. 50
    difficulty: str = "v1" # e.g. "v1", "v1.1" (randomization level)


@dataclass
class EvalResult:
    """Result of evaluating a policy on a task."""
    score: float           # Average normalized reward [0, 1]
    success_rate: float    # Fraction of episodes that succeeded [0, 1]
    n_seeds: int           # Number of seeds evaluated
    seeds: list = field(default_factory=list)  # The seeds used
    scores_per_seed: list = field(default_factory=list)  # Per-seed scores
    video_path: Optional[str] = None  # Path to rendered video


class TaskSpec(ABC):
    """Platform-agnostic task specification.

    A task defines:
    1. How to build a Gymnasium environment
    2. What info to expose (metadata)
    3. (Optional) How to build a sim2real environment

    The physics backend is determined by the task implementation,
    not by the arena. The arena only cares about the score.
    """

    @abstractmethod
    def info(self) -> TaskInfo:
        """Return task metadata."""
        pass

    @abstractmethod
    def build_env(self, seed: int = 0, **kwargs):
        """Build and return a Gymnasium environment.

        The returned env must implement the standard Gymnasium API:
        - env.reset(seed=...) -> (obs, info)
        - env.step(action) -> (obs, reward, terminated, truncated, info)
        - env.close()

        The env's reward is the normalized dense reward [0, 1].
        The env's info dict must contain 'success' (bool).
        """
        pass

    def build_real_env(self, real_robot_config: dict, **kwargs):
        """Build a real-world environment for sim2real deployment.

        Optional — only implemented for backends that support sim2real.
        """
        raise NotImplementedError("Sim2Real not supported for this task")

    def get_seeds(self, n_seeds: int = 1000, salt: int = 0) -> list[int]:
        """Generate deterministic evaluation seeds.

        Seeds are derived from the task_id and a salt (e.g. PR number + date).
        This ensures reproducibility while preventing seed memorization.
        """
        import hashlib
        task_id = self.info().task_id
        seeds = []
        for i in range(n_seeds):
            h = hashlib.sha256(f"{task_id}:{salt}:{i}".encode()).hexdigest()
            seeds.append(int(h[:8], 16))
        return seeds
