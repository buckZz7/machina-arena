"""PickCube task on SO-100 robot, ManiSkill backend.

The SO-100 is a $200 open-source robot arm from TheRobotStudio,
designed to work with HuggingFace's LeRobot library.

This task wraps ManiSkill's built-in PickCubeSO100-v1 environment.
"""

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers all ManiSkill environments

from machina_arena.task_spec import TaskSpec, TaskInfo


class PickCubeSO100Task(TaskSpec):
    """Pick up a cube and place it at a goal position using the SO-100 arm."""

    def info(self) -> TaskInfo:
        return TaskInfo(
            task_id="pickcube-v1",
            name="Pick Cube (SO-100)",
            description=(
                "Pick up a cube with the SO-100 robot arm and move it "
                "to a target goal position."
            ),
            robot="so100",
            backend="maniskill",
            max_episode_steps=50,
            difficulty="v1",
        )

    def build_env(self, seed: int = 0, **kwargs):
        env = gym.make(
            "PickCubeSO100-v1",
            obs_mode=kwargs.get("obs_mode", "state"),
            max_episode_steps=self.info().max_episode_steps,
            num_envs=1,
            render_backend=kwargs.get("render_backend", "none"),
            **{k: v for k, v in kwargs.items() if k not in ("obs_mode", "render_backend")}
        )
        return env
