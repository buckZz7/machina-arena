# Mining Guide

## What you do

You train a robot policy and submit it. If it beats the current champion, your PR merges and you're the new champion.

## What you submit

```
policies/
  pickcube-v1/
    your-github-username/
      policy.py          # Inference code (required)
      weights.pt         # Model weights (required)
      requirements.txt   # Python dependencies (required)
```

### policy.py

Your policy must implement this interface:

```python
import numpy as np

class Policy:
    def __init__(self, task_id: str, obs_space, action_space):
        """Load your model here."""
        pass

    def predict(self, obs) -> np.ndarray:
        """Take an observation, return an action."""
        return np.zeros(action_space.shape)
```

### weights.pt

Your trained model weights. Any format — PyTorch, ONNX, JAX. Your `policy.py` loads them.

### requirements.txt

Any pip dependencies beyond what's in the base environment (ManiSkill, numpy, torch, gymnasium).

## What you DON'T submit

- Training scripts (keep your methods secret)
- Training data
- Environment modifications

## How evaluation works

1. **Validation** (fast, CPU): Your policy loads, produces valid actions, doesn't crash on 10 seeds.
2. **Competition** (CPU): Your policy runs on 1000 random seeds of the task. Average normalized dense reward is your score.
3. **Comparison**: Your score vs the current champion's score on the same 1000 seeds.
4. **Decision**: If your score > champion score, your PR merges. You're the new champion.
5. **Video**: 3 seeds (best, worst, median) are rendered and uploaded as a GitHub Release.

The 1000 seeds are deterministic — derived from a hash of the PR number. They're posted in the PR comment. You can reproduce the evaluation locally.

## How to train

### Install ManiSkill

```bash
pip install mani-skill
```

### Train on PickCube with SO-100

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCubeSO100-v1", obs_mode="state", num_envs=1, render_backend="none")
obs, info = env.reset(seed=0)

# Train your policy here using PPO, SAC, or any method
# See ManiSkill baselines: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines

env.close()
```

### Test locally

```bash
pip install -e .
python -m machina_arena.eval --task pickcube-v1 --policy policies/pickcube-v1/your-name/
```

## Domain randomization

Tasks start with position randomization only. Over time we add:
- Friction randomization
- Mass randomization
- Lighting randomization
- Camera jitter

When a new randomization level is added, the champion is re-evaluated. If it fails, the throne opens.

## The SO-100 robot

The SO-100 is a $200 open-source robot arm from TheRobotStudio, designed to work with HuggingFace's LeRobot library. Anyone can build one at home.

If you have a physical SO-100, you can deploy winning policies to it using ManiSkill's Sim2RealEnv and the [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real) project.
