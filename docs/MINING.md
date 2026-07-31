# Mining Guide

## What you do

You train a robot policy and submit it. If it beats the current champion, your PR merges and you earn TAO through Gittensor.

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

The 1000 seeds are deterministic — derived from a hash of the PR number and the current date. They're posted in the PR comment. You can reproduce the evaluation locally.

## How to train

### Install ManiSkill

```bash
pip install mani-skill
```

### Train on PickCube with SO-100

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCubeSO100-v1", obs_mode="state", render_mode="rgb_array")
obs, info = env.reset(seed=0)

# Train your policy here using PPO, SAC, or any method
# See ManiSkill baselines: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines

env.close()
```

### Test locally

```bash
python -m machina_arena.eval --task pickcube-v1 --policy policies/pickcube-v1/your-name/
```

## Domain randomization

Tasks start with position randomization only. Over time we add:
- Friction randomization
- Mass randomization
- Lighting randomization
- Camera jitter

When a new randomization level is added, the champion is re-evaluated. If it fails, the throne opens.

## Earning TAO

1. Register on [Gittensor](https://gittensor.io)
2. Link your GitHub account
3. Submit PRs to this repo
4. Merged PRs earn TAO based on code contribution (AST token analysis)

See [Gittensor docs](https://docs.gittensor.io) for details.
