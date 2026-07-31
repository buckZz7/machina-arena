# Machina Arena

Open robotics policy competition. Train in simulation, prove it in the arena.

## How it works

1. **We define tasks.** Pick up a cube. Stack blocks. Turn a faucet. Each task has a success condition and a reward function.
2. **You train a policy.** Use any method — RL, imitation learning, whatever. Keep your secret sauce.
3. **You submit.** Send us your policy weights + inference code. No training code needed.
4. **The arena evaluates.** Your policy runs on 1000+ random seeds. Average reward is your score.
5. **Beat the champion.** If your score beats the current champion, your PR merges and you're the new champion.
6. **It gets harder.** Tasks start simple. We add domain randomization over time — friction, mass, lighting. The champion must defend or lose the throne.

## Why compete

- **Earn TAO** through Gittensor for every merged PR
- **Your policy works** — every submission is evaluated in simulation and rendered to video
- **Sim2Real ready** — domain randomization and ManiSkill's Sim2RealEnv mean winning policies can deploy to real robots
- **Open weights** — all champion policies are open source

## Current tasks

| Task | Robot | Backend | Status |
|------|-------|---------|--------|
| PickCube-v1 | SO-100 | ManiSkill | Active |

More tasks coming: PushCube, StackCube, PegInsertion, TurnFaucet, G1 tasks, and eventually G1 boxing.

## How to submit

1. Clone this repo
2. Train a policy on the task using ManiSkill (or any compatible platform)
3. Create a directory under `policies/<task>/<your-name>/`
4. Add `policy.py`, `weights.pt`, and `requirements.txt`
5. Open a PR

See [MINING.md](docs/MINING.md) for the full guide.

## Platform-agnostic

Tasks currently run on ManiSkill (SAPIEN/PhysX). The task spec interface is backend-agnostic — Genesis, Isaac Lab, and MuJoCo backends are planned. The arena doesn't care which physics engine your task uses, only that your policy scores.

## Dashboard

Leaderboard, task cards, and submission renderings at our [GitHub Pages site](https://buckzzz7.github.io/machina-arena/).

## License

MIT
