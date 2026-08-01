# Machina Arena

Open robotics benchmark. Train policies in simulation, prove them in the arena.

## How it works

1. **We define tasks.** Pick up a cube. Stack blocks. Turn a faucet. Each task has a success condition and a reward function.
2. **You train a policy.** Use any method — RL, imitation learning, whatever. Keep your secret sauce.
3. **You submit.** Send us your policy weights + inference code via pull request.
4. **The arena evaluates.** Your policy runs on 1000 random seeds. Average reward is your score.
5. **Beat the champion.** If your score beats the current champion, your PR merges and you're the new champion.
6. **It gets harder.** Tasks start simple. We add domain randomization over time — friction, mass, lighting. The champion must defend or lose the throne.

## Why use Machina Arena

- **Benchmark your policies** against the community on standardized tasks
- **See results** — every submission is rendered to video
- **Sim2Real ready** — domain randomization and ManiSkill's Sim2RealEnv mean winning policies can deploy to real robots
- **Open weights** — all champion policies are open source
- **Works on cheap hardware** — SO-100 is a $200 arm anyone can build

## Current tasks

| Task | Robot | Backend | Status |
|------|-------|---------|--------|
| PickCube-v1 | SO-100 | ManiSkill | Active |

More tasks coming: PushCube, StackCube, PegInsertion, TurnFaucet, G1 tasks.

## How to submit

1. Clone this repo
2. Train a policy on the task using ManiSkill
3. Create a directory under `policies/<task>/<your-name>/`
4. Add `policy.py`, `weights.pt`, and `requirements.txt`
5. Open a PR

CI runs your policy on 1000 seeds, posts the score as a comment, and renders a video. If you beat the champion, your PR merges.

See [MINING.md](docs/MINING.md) for the full guide.

## Dashboard

Leaderboard, task cards, and submission videos at [buckzz7.github.io/machina-arena/](https://buckzz7.github.io/machina-arena/).

## Platform-agnostic

Tasks currently run on ManiSkill (SAPIEN/PhysX). The task spec interface is backend-agnostic — Genesis, Isaac Lab, and MuJoCo backends are planned. The arena doesn't care which physics engine your task uses, only that your policy scores.

## For enterprises

Need a benchmark for your robot? We build custom tasks for specific robots and use cases. Winning policies can be deployed to real hardware via ManiSkill's Sim2RealEnv. Contact us for custom task creation and sim2real deployment.

## License

MIT
