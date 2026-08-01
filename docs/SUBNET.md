# Machina Arena — Bittensor Subnet Architecture

## Overview

Machina Arena is a Bittensor subnet where miners compete to produce the best robotics policies. Validators evaluate policies in simulation and score them. TAO emissions reward the best policies.

## How it works

### Miner
1. Registers a hotkey on the subnet (UID)
2. Trains a policy on the current task (e.g., PickCube on SO-100)
3. Serves an axon endpoint that accepts observation states and returns actions
4. Validator queries the miner's axon with task observations
5. Miner returns actions — the policy's response to those observations
6. Earnings based on policy performance (success rate, reward)

### Validator
1. Registers a hotkey with sufficient stake
2. For each evaluation cycle (tempo):
   - Loads the current task in ManiSkill
   - Queries each miner's axon with observations from the simulation
   - Runs the miner's actions through the simulation
   - Scores the miner (normalized reward, success rate)
3. Sets weights on chain based on miner scores
4. Yuma Consensus distributes TAO based on weights

### Subnet Owner (Us)
1. Defines tasks (which ManiSkill environments, which robots)
2. Sets difficulty parameters (domain randomization levels)
3. Maintains the scoring code (open source)
4. Earns 18% of subnet emissions
5. Can rotate tasks over time (new tasks = new competitions)

## Architecture

```
machina_arena/
  __init__.py
  task_spec.py              # Platform-agnostic task interface (existing)
  tasks/                     # Task definitions (existing)
  eval.py                    # Evaluation engine (existing)
  render_submission.py       # Rendering (existing)
  render_patch.py            # SAPIEN lavapipe patch (existing)

  subnet/                    # Bittensor subnet code (new)
    __init__.py
    protocol.py              # Wire protocol (observation -> action)
    miner.py                 # Miner neuron — serves policy via axon
    validator.py             # Validator neuron — evaluates miners
    scoring.py               # Scoring mechanism (reward, success rate)
    tasks.py                 # Task management (which task is active)
    config.py                # Subnet configuration
```

## Protocol

The miner-validator protocol is simple:

1. Validator sends an observation (numpy array, the current sim state)
2. Miner returns an action (numpy array, what the robot should do)
3. This repeats for each step of the episode
4. Validator runs the full episode and scores the miner

This means miners don't submit policy weights — they serve inference live. The validator drives the simulation, sends observations, receives actions, and evaluates the full rollout.

## Scoring

Score = average normalized dense reward over N episodes on the current task.

- Each episode: validator resets the env with a random seed, sends observations, receives actions, steps the sim, accumulates reward
- N episodes per evaluation cycle (e.g., 100)
- Normalized reward = ManiSkill's compute_normalized_dense_reward (0 to 1)
- Final score = mean across all episodes

Weights on chain = softmax of scores across all miners.

## Why this is better than the PR model

The subnet model has key advantages over the GitHub PR approach:

1. **Live evaluation.** Validators query miners in real-time. No waiting for CI.
2. **Continuous competition.** Every tempo (360 blocks ~72 min), miners are re-evaluated.
3. **No GitHub dependency.** Miners run their own infrastructure.
4. **Decentralized validation.** Multiple validators independently score miners.
5. **TAO emissions.** Real financial incentive, not just reputation.

## What stays the same

- ManiSkill as the sim backend
- SO-100 as the first robot
- PickCube as the first task
- Domain randomization difficulty ladder
- Video rendering (validators can render top policies)
- Platform-agnostic task spec

## Cost to launch

- Subnet registration: ~1000 TAO (dynamic, check with `btcli query subnet-registration-cost`)
- This is a lock, not a burn — enters the subnet's liquidity pool
- 4090 GPU for validator (we already have the pod)
- Ongoing: validator + miner infrastructure

## Bittensor v11 notes

- SDK: `pip install bittensor` (v11, unified package)
- CLI: `btcli`
- Axon/Synapse replaced with signed requests
- Old subnet template (bittensor-subnet-template) is archived/deprecated
- Must use v11 SDK patterns
