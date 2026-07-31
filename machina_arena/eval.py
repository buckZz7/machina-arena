"""Machina Arena — Evaluation Runner

Runs a policy on a task across N seeds, computes the average score,
and optionally renders a video of the policy executing the task.
"""

import argparse
import json
import os
import sys
import importlib.util
import numpy as np

from machina_arena.tasks import get_task


def load_policy(policy_dir: str, task_id: str):
    """Load a policy from a directory.

    Expects policy.py with a Policy class and weights.pt.
    """
    policy_path = os.path.join(policy_dir, "policy.py")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"No policy.py found in {policy_dir}")

    # Install requirements if any
    req_path = os.path.join(policy_dir, "requirements.txt")
    if os.path.exists(req_path):
        import subprocess
        with open(req_path) as f:
            reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if reqs:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + reqs,
                         capture_output=True)

    # Load the policy module
    spec = importlib.util.spec_from_file_location("policy_module", policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Policy"):
        raise AttributeError(f"policy.py must define a Policy class")

    return module.Policy


def evaluate(
    task_id: str,
    policy_dir: str,
    n_seeds: int = 1000,
    salt: int = 0,
    render: bool = False,
    render_seeds: int = 3,
) -> dict:
    """Evaluate a policy on a task.

    Returns a dict with:
    - score: average normalized reward [0, 1]
    - success_rate: fraction of episodes that succeeded
    - n_seeds: number of seeds evaluated
    - seeds: list of seeds used
    - scores_per_seed: per-seed scores
    - video_path: path to rendered video (if render=True)
    """
    task = get_task(task_id)
    info = task.info()
    seeds = task.get_seeds(n_seeds=n_seeds, salt=salt)

    # Load policy
    PolicyClass = load_policy(policy_dir, task_id)

    # Build a sample env to get obs/action spaces
    sample_env = task.build_env(seed=0)
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    policy = PolicyClass(task_id=task_id, obs_space=obs_space, action_space=action_space)

    scores = []
    successes = []

    for i, seed in enumerate(seeds):
        env = task.build_env(seed=seed)
        obs, info_dict = env.reset(seed=seed)

        total_reward = 0.0
        done = False
        success = False

        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info_dict = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            if info_dict.get("success", False):
                success = True

        # Normalized score: ManiSkill's compute_normalized_dense_reward returns [0, 1]
        # If the env has it, use it. Otherwise normalize by max possible reward.
        try:
            norm_reward = float(env.compute_normalized_dense_reward(obs, action=action, info=info_dict))
        except Exception:
            max_steps = task.info().max_episode_steps
            norm_reward = total_reward / max_steps if max_steps > 0 else total_reward

        scores.append(float(norm_reward))
        successes.append(float(success))
        env.close()

        if i % 100 == 0:
            print(f"  Seed {i}/{n_seeds}: reward={norm_reward:.4f} success={success}")

    result = {
        "task_id": task_id,
        "score": float(np.mean(scores)),
        "success_rate": float(np.mean(successes)),
        "n_seeds": n_seeds,
        "seeds": seeds,
        "scores_per_seed": scores,
    }

    if render:
        video_path = render_policy(task, policy, seeds[:render_seeds])
        result["video_path"] = video_path

    return result


def render_policy(task, policy, seeds) -> str:
    """Render a policy on a few seeds and save as video."""
    import imageio

    frames = []
    for seed in seeds:
        env = task.build_env(seed=seed, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)

        for _ in range(task.info().max_episode_steps):
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break

        env.close()

    video_path = "/tmp/machina_arena_render.mp4"
    if frames:
        imageio.mimsave(video_path, frames, fps=20)

    return video_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy in Machina Arena")
    parser.add_argument("--task", required=True, help="Task ID (e.g. pickcube-v1)")
    parser.add_argument("--policy", required=True, help="Path to policy directory")
    parser.add_argument("--seeds", type=int, default=1000, help="Number of seeds")
    parser.add_argument("--salt", type=int, default=0, help="Seed salt (e.g. PR number)")
    parser.add_argument("--render", action="store_true", help="Render video")
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()

    print(f"Machina Arena — Evaluating {args.task}")
    print(f"Policy: {args.policy}")
    print(f"Seeds: {args.seeds}")

    result = evaluate(
        task_id=args.task,
        policy_dir=args.policy,
        n_seeds=args.seeds,
        salt=args.salt,
        render=args.render,
    )

    print(f"\n=== Results ===")
    print(f"Score: {result['score']:.4f}")
    print(f"Success rate: {result['success_rate']:.4f}")
    print(f"Seeds: {result['n_seeds']}")

    if result.get("video_path"):
        print(f"Video: {result['video_path']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
