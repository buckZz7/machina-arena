"""Render a policy submission and upload video as GitHub Release asset.

Runs on the self-hosted runner (GPU pod). Called by CI after eval completes.

Usage:
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    python render_submission.py \
        --task pickcube-v1 \
        --policy-dir policies/pickcube-v1/test-dummy \
        --scores-json /tmp/result.json \
        --pr 10 \
        --repo buckZz7/machina-arena \
        --token $GITHUB_TOKEN
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Must import render_patch before gymnasium/mani_skill
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_patch  # noqa: F401

import gymnasium as gym
import mani_skill.envs
import imageio

TASK_ENVS = {
    "pickcube-v1": "PickCubeSO100-v1",
}


def load_policy(policy_dir, task_id, env):
    policy_path = os.path.join(policy_dir, "policy.py")
    spec = importlib.util.spec_from_file_location("policy_module", policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy(
        task_id=task_id,
        obs_space=env.observation_space,
        action_space=env.action_space,
    )


def get_seeds(task_id, n_seeds, salt=0):
    seeds = []
    for i in range(n_seeds):
        h = hashlib.sha256(f"{task_id}:{salt}:{i}".encode()).hexdigest()
        seeds.append(int(h[:8], 16))
    return seeds


def render(task_id, policy_dir, scores_path, pr_number, repo, token):
    with open(scores_path) as f:
        result = json.load(f)

    scores = result["scores_per_seed"]
    n_seeds = result["n_seeds"]
    salt = 0  # Same salt used in eval
    seeds = get_seeds(task_id, n_seeds, salt)

    env_name = TASK_ENVS.get(task_id, task_id)

    # Pick 3 seeds: worst, median, best
    scores_arr = np.array(scores)
    sorted_idx = np.argsort(scores_arr)
    render_indices = [
        int(sorted_idx[0]),                    # worst
        int(sorted_idx[len(sorted_idx) // 2]), # median
        int(sorted_idx[-1]),                   # best
    ]

    frames = []
    for idx in render_indices:
        seed = seeds[idx]
        env = gym.make(env_name, obs_mode="state", render_mode="rgb_array",
                       num_envs=1, render_backend="cpu")
        obs, info = env.reset(seed=seed)

        if hasattr(obs, "numpy"):
            obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs.numpy()

        policy = load_policy(policy_dir, task_id, env)

        seed_frames = []
        for step in range(50):
            action = policy.predict(obs)
            if hasattr(action, "numpy"):
                action = action.numpy()
            elif not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(obs, "numpy"):
                obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs.numpy()

            frame = env.render()
            if frame is not None:
                f = frame.cpu().numpy() if hasattr(frame, "cpu") else np.array(frame)
                if f.ndim == 4:
                    f = f[0]
                seed_frames.append(f.astype(np.uint8))

            if bool(terminated or truncated):
                break

        env.close()
        frames.extend(seed_frames)
        # Separator frame
        if seed_frames:
            h, w = seed_frames[0].shape[:2]
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Save video
    video_path = f"/tmp/render_pr{pr_number}.mp4"
    if frames:
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Video: {video_path} ({len(frames)} frames)")

    # Upload to GitHub Release
    if os.path.exists(video_path) and token:
        upload_release(video_path, pr_number, repo, token)

    return video_path


def upload_release(video_path, pr_number, repo, token):
    import urllib.request

    tag = f"render-pr-{pr_number}"

    # Create release
    data = json.dumps({
        "tag_name": tag,
        "name": f"Render PR #{pr_number}",
        "body": f"Policy rendering for PR #{pr_number}",
        "prerelease": True,
    }).encode()

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases",
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github.v3+json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            release = json.loads(resp.read())
            upload_url = release.get("upload_url", "").replace("{?name,label}", "")
            print(f"Release created: {release.get('html_url')}")
    except Exception as e:
        print(f"Release failed: {e}")
        return

    # Upload asset
    with open(video_path, "rb") as f:
        video_data = f.read()

    req = urllib.request.Request(
        f"{upload_url}?name=render.mp4",
        data=video_data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "video/mp4",
            "Accept": "application/vnd.github.v3+json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            asset = json.loads(resp.read())
            url = asset.get("browser_download_url", "")
            print(f"Asset uploaded: {url}")
            return url
    except Exception as e:
        print(f"Upload failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--scores-json", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--repo", default="buckZz7/machina-arena")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    render(args.task, args.policy_dir, args.scores_json, args.pr, args.repo, args.token)


if __name__ == "__main__":
    main()
