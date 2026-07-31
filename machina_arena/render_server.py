"""Render server for Machina Arena.

Runs on the RunPod GPU pod. Listens for render requests from CI.
Renders 3 seeds (best, worst, median) of a policy and uploads
the video as a GitHub Release asset.

Usage:
    python render_server.py --port 8080

CI calls:
    curl -X POST http://pod-ip:port/render \
        -d '{"policy_url": "...", "task_id": "pickcube-v1", "pr_number": 10, "scores": [...], "github_token": "..."}'
"""

import argparse
import json
import os
import subprocess
import tempfile
import urllib.request
import numpy as np

import gymnasium as gym
import mani_skill.envs
import imageio


def render_policy(
    task_id: str,
    policy_url: str,
    scores: list,
    github_token: str,
    pr_number: int,
    repo: str = "buckZz7/machina-arena",
):
    """Render a policy on 3 seeds (best, worst, median) and upload as GitHub Release."""
    
    # Download policy files
    tmpdir = tempfile.mkdtemp()
    policy_dir = os.path.join(tmpdir, "policy")
    os.makedirs(policy_dir, exist_ok=True)
    
    # Download policy.py and weights from raw GitHub URLs
    for fname in ["policy.py", "requirements.txt", "weights.pt"]:
        url = f"{policy_url}/{fname}"
        try:
            urllib.request.urlretrieve(url, os.path.join(policy_dir, fname))
        except Exception:
            pass  # File may not exist (e.g., no requirements.txt)
    
    # Install requirements
    req_path = os.path.join(policy_dir, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path) as f:
            reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if reqs:
            subprocess.run(["pip", "install", "--quiet"] + reqs, capture_output=True)
    
    # Load policy
    import importlib.util
    spec = importlib.util.spec_from_file_location("policy_module", os.path.join(policy_dir, "policy.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Map task_id to ManiSkill env name
    task_envs = {
        "pickcube-v1": "PickCubeSO100-v1",
    }
    env_name = task_envs.get(task_id, task_id)
    
    # Pick 3 seeds: best, worst, median
    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    worst_idx = int(np.argmin(scores_arr))
    median_idx = int(np.argsort(scores_arr)[len(scores_arr) // 2])
    render_indices = [worst_idx, median_idx, best_idx]
    
    # Generate seeds (same algorithm as task_spec)
    import hashlib
    seeds = []
    for i in range(len(scores)):
        h = hashlib.sha256(f"{task_id}:0:{i}".encode()).hexdigest()
        seeds.append(int(h[:8], 16))
    
    # Build env and render
    frames = []
    for idx in render_indices:
        seed = seeds[idx]
        env = gym.make(env_name, obs_mode="state", render_mode="rgb_array", num_envs=1)
        obs, info = env.reset(seed=seed)
        
        # Convert obs to numpy
        if hasattr(obs, "numpy"):
            obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs.numpy()
        
        # Load policy
        policy = module.Policy(
            task_id=task_id,
            obs_space=env.observation_space,
            action_space=env.action_space,
        )
        
        seed_frames = []
        for step in range(50):  # max_episode_steps
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
                if hasattr(frame, "cpu"):
                    frame = frame.cpu().numpy()
                seed_frames.append(frame)
            
            if bool(terminated or truncated):
                break
        
        env.close()
        # Add label frame between seeds
        if seed_frames:
            frames.extend(seed_frames)
            # Add blank frame as separator
            h, w = seed_frames[0].shape[:2]
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Save video
    video_path = os.path.join(tmpdir, f"render_pr{pr_number}.mp4")
    if frames:
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Video saved: {video_path} ({len(frames)} frames)")
    
    # Upload as GitHub Release asset
    if os.path.exists(video_path) and github_token:
        upload_to_github_release(
            video_path=video_path,
            pr_number=pr_number,
            github_token=github_token,
            repo=repo,
        )
    
    return video_path


def upload_to_github_release(video_path, pr_number, github_token, repo):
    """Create a GitHub Release and upload the video as an asset."""
    import urllib.parse
    
    tag = f"render-pr-{pr_number}"
    
    # Create release
    release_data = json.dumps({
        "tag_name": tag,
        "name": f"Render PR #{pr_number}",
        "body": f"Video rendering for PR #{pr_number}",
        "prerelease": True,
    }).encode()
    
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases",
        data=release_data,
        headers={
            "Authorization": f"token {github_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            release = json.loads(resp.read())
            upload_url = release.get("upload_url", "").replace("{?name,label}", "")
            release_id = release.get("id")
            html_url = release.get("html_url")
            print(f"Release created: {html_url}")
    except Exception as e:
        print(f"Release creation failed: {e}")
        return
    
    # Upload asset
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    asset_name = f"render.mp4"
    req = urllib.request.Request(
        f"{upload_url}?name={asset_name}",
        data=video_data,
        headers={
            "Authorization": f"token {github_token}",
            "Content-Type": "video/mp4",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            asset = json.loads(resp.read())
            print(f"Asset uploaded: {asset.get('browser_download_url', 'N/A')}")
            return asset.get("browser_download_url")
    except Exception as e:
        print(f"Asset upload failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--task", default="pickcube-v1")
    parser.add_argument("--policy-url", required=False)
    parser.add_argument("--scores", required=False)
    parser.add_argument("--pr", type=int, required=False)
    parser.add_argument("--token", required=False)
    args = parser.parse_args()
    
    if args.render_only:
        # Direct render mode (for testing)
        render_policy(
            task_id=args.task,
            policy_url=args.policy_url,
            scores=json.loads(args.scores) if args.scores else [0.1] * 1000,
            github_token=args.token or os.environ.get("GITHUB_TOKEN", ""),
            pr_number=args.pr or 0,
        )
    else:
        # HTTP server mode
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(content_length))
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                try:
                    video_path = render_policy(
                        task_id=body["task_id"],
                        policy_url=body["policy_url"],
                        scores=body["scores"],
                        github_token=body.get("github_token", ""),
                        pr_number=body["pr_number"],
                    )
                    self.wfile.write(json.dumps({"status": "ok", "video": video_path}).encode())
                except Exception as e:
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
            
            def log_message(self, format, *args):
                print(f"[render] {args[0]}")
        
        server = HTTPServer(("0.0.0.0", args.port), Handler)
        print(f"Render server on port {args.port}")
        server.serve_forever()
