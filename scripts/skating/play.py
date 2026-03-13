"""Evaluation / visualisation script for trained skating policy.

Loads a trained checkpoint and runs rollouts, optionally recording video
and computing quantitative evaluation metrics:
  - Mean forward skating speed (m/s)
  - Fall rate (% of episodes ending with base height below threshold)
  - Mean episode length (s)
  - AMP style score (discriminator logit on rollout obs, if Phase 2+)
  - Lateral deviation (mean |v_y| during glide phases)

Usage:
    conda activate env_isaaclab
    cd /home/muchenxu/unitree_rl_lab

    # Visual rollout (16 envs, video recording)
    python scripts/skating/play.py \\
        --task Unitree-G1-Skating-Phase3-v0 \\
        --num_envs 16 \\
        --load_run skating_phase3 \\
        --video --video_length 500

    # Headless quantitative evaluation (512 envs)
    python scripts/skating/play.py \\
        --task Unitree-G1-Skating-Phase3-v0 \\
        --num_envs 512 --headless \\
        --load_run skating_phase3 \\
        --eval_episodes 200

    # Evaluate a specific checkpoint
    python scripts/skating/play.py \\
        --task Unitree-G1-Skating-Phase3-v0 \\
        --num_envs 64 --headless \\
        --load_run skating_phase3 \\
        --load_checkpoint model_5000.pt
"""

import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import argparse
import argcomplete
from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)

parser = argparse.ArgumentParser(description="Evaluate roller-skating policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--video_folder", type=str, default="videos/eval")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Unitree-G1-Skating-Eureka-v0",
                    choices=[
                        "Unitree-G1-Skating-Phase1-v0",
                        "Unitree-G1-Skating-Eureka-v0",
                        "Unitree-G1-Skating-AMP-v0",
                    ])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_episodes", type=int, default=100,
                    help="Number of complete episodes to evaluate.")
parser.add_argument(
    "--load_experiment",
    type=str,
    default=None,
    help="Override the experiment name used to locate the checkpoint (e.g. skating_phase3).",
)
parser.add_argument(
    "--init_vel_x",
    type=float,
    default=None,
    help="Override reset init velocity x range to (v, v+0.5) m/s for a more dynamic eval start.",
)
parser.add_argument(
    "--fixed_cmd_vel_x",
    type=float,
    default=None,
    help="Fix the forward velocity command to this value (m/s) for the entire eval.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Post-launch imports."""

import os
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run evaluation rollout and compute skating metrics."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # Allow loading from a different experiment via --load_experiment
    load_exp = args_cli.load_experiment if args_cli.load_experiment else agent_cfg.experiment_name
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", load_exp))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Override reset init velocity for a more dynamic eval start
    if args_cli.init_vel_x is not None:
        v = args_cli.init_vel_x
        env_cfg.events.reset_base.params["init_velocity_range"]["x"] = (v, v + 0.5)
        print(f"[INFO] Overriding init velocity x to ({v:.1f}, {v+0.5:.1f}) m/s")

    # Fix forward command velocity for the entire eval
    if args_cli.fixed_cmd_vel_x is not None:
        v = args_cli.fixed_cmd_vel_x
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        # Also override limit_ranges if it exists (UniformLevelVelocityCommandCfg)
        if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
            env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (v, v)
        print(f"[INFO] Fixing forward command velocity to {v:.1f} m/s")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.join(log_root_path, args_cli.video_folder)
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: ep == 0,
            video_length=args_cli.video_length,
            name_prefix=f"skating_eval_{datetime.now().strftime('%H%M%S')}",
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # ── Metrics tracking ──────────────────────────────────────────────────────
    metrics = defaultdict(list)
    episodes_done = 0
    num_envs = args_cli.num_envs
    target_episodes = args_cli.eval_episodes

    episode_speeds: list[list[float]] = [[] for _ in range(num_envs)]
    episode_lateral: list[list[float]] = [[] for _ in range(num_envs)]
    episode_falls = [False] * num_envs
    episode_lengths = [0] * num_envs

    obs, extras = env.get_observations()
    obs = obs.to(agent_cfg.device)

    print(f"\n[Eval] Running {target_episodes} episodes with {num_envs} envs...")
    print("-" * 60)

    with torch.inference_mode():
        while episodes_done < target_episodes:
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            obs = obs.to(agent_cfg.device)

            # Read kinematics from the env wrapper
            robot = env.unwrapped.scene["robot"]
            forward_vel = robot.data.root_lin_vel_b[:, 0].cpu().numpy()   # (N,)
            lateral_vel = robot.data.root_lin_vel_b[:, 1].abs().cpu().numpy()  # (N,)
            base_height  = robot.data.root_pos_w[:, 2].cpu().numpy()       # (N,)

            for i in range(num_envs):
                episode_speeds[i].append(float(forward_vel[i]))
                episode_lateral[i].append(float(lateral_vel[i]))
                episode_lengths[i] += 1
                if base_height[i] < 0.45:
                    episode_falls[i] = True

                if dones[i]:
                    # Episode complete — record metrics
                    metrics["forward_speed"].append(float(np.mean(episode_speeds[i])))
                    metrics["lateral_drift"].append(float(np.mean(episode_lateral[i])))
                    metrics["episode_length"].append(float(episode_lengths[i]) * env.unwrapped.step_dt)
                    metrics["fall"].append(float(episode_falls[i]))
                    # Reset buffers
                    episode_speeds[i] = []
                    episode_lateral[i] = []
                    episode_falls[i] = False
                    episode_lengths[i] = 0
                    episodes_done += 1

                    if episodes_done % 20 == 0:
                        print(f"  Episodes {episodes_done}/{target_episodes}  "
                              f"  speed={np.mean(metrics['forward_speed']):.2f}m/s  "
                              f"  fall%={100*np.mean(metrics['fall']):.1f}%")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Task:                 {args_cli.task}")
    print(f"  Checkpoint:           {resume_path}")
    print(f"  Episodes evaluated:   {episodes_done}")
    print(f"  Mean forward speed:   {np.mean(metrics['forward_speed']):.3f} ± {np.std(metrics['forward_speed']):.3f} m/s")
    print(f"  Mean lateral drift:   {np.mean(metrics['lateral_drift']):.3f} ± {np.std(metrics['lateral_drift']):.3f} m/s")
    print(f"  Mean episode length:  {np.mean(metrics['episode_length']):.2f} ± {np.std(metrics['episode_length']):.2f} s")
    print(f"  Fall rate:            {100 * np.mean(metrics['fall']):.1f}%")
    print("=" * 60)

    # Save metrics to CSV
    import csv
    metrics_path = os.path.join(log_root_path, "eval_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        for i in range(len(metrics["forward_speed"])):
            writer.writerow({k: metrics[k][i] for k in metrics})
    print(f"\nMetrics saved to: {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
