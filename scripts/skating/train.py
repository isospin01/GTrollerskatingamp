"""Script to train roller-skating RL agent with AMP + RSL-RL PPO.

This script is nearly identical to scripts/rsl_rl/train.py but uses the
AmpOnPolicyRunner instead of the standard OnPolicyRunner so that the AMP
discriminator is properly initialised and updated during training.

Usage (Phase 1 — gliding balance):
    conda activate env_isaaclab
    cd /home/muchenxu/unitree_rl_lab
    python scripts/skating/train.py \\
        --task Unitree-G1-Skating-Phase1-v0 \\
        --num_envs 4096 --headless \\
        --logger wandb --log_project_name g1_skating \\
        --experiment_name skating_phase1

Usage (Phase 2 — push-off, resume from Phase 1):
    python scripts/skating/train.py \\
        --task Unitree-G1-Skating-Phase2-v0 \\
        --num_envs 4096 --headless \\
        --resume --load_run skating_phase1 \\
        --logger wandb --log_project_name g1_skating \\
        --experiment_name skating_phase2

Usage (Phase 3 — full velocity + turning, resume from Phase 2):
    python scripts/skating/train.py \\
        --task Unitree-G1-Skating-Phase3-v0 \\
        --num_envs 4096 --headless \\
        --resume --load_run skating_phase2 \\
        --logger wandb --log_project_name g1_skating \\
        --experiment_name skating_phase3
"""

import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import argparse
import argcomplete
from isaaclab.app import AppLauncher

# local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)

_VALID_TASKS = [
    "Unitree-G1-Skating-Phase1-v0",
    "Unitree-G1-Skating-Phase2-v0",
    "Unitree-G1-Skating-Phase3-v0",
    "Unitree-G1-Skating-Phase3b-v0",
    "Unitree-G1-Skating-Phase4-v0",
]

parser = argparse.ArgumentParser(description="Train roller-skating RL agent (AMP + PPO).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None, choices=_VALID_TASKS)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument(
    "--load_experiment",
    type=str,
    default=None,
    help="Override the experiment name used to locate the checkpoint for --resume. "
         "Use this to load weights from a *different* experiment (e.g. skating_phase1 → phase2).",
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

"""Rest everything follows (after Isaac Sim is up)."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

# Import AMP runner instead of standard OnPolicyRunner
from unitree_rl_lab.tasks.skating.agents.amp_runner import AmpOnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with AMP + PPO."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        # Allow loading from a different experiment's folder via --load_experiment
        load_exp = args_cli.load_experiment if args_cli.load_experiment else agent_cfg.experiment_name
        load_root = os.path.abspath(os.path.join("logs", "rsl_rl", load_exp))
        resume_path = get_checkpoint_path(load_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Use AmpOnPolicyRunner instead of the standard OnPolicyRunner ──────────
    train_dict = agent_cfg.to_dict()
    # Inject amp config from the agent cfg dataclass (if present)
    if hasattr(agent_cfg, "amp") and agent_cfg.amp is not None:
        train_dict["amp"] = agent_cfg.amp

    runner = AmpOnPolicyRunner(env, train_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
