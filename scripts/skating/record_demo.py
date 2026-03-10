"""Record demo footage of the trained skating policy.

Presets:
  --clip straight   → vx=0.8, yaw=0.0
  --clip turning    → vx=0.5, yaw=0.4

Or supply custom commands directly:
  --vx 2.0 --wz 0.0

Env:  1 env, configurable duration, flat terrain, fall termination disabled.
Cam:  Side-tracking at world-frame offset [0, -4, 1.2], 60° FOV,
      follows robot position but does NOT rotate with yaw.

Usage:
    cd /home/muchenxu/unitree_rl_lab

    # Custom from-rest forward demo
    conda run -n env_isaaclab python scripts/skating/record_demo.py \
        --vx 2.0 --wz 0.0 --duration 40 --name demo_forward --headless
"""

import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demo footage of skating policy.")
parser.add_argument("--clip", choices=["straight", "turning"], default=None,
                    help="Preset clip. Overridden by --vx/--wz if provided.")
parser.add_argument("--vx", type=float, default=None, help="Forward velocity command (m/s).")
parser.add_argument("--wz", type=float, default=None, help="Yaw rate command (rad/s).")
parser.add_argument("--duration", type=float, default=65.0, help="Episode length in seconds.")
parser.add_argument("--name", type=str, default=None, help="Output filename stem (no extension).")
parser.add_argument("--output_dir", type=str, default="/home/muchenxu/rollerskating")
parser.add_argument(
    "--checkpoint", type=str, default="model_8000.pt",
    help="Checkpoint filename inside the latest run dir.",
)
parser.add_argument(
    "--log_dir", type=str, default="skating_phase3",
    help="Experiment folder name under logs/rsl_rl/ (e.g. skating_phase3b).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, extra = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + extra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-launch imports (Isaac Sim is now initialised) ──────────────────────

import math
import os
import torch
import numpy as np
import gymnasium as gym
import imageio

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401 — registers gym envs

# ─── Constants ───────────────────────────────────────────────────────────────

CLIP_CFG = {
    "straight": {"lin_vel_x": 0.8, "ang_vel_z": 0.0},
    "turning":  {"lin_vel_x": 0.5, "ang_vel_z": 0.4},
}

CAM_OFFSET = np.array([0.0, -4.0, 1.2])
CAM_TARGET_HEIGHT = 0.85
CAM_FOV_DEG = 60.0
RESOLUTION = (1920, 1080)
DEVICE = args_cli.device if args_cli.device else "cuda:0"


def set_camera_fov(fov_deg: float):
    """Set the viewport camera horizontal FOV by adjusting focal length."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if cam_prim.IsValid():
            camera = UsdGeom.Camera(cam_prim)
            h_aperture = camera.GetHorizontalApertureAttr().Get() or 20.955
            focal = h_aperture / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
            camera.GetFocalLengthAttr().Set(focal)
            print(f"[CAM] FOV set to {fov_deg}° (focal_length={focal:.2f} mm)")
    except Exception as e:
        print(f"[WARN] Could not set FOV: {e}")


def track_camera(sim, robot_pos: np.ndarray):
    """Move viewport camera to side-track the robot in world frame."""
    eye = [
        float(robot_pos[0]) + float(CAM_OFFSET[0]),
        float(robot_pos[1]) + float(CAM_OFFSET[1]),
        float(CAM_OFFSET[2]),
    ]
    target = [float(robot_pos[0]), float(robot_pos[1]), CAM_TARGET_HEIGHT]
    sim.set_camera_view(eye, target)


def resolve_commands():
    """Resolve vx/wz from --vx/--wz flags or --clip preset."""
    if args_cli.vx is not None:
        vx = args_cli.vx
        wz = args_cli.wz if args_cli.wz is not None else 0.0
    elif args_cli.clip is not None:
        vx = CLIP_CFG[args_cli.clip]["lin_vel_x"]
        wz = CLIP_CFG[args_cli.clip]["ang_vel_z"]
    else:
        parser.error("Provide either --clip or --vx.")
        return 0, 0  # unreachable
    return vx, wz


def output_stem():
    """Determine the output filename stem."""
    if args_cli.name:
        return args_cli.name
    if args_cli.clip:
        return f"skating_phase3_{args_cli.clip}_raw"
    return f"skating_vx{args_cli.vx:.1f}_wz{args_cli.wz or 0.0:.1f}"


def build_env_cfg(vx: float, wz: float, duration: float):
    """Create a Phase 3 env config with demo-specific overrides."""
    from unitree_rl_lab.tasks.skating.robots.g1_29dof.skating_env_cfg import (
        G1SkatingFullEnvCfg,
    )

    cfg = G1SkatingFullEnvCfg()

    cfg.scene.num_envs = 1
    cfg.episode_length_s = duration
    cfg.scene.env_spacing = 10.0

    # Disable fall termination (reset_on_fall=False)
    cfg.terminations.base_height.params["minimum_height"] = -1.0
    cfg.terminations.bad_orientation.params["limit_angle"] = 100.0

    # Fixed velocity commands for the entire episode
    cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = (vx, vx)
    cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.limit_ranges.ang_vel_z = (wz, wz)
    cfg.commands.base_velocity.resampling_time_range = (200.0, 200.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0

    # Start from rest (no initial velocity)
    cfg.events.reset_base.params["init_velocity_range"]["x"] = (0.0, 0.0)

    # Disable external pushes for a clean demo
    cfg.events.push_robot.interval_range_s = (9999.0, 9999.0)

    # Viewer settings
    if hasattr(cfg, "viewer"):
        if hasattr(cfg.viewer, "resolution"):
            cfg.viewer.resolution = RESOLUTION
        cfg.viewer.eye = (0.0, -4.0, 1.2)
        cfg.viewer.lookat = (0.0, 0.0, CAM_TARGET_HEIGHT)

    return cfg


def main():
    vx, wz = resolve_commands()
    env_cfg = build_env_cfg(vx, wz, args_cli.duration)

    # ── Create environment ───────────────────────────────────────────────────
    gym_env = gym.make(
        "Unitree-G1-Skating-Phase3-v0",
        cfg=env_cfg,
        render_mode="rgb_array",
    )
    base_env = gym_env.unwrapped
    sim = base_env.sim

    from unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg import (
        SkatingPhase3PPORunnerCfg,
    )
    agent_cfg = SkatingPhase3PPORunnerCfg()
    rsl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    # ── Load policy ──────────────────────────────────────────────────────────
    log_root = os.path.join(
        "/home/muchenxu/unitree_rl_lab", "logs", "rsl_rl", args_cli.log_dir,
    )
    ckpt_path = get_checkpoint_path(log_root, "2026-.*", args_cli.checkpoint)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=DEVICE)

    # ── Camera ───────────────────────────────────────────────────────────────
    set_camera_fov(CAM_FOV_DEG)

    # ── Recording parameters ─────────────────────────────────────────────────
    step_dt = env_cfg.decimation * env_cfg.sim.dt   # 4 × 0.005 = 0.02s
    total_steps = int(env_cfg.episode_length_s / step_dt)  # 3250
    native_fps = int(round(1.0 / step_dt))           # 50

    stem = output_stem()
    out_path = os.path.join(args_cli.output_dir, f"{stem}.mp4")
    os.makedirs(args_cli.output_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Name:       {stem}")
    print(f"  Commands:   lin_x={vx} m/s,  yaw={wz} rad/s")
    print(f"  Duration:   {env_cfg.episode_length_s}s  ({total_steps} steps)")
    print(f"  Native FPS: {native_fps}")
    print(f"  Resolution: {RESOLUTION[0]}×{RESOLUTION[1]}")
    print(f"  Output:     {out_path}")
    print(f"{'='*64}\n")

    writer = imageio.get_writer(
        out_path,
        fps=native_fps,
        codec="libx264",
        output_params=["-crf", "18", "-preset", "slow"],
    )

    # ── Run episode and capture frames ───────────────────────────────────────
    obs, _ = rsl_env.get_observations()
    obs = obs.to(DEVICE)
    n_frames = 0

    with torch.inference_mode():
        for step in range(total_steps):
            # Side-track camera to current robot position (world frame)
            rpos = base_env.scene["robot"].data.root_pos_w[0].cpu().numpy()
            track_camera(sim, rpos)

            # Policy forward + env step (physics + render)
            actions = policy(obs)
            obs, _, dones, _ = rsl_env.step(actions)
            obs = obs.to(DEVICE)

            # Capture rendered frame
            frame = gym_env.render()
            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 3:
                if frame.ndim == 4:
                    frame = frame[0]
                writer.append_data(frame)
                n_frames += 1

            # Progress report every ~10 s of sim time
            if step % 500 == 0:
                spd = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
                yaw = base_env.scene["robot"].data.root_ang_vel_b[0, 2].item()
                h = base_env.scene["robot"].data.root_pos_w[0, 2].item()
                print(
                    f"  [{step:5d}/{total_steps}]  "
                    f"v={spd:+5.2f} m/s  yaw={yaw:+5.2f} rad/s  "
                    f"h={h:.3f}m  pos=({rpos[0]:+.1f}, {rpos[1]:+.1f})"
                )

    writer.close()
    rsl_env.close()

    duration_s = n_frames / native_fps
    print(f"\n{'='*64}")
    print(f"  DONE  {n_frames} frames  |  {duration_s:.1f}s @ {native_fps} fps")
    print(f"  Saved: {out_path}")
    print(f"{'='*64}")

    print(f"\n# Download to laptop:")
    print(f"scp muchenxu@phe108c-yuewang-02:{out_path} .")


if __name__ == "__main__":
    main()
    simulation_app.close()
