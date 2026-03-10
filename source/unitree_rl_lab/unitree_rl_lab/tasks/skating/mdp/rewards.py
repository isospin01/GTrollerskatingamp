"""Reward functions for humanoid roller-skating.

Design philosophy (inspired by HUSKY, arXiv:2602.03205):
  - Positive rewards: forward speed tracking, staying upright, balanced height,
    alternating push-off rhythm, and AMP-style discriminator score (wired externally).
  - Negative penalties: lateral drift, energy, action rate jitter, joint limits,
    undesired body-floor contacts.

All functions follow the Isaac Lab convention:
    fn(env: ManagerBasedRLEnv, ...) -> torch.Tensor  shape=(num_envs,)
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ────────────────────────────────────────────────────────────────────────────
# Task rewards (positive)
# ────────────────────────────────────────────────────────────────────────────

def forward_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = math.sqrt(0.25),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward velocity matching the commanded speed.

    Uses the yaw-aligned body-frame velocity so the robot can rotate while
    still receiving full credit for skating in the commanded direction.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 3): vx, vy, wz

    # Project root linear velocity onto yaw-frame X axis
    lin_vel_b = asset.data.root_lin_vel_b  # body-frame velocity
    # Use only x-component (forward in body frame)
    vel_error = cmd[:, 0] - lin_vel_b[:, 0]
    return torch.exp(-vel_error.pow(2) / std**2)


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise lateral (Y) body-frame velocity — skates should not drift sideways."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 1].pow(2)


def skate_glide_continuity(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_speed: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining a minimum glide speed when commanded to move.

    Encourages the robot to use momentum rather than stopping between pushes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_active = (cmd[:, 0].abs() > 0.1)

    forward_vel = asset.data.root_lin_vel_b[:, 0]
    gliding = (forward_vel > min_speed).float()
    return gliding * cmd_active.float()


def push_off_rhythm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    period: float = 1.2,
    offset: list[float] | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward alternating foot push-off matching a phase clock.

    During Phase 2+, one foot is on the floor pushing while the other glides.
    This uses the same phase-clock mechanism as the locomotion gait reward.
    """
    if offset is None:
        offset = [0.0, 0.5]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = [(global_phase + off) % 1.0 for off in offset]
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()

    # Only active when commanded to move
    cmd_norm = env.command_manager.get_command(command_name).norm(dim=1)
    return reward * (cmd_norm > 0.1).float()


def skate_foot_edge_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    ang_vel_threshold: float = 0.2,
    ankle_roll_joint_names: tuple[str, str] = (
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward ankle roll (edge tilt) that matches the turning command direction.

    When angular velocity is commanded, the robot should lean into the turn by
    rolling the inside ankle inward.  This reward checks that:
      1. Both feet are in contact (gliding, not airborne).
      2. The ankle roll joints are actually tilted in the correct direction
         for the commanded turn (positive yaw cmd → roll toward left = negative
         left_ankle_roll, positive right_ankle_roll, and vice versa).

    The reward magnitude scales with the ankle roll angle, so a larger lean
    earns a higher score.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Foot contact check
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    has_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    both_contact = has_contact.all(dim=-1)  # (N,) bool

    # Commanded angular velocity
    cmd = env.command_manager.get_command(command_name)
    wz_cmd = cmd[:, 2]                               # (N,) positive = left turn
    turning = wz_cmd.abs() > ang_vel_threshold       # (N,) bool

    # Ankle roll joint positions — find their indices in the articulation
    all_joint_names = asset.data.joint_names
    idx_left  = all_joint_names.index(ankle_roll_joint_names[0])
    idx_right = all_joint_names.index(ankle_roll_joint_names[1])

    q_left  = asset.data.joint_pos[:, idx_left]   # (N,) positive = roll left/inward
    q_right = asset.data.joint_pos[:, idx_right]  # (N,) positive = roll right/inward

    # For a left turn (wz > 0): lean left → left ankle rolls outward (negative),
    # right ankle rolls inward (positive).
    # Sign convention: reward when sign(q_right) == sign(wz) and sign(q_left) == -sign(wz).
    edge_alignment = (q_right * wz_cmd - q_left * wz_cmd).clamp(min=0.0)

    return edge_alignment * both_contact.float() * turning.float()


def upright_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise deviation from upright posture (projected gravity XY components)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.projected_gravity_b[:, :2].pow(2), dim=1)


def base_height_skating_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise deviation from the target skating stance height.

    Target is slightly lower than standing (0.80 m vs 0.84 m init) to encourage
    a bent-knee skating posture.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - target_height).pow(2)


def skate_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise total mechanical energy (|torque| * |velocity|)."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(qvel.abs() * qfrc.abs(), dim=-1)


def skate_action_rate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalise rapid action changes to encourage smooth joint trajectories."""
    return torch.sum((env.action_manager.action - env.action_manager.prev_action).pow(2), dim=1)


def ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])
