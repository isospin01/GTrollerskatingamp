"""AMP (Adversarial Motion Prior) observation computation for roller-skating.

The AMP state vector is a compact kinematic descriptor used by the discriminator.
It captures the robot's pose and motion in a discriminator-friendly representation.

AMP state per timestep (85-dim for G1-29DOF):
  - Joint positions (29)
  - Joint velocities (29) × 0.05
  - Projected gravity (3)
  - Base linear velocity (3) × 0.1
  - Base angular velocity (3) × 0.2
  - Left+Right ankle positions relative to base (6)   [2 × xyz]
  - Left+Right wrist positions relative to base (6)   [2 × xyz]
  - Left+Right ankle velocities (6) × 0.1             [2 × xyz]
  Total: 29+29+3+3+3+6+6+6 = 85 dims

The reference motion NPZ (gen_skating_reference.py) must produce the same
85-dim vectors in exactly the same order for the discriminator to work.

The AMP obs is injected into env extras under the key "amp_obs" by
registering it as an ObsGroup with ``enable_corruption=False`` and then
forwarding it to the extras dict via a custom env post-step hook.

For simplicity with the manager-based env, we implement it as a dedicated
observation group that the AmpOnPolicyRunner reads from the extras dict.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def amp_observation_state(
    env: ManagerBasedRLEnv,
    ankle_body_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compact AMP state vector for the G1 humanoid skating task.

    Returns:
        Tensor (num_envs, amp_obs_dim) containing the concatenated AMP state.
        This is registered as an ObsGroup "amp" which the Isaac Lab manager
        places in extras["observations"]["amp"].  The AmpOnPolicyRunner then
        reads it from ``infos["observations"]["amp"]`` and passes it to the
        discriminator as ``infos["amp_obs"]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos                    # (N, 29)
    joint_vel = asset.data.joint_vel * 0.05             # (N, 29)
    proj_grav = asset.data.projected_gravity_b          # (N, 3)
    lin_vel_b = asset.data.root_lin_vel_b * 0.1         # (N, 3)
    ang_vel_b = asset.data.root_ang_vel_b * 0.2         # (N, 3)

    # Ankle positions relative to root (in world frame, re-centred)
    ankle_pos_w = asset.data.body_pos_w[:, ankle_body_cfg.body_ids, :]   # (N, 2, 3)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)                       # (N, 1, 3)
    ankle_rel = (ankle_pos_w - root_pos_w).view(env.num_envs, -1)        # (N, 6)

    # Wrist positions relative to root
    wrist_pos_w = asset.data.body_pos_w[:, wrist_body_cfg.body_ids, :]   # (N, 2, 3)
    wrist_rel = (wrist_pos_w - root_pos_w).view(env.num_envs, -1)        # (N, 6)

    # Ankle linear velocities (world frame, scaled)
    ankle_vel_w = asset.data.body_lin_vel_w[:, ankle_body_cfg.body_ids, :] * 0.1  # (N, 2, 3)
    ankle_vel = ankle_vel_w.view(env.num_envs, -1)                                 # (N, 6)

    return torch.cat(
        [joint_pos, joint_vel, proj_grav, lin_vel_b, ang_vel_b,
         ankle_rel, wrist_rel, ankle_vel],
        dim=-1,
    )  # (N, 85)  — 29+29+3+3+3+6+6+6
