"""Domain randomisation events for the skating environment.

Custom events added on top of the standard mdp events:
  - reset_skating_pose: resets robot to a skating stance with initial forward velocity.

Note:
  randomize_skate_friction is NOT a custom wrapper function.  Instead, the
  skating_env_cfg.py uses the standard mdp.randomize_rigid_body_material
  directly (a ManagerTermBase class) with skating-appropriate friction ranges.
  This file only provides reset_skating_pose which genuinely needs custom logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_skating_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    pose_range: dict | None = None,
    init_velocity_range: dict | None = None,
) -> None:
    """Reset robot to a stable skating stance with optional forward momentum.

    During Phase 1 (gliding) training, the robot is initialised with a small
    forward velocity so it immediately needs to balance rather than stand still.

    Args:
        env_ids:             Indices of environments to reset.
        pose_range:          Dict with keys x, y, yaw each a (min, max) tuple.
        init_velocity_range: Dict with key x: (min_vx, max_vx) for initial
                             forward velocity injection.
    """
    if pose_range is None:
        pose_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)}
    if init_velocity_range is None:
        init_velocity_range = {"x": (0.3, 0.8)}

    asset: Articulation = env.scene["robot"]
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, torch.Tensor) else env_ids
    n = len(env_ids_tensor)

    # Start from the articulation's default root state (set by ArticulationCfg.init_state)
    new_state = asset.data.default_root_state[env_ids_tensor].clone()

    # ── Position jitter ───────────────────────────────────────────────────────
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    new_state[:, 0] += torch.zeros(n, device=env.device).uniform_(*x_range)
    new_state[:, 1] += torch.zeros(n, device=env.device).uniform_(*y_range)

    # ── Yaw randomisation ────────────────────────────────────────────────────
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    dyaw = torch.zeros(n, device=env.device).uniform_(*yaw_range)
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    dq = quat_from_euler_xyz(
        torch.zeros_like(dyaw), torch.zeros_like(dyaw), dyaw
    )
    new_state[:, 3:7] = quat_mul(new_state[:, 3:7], dq)

    # ── Initial forward velocity ─────────────────────────────────────────────
    # Velocity must be in world frame, but vx_range is defined in the robot's
    # body forward direction.  Rotate by the applied yaw so the robot always
    # starts moving in the direction it is facing, regardless of spawn yaw.
    vx_range = init_velocity_range.get("x", (0.0, 0.0))
    vx_body = torch.zeros(n, device=env.device).uniform_(*vx_range)
    new_state[:, 7] = vx_body * torch.cos(dyaw)   # world lin_vel_x
    new_state[:, 8] = vx_body * torch.sin(dyaw)   # world lin_vel_y

    asset.write_root_state_to_sim(new_state, env_ids=env_ids_tensor)
