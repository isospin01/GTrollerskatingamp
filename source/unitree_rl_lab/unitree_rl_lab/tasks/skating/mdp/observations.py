"""Observation terms specific to the roller-skating task.

These are registered in ObservationsCfg as additional ObsTerm entries on top
of the standard locomotion observations (joint_pos_rel, joint_vel_rel, etc.).
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


def skating_phase_signal(
    env: ManagerBasedRLEnv,
    period: float = 1.2,
) -> torch.Tensor:
    """Return (sin, cos) of the current skating gait phase.

    Provides the policy with a periodic clock signal so it can produce
    phase-coherent push-off actions.

    Returns:
        Tensor of shape (num_envs, 2): [sin(phase), cos(phase)]
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * 2.0 * math.pi)
    phase[:, 1] = torch.cos(global_phase * 2.0 * math.pi)
    return phase


def foot_contact_forces_normalized(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 500.0,
) -> torch.Tensor:
    """Normalised vertical contact forces at the ankle roll links.

    Gives the policy proprioceptive feedback about which foot is loaded,
    which is crucial for push-off timing.

    Returns:
        Tensor of shape (num_envs, num_feet): normalised force in [0, 1]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces_z / max_force).clamp(0.0, 1.0)


def base_lin_vel_forward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward (X) linear velocity of the base in the body frame.

    Provides the actor with explicit speed feedback, complementing the
    privileged critic observation that includes the full 3-DoF base velocity.

    Returns:
        Tensor of shape (num_envs, 1)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0:1]
