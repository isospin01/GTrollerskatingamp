"""Eureka-driven roller-skating environment for Unitree G1-29DOF.

This environment is the target for NVIDIA Eureka (LLM-driven reward search,
scripts/skating/eureka_phase2.py).  It replaces the hand-designed Phases 2–4
with a single environment whose task-specific reward is proposed iteratively
by an LLM and injected at runtime via the EUREKA_REWARD_FN_PATH env var.

Training flow
-------------
  Phase 1  ──→  Eureka loop (Phase 2+)
  (glide)        LLM proposes compute_reward()
                 Train 1 000–2 000 iters, resume from Phase 1 ckpt
                 Evaluate: forward speed, fall rate, episode length
                 Feed metrics back → LLM refines reward
                 Repeat K times → keep best policy

Design principles
-----------------
* Fixed structural rewards (safety, smoothness, posture) live here.
* All task-specific signals (speed, push-off, turning) come from the
  LLM-generated compute_reward() via ``eureka_task_reward``.
* Commands cover the full skating space (forward + turning); the skating
  curriculum advances the ranges as performance improves.
* Episodes start from near-rest so the robot must actively push off —
  this is the key new skill vs. Phase 1's pre-injected glide velocity.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# ── Shared components from Phase 1 (scene, observations, actions, terminations)
from .skating_env_cfg import (
    ActionsCfg,
    ObservationsCfg,
    SkatingSceneCfg,
    TerminationsCfg,
)
from unitree_rl_lab.tasks.skating import mdp


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class EurekaCommandsCfg:
    """Full skating command space: forward + turning, expanded by curriculum."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,   # almost always moving
        rel_heading_envs=1.0,     # heading-aligned commands
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Conservative start — curriculum drives forward
            lin_vel_x=(0.3, 1.5),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.4, 0.4),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Curriculum ceiling: full skating speed + turning
            lin_vel_x=(0.3, 3.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rewards
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class EurekaRewardsCfg:
    """Rewards for the Eureka-guided skating environment.

    Fixed structural terms keep the robot safe and physically plausible.
    All task-specific signals (speed, push-off, turning, style) are delegated
    to ``eureka_task_reward``, which calls the LLM-generated compute_reward().
    """

    # ── Survival ─────────────────────────────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # ── Posture / balance ─────────────────────────────────────────────────────
    upright = RewTerm(func=mdp.upright_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_skating_l2,
        weight=-10.0,
        # 0.85 m: slightly below full upright to encourage bent-knee posture.
        params={"target_height": 0.85},
    )
    lateral_vel = RewTerm(func=mdp.lateral_velocity_penalty, weight=-1.0)

    # ── Efficiency / smoothness ───────────────────────────────────────────────
    energy     = RewTerm(func=mdp.skate_energy,      weight=-2e-5)
    action_rate = RewTerm(func=mdp.skate_action_rate, weight=-0.05)
    joint_acc  = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # ── Posture naturalness ───────────────────────────────────────────────────
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    # ── Contact safety ────────────────────────────────────────────────────────
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*).*"],
            ),
        },
    )

    # ── Eureka LLM task reward ─────────────────────────────────────────────────
    # weight=1.0 because the generated function is expected to scale its own outputs.
    # The term returns zeros if EUREKA_REWARD_FN_PATH is not set, so baseline
    # training with only the structural terms is also possible.
    eureka_task = RewTerm(func=mdp.eureka_task_reward, weight=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class EurekaEventCfg:
    """Domain randomisation for the Eureka phase.

    Key difference from Phase 1: episodes reset from near-rest (0–0.5 m/s)
    instead of with a pre-injected glide velocity.  The robot must use the
    push-off skill it learns to build up speed.
    """

    attach_skates = EventTerm(
        func=mdp.attach_skates_to_robot,
        mode="startup",
    )

    skate_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range":  (0.01, 0.03),
            "dynamic_friction_range": (0.005, 0.02),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    # Near-rest reset: robot must push off to earn task reward
    reset_base = EventTerm(
        func=mdp.reset_skating_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-3.14, 3.14)},
            "init_velocity_range": {"x": (0.0, 0.5)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class EurekaCurriculumCfg:
    """Advance speed and turning commands as Eureka task reward improves."""

    lin_vel_cmd_levels = CurrTerm(
        mdp.skating_lin_vel_cmd_levels,
        params={"reward_term_name": "eureka_task"},
    )
    ang_vel_cmd_levels = CurrTerm(
        mdp.skating_ang_vel_cmd_levels,
        params={"reward_term_name": "eureka_task"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class G1SkatingEurekaEnvCfg(ManagerBasedRLEnvCfg):
    """Eureka-driven full skating environment.

    Structural/safety rewards are fixed here.
    Task-specific reward is injected per Eureka iteration via EUREKA_REWARD_FN_PATH.
    """

    scene: SkatingSceneCfg = SkatingSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: EurekaCommandsCfg = EurekaCommandsCfg()
    rewards: EurekaRewardsCfg = EurekaRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EurekaEventCfg = EurekaEventCfg()
    curriculum: EurekaCurriculumCfg = EurekaCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4           # 50 Hz policy
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1SkatingEurekaPlayEnvCfg(G1SkatingEurekaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
