"""Roller-skating environment configurations for Unitree G1-29DOF.

Three phase-progressive environments are defined following the HUSKY
phase-wise training strategy (arXiv:2602.03205):

  Phase 1 — G1SkatingGlideEnvCfg:
    Both feet on skates, no push-off. Train balance and passive gliding.
    Velocity command is small (0.0–0.5 m/s) with no angular velocity.

  Phase 2 — G1SkatingPushEnvCfg:
    Add alternating push-off (gait rhythm reward). AMP reference motion
    from gen_skating_reference.py (push-glide NPZ) is introduced.
    Velocity commands extended to 0–1.5 m/s.

  Phase 3 — G1SkatingFullEnvCfg:
    Full velocity + turning commands enabled.
    Lean-to-steer mechanism: ankle roll commands induce edge pressure.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_SKATE_CFG as _BASE_ROBOT_CFG
from unitree_rl_lab.assets.scenes import SKATING_RINK_USD
import isaaclab.sim as _sim_utils

# ── USD path resolution ───────────────────────────────────────────────────────
# Priority: 1) UNITREE_MODEL_DIR env var  2) known local path  3) Nucleus CDN
_G1_LOCAL_CANDIDATES = [
    # Local HDMI project assets (no-hand variant for clean foot contacts)
    "/home/muchenxu/HDMI/active_adaptation/assets/g1/g1_29dof_nohand-feet_sphere.usd",
    # IsaacLab Nucleus (requires Nucleus server access)
    "${ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
]

def _resolve_g1_usd() -> str:
    import os
    # Check user-set env var first (unitree_rl_lab convention)
    model_dir = os.environ.get("UNITREE_MODEL_DIR", "")
    if model_dir and model_dir != "path/to/unitree_model":
        candidate = os.path.join(model_dir, "G1", "29dof", "usd",
                                 "g1_29dof_rev_1_0", "g1_29dof_rev_1_0.usd")
        if os.path.isfile(candidate):
            return candidate
    # Fall back to known local paths
    for c in _G1_LOCAL_CANDIDATES:
        expanded = os.path.expandvars(c)
        if os.path.isfile(expanded):
            return expanded
    # Last resort: Nucleus (may download on first run)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    return f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"

_G1_USD_PATH = _resolve_g1_usd()

ROBOT_CFG = _BASE_ROBOT_CFG.replace(
    spawn=_sim_utils.UsdFileCfg(
        usd_path=_G1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=_sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )
)
from unitree_rl_lab.tasks.skating import mdp

# ─── Scene ───────────────────────────────────────────────────────────────────

@configclass
class SkatingSceneCfg(InteractiveSceneCfg):
    """Flat rink with low-friction surface and G1 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",          # flat plane; USD rink is overlaid as visual
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # Rink floor: low friction to simulate polished sports-hall surface
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.05,
            dynamic_friction=0.02,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # White/light-grey floor visual
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                     "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.5, 0.5),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=5.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                         "kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ─── MDP components shared across all phases ─────────────────────────────────

@configclass
class ActionsCfg:
    """29-DOF joint position targets, same as standard locomotion."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations for actor (policy) and critic networks.

    Policy (actor) — noise-corrupted, no privileged info:
        ~89 dims total:
            projected_gravity (3)
            base_ang_vel (3)
            velocity_commands (3)
            joint_pos_rel (29)
            joint_vel_rel (29)
            last_action (29)
            skating_phase_signal (2)
            foot_contact_forces_normalized (2)
            base_lin_vel_forward (1)   ← only forward speed, not full vel

    Critic — privileged (no noise, includes full base_lin_vel):
        +3 (full base_lin_vel) over actor
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — noise-corrupted."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)
        skating_phase = ObsTerm(
            func=mdp.skating_phase_signal,
            params={"period": 1.2},
        )
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_forces_normalized,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*ankle_roll.*"],
                ),
                "max_force": 500.0,
            },
        )
        forward_speed = ObsTerm(
            func=mdp.base_lin_vel_forward,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — privileged, no noise."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        skating_phase = ObsTerm(
            func=mdp.skating_phase_signal,
            params={"period": 1.2},
        )
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_forces_normalized,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*ankle_roll.*"],
                ),
                "max_force": 500.0,
            },
        )

        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        """AMP state vector (no noise, no history) — passed to discriminator.

        The AmpOnPolicyRunner reads this from infos["observations"]["amp"]
        and routes it to the discriminator as infos["amp_obs"].
        """

        amp_state = ObsTerm(
            func=mdp.amp_observation_state,
            params={
                "ankle_body_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "wrist_body_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class TerminationsCfg:
    """Shared termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Fallen: base too close to floor (G1 root ≈ 0.84 m above floor when skating)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45},
    )

    # Badly tilted: roll or pitch exceeds ~46 degrees
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Gliding Balance
# Goal: learn to stand / glide in a skating posture on the low-friction rink.
# No push-off, no angular commands. Small forward velocity command (0–0.5 m/s).
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Phase1CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.02,   # almost no standing — always moving
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5),  # start with meaningful glide speed
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 2.0),  # curriculum can push to 2 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class Phase1RewardsCfg:
    """Phase 1: balance + glide at meaningful speed. No standing reward."""

    # ── Task (positive) ──────────────────────────────────────────────────────
    # Strong speed tracking — being slow is heavily penalised implicitly
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_tracking_exp,
        weight=5.0,                          # increased from 3.0
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # Reward sustained glide speed — must actually be skating, not creeping
    glide_continuity = RewTerm(
        func=mdp.skate_glide_continuity,
        weight=2.0,                          # increased from 1.0
        params={"command_name": "base_velocity", "min_speed": 0.8},  # 0.8 m/s minimum
    )

    # ── Posture penalties ────────────────────────────────────────────────────
    upright = RewTerm(func=mdp.upright_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_skating_l2,
        weight=-10.0,
        # Target height accounts for wheel geometry (0.80m baseline + 0.076m wheel offset).
        # 0.85m is slightly below full upright (0.876m) to encourage a bent-knee posture.
        params={"target_height": 0.85},
    )
    lateral_vel = RewTerm(func=mdp.lateral_velocity_penalty, weight=-2.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ── Joint penalties ──────────────────────────────────────────────────────
    energy = RewTerm(func=mdp.skate_energy, weight=-2e-5)
    action_rate = RewTerm(func=mdp.skate_action_rate, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

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

    # ── Contact penalties ────────────────────────────────────────────────────
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


@configclass
class Phase1EventCfg:
    """Domain randomisation for Phase 1."""

    # Attach visual skate geometry (boot + wheels) to ankle links
    attach_skates = EventTerm(
        func=mdp.attach_skates_to_robot,
        mode="startup",
    )

    # Apply low skate wheel friction at startup (simulates polished wheels on floor)
    skate_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.01, 0.03),  # lower than before
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

    # Reset with meaningful initial glide velocity
    reset_base = EventTerm(
        func=mdp.reset_skating_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "init_velocity_range": {"x": (0.5, 1.5)},  # start with actual glide speed
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # External pushes to test balance
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}},
    )


@configclass
class G1SkatingGlideEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 1: Gliding balance on roller skates."""

    scene: SkatingSceneCfg = SkatingSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: Phase1CommandsCfg = Phase1CommandsCfg()
    rewards: Phase1RewardsCfg = Phase1RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Phase1EventCfg = Phase1EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4          # 50 Hz policy
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1SkatingGlidePlayEnvCfg(G1SkatingGlideEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Push-Off Learning
# Goal: learn alternating push-off gait to accelerate from rest.
# AMP reference motion (gen_skating_reference NPZ) activates here.
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Phase2CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,   # almost never standing
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 2.0),   # push-off phase: clearly moving
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 3.0),   # curriculum can reach 3 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class Phase2RewardsCfg(Phase1RewardsCfg):
    """Phase 2: adds push-off rhythm reward on top of Phase 1 rewards."""

    push_off = RewTerm(
        func=mdp.push_off_rhythm,
        weight=3.0,                          # increased from 1.5 — gait is primary signal
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"]),
            "command_name": "base_velocity",
            "period": 1.2,                   # must match skating_phase_signal period in ObservationsCfg
            "offset": [0.0, 0.5],
            "threshold": 0.5,
        },
    )
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_tracking_exp,
        weight=6.0,                          # increased from 4.0
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Lower min_speed threshold: Phase 2 resets from rest (0–0.5 m/s) so the
    # robot must push off to earn this reward.  0.3 m/s is achievable within
    # the first push cycle, giving a reward signal throughout the episode.
    glide_continuity = RewTerm(
        func=mdp.skate_glide_continuity,
        weight=2.0,
        params={"command_name": "base_velocity", "min_speed": 0.3},
    )


@configclass
class Phase2EventCfg(Phase1EventCfg):
    """Phase 2 events: slightly more aggressive pushes."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}},
    )

    reset_base = EventTerm(
        func=mdp.reset_skating_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-3.14, 3.14)},
            # Start from near-rest so the push-off reward is immediately relevant.
            # 0.0–0.3 m/s means the robot must actively push to satisfy the
            # glide_continuity (min_speed=0.3) and forward_velocity rewards.
            "init_velocity_range": {"x": (0.0, 0.3)},
        },
    )


@configclass
class G1SkatingPushEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 2: Push-off with alternating feet."""

    scene: SkatingSceneCfg = SkatingSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: Phase2CommandsCfg = Phase2CommandsCfg()
    rewards: Phase2RewardsCfg = Phase2RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Phase2EventCfg = Phase2EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1SkatingPushPlayEnvCfg(G1SkatingPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Full Velocity + Turning
# Goal: follow arbitrary velocity + angular velocity commands using
# lean-to-steer edge control.
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Phase3CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # Start modest — robot pushes from rest, curriculum raises the bar
            lin_vel_x=(0.3, 1.2),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 3.5),    # curriculum can push to 3.5 m/s
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


@configclass
class Phase3RewardsCfg(Phase2RewardsCfg):
    """Phase 3: adds angular velocity tracking and edge contact rewards."""

    # Stronger forward speed signal — drives curriculum to advance past 2 m/s
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_tracking_exp,
        weight=6.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Low min_speed so the robot earns glide reward even while accelerating from rest
    glide_continuity = RewTerm(
        func=mdp.skate_glide_continuity,
        weight=2.0,
        params={"command_name": "base_velocity", "min_speed": 0.2},
    )

    # Stronger turning signal — 2.5 vs 1.5 makes turning a primary objective
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Doubled edge-contact reward to reinforce ankle carving during turns
    edge_contact = RewTerm(
        func=mdp.skate_foot_edge_contact,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"]),
            "command_name": "base_velocity",
            "ang_vel_threshold": 0.2,
        },
    )
    # Reduce lateral penalty slightly to allow controlled lateral movement during turns
    lateral_vel = RewTerm(func=mdp.lateral_velocity_penalty, weight=-1.0)


@configclass
class Phase3CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(
        mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "forward_velocity"},
    )
    ang_vel_cmd_levels = CurrTerm(
        mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "track_ang_vel"},
    )


@configclass
class Phase3EventCfg(Phase2EventCfg):
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class G1SkatingFullEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 3: Full velocity + turning commands."""

    scene: SkatingSceneCfg = SkatingSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: Phase3CommandsCfg = Phase3CommandsCfg()
    rewards: Phase3RewardsCfg = Phase3RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Phase3EventCfg = Phase3EventCfg()
    curriculum: Phase3CurriculumCfg = Phase3CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1SkatingFullPlayEnvCfg(G1SkatingFullEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3b: Curriculum Fix — Lower Threshold + Warm Start
# Warm-starts the command range at (0.3, 1.5) since the robot already
# handles 1.2 m/s, and uses skating-specific curriculum functions with
# a 0.3 threshold (achievable) and 0.2 step size (faster progression).
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Phase3bCommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 1.5),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.4, 0.4),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 3.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


@configclass
class Phase3bCurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(
        mdp.skating_lin_vel_cmd_levels,
        params={"reward_term_name": "forward_velocity"},
    )
    ang_vel_cmd_levels = CurrTerm(
        mdp.skating_ang_vel_cmd_levels,
        params={"reward_term_name": "track_ang_vel"},
    )


@configclass
class Phase3bRewardsCfg(Phase3RewardsCfg):
    action_rate = RewTerm(func=mdp.skate_action_rate, weight=-0.03)


@configclass
class G1SkatingPhase3bEnvCfg(G1SkatingFullEnvCfg):
    """Phase 3b: lower curriculum threshold + warm-started command range."""

    commands: Phase3bCommandsCfg = Phase3bCommandsCfg()
    rewards: Phase3bRewardsCfg = Phase3bRewardsCfg()
    curriculum: Phase3bCurriculumCfg = Phase3bCurriculumCfg()


@configclass
class G1SkatingPhase3bPlayEnvCfg(G1SkatingPhase3bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Straight-Line Skating Mastery
# Focus purely on forward speed — no turning commands, no turning rewards.
# All reward budget redirected to forward velocity and posture.
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Phase4CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 2.3),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 3.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class Phase4RewardsCfg(Phase2RewardsCfg):
    """Straight-line skating: boost forward speed, remove turning rewards."""

    forward_velocity = RewTerm(
        func=mdp.forward_velocity_tracking_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    glide_continuity = RewTerm(
        func=mdp.skate_glide_continuity,
        weight=2.0,
        params={"command_name": "base_velocity", "min_speed": 0.2},
    )
    lateral_vel = RewTerm(func=mdp.lateral_velocity_penalty, weight=-1.0)
    yaw_rate = RewTerm(func=mdp.ang_vel_z_l2, weight=-1.0)
    action_rate = RewTerm(func=mdp.skate_action_rate, weight=-0.03)


@configclass
class Phase4CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(
        mdp.skating_lin_vel_cmd_levels,
        params={"reward_term_name": "forward_velocity"},
    )


@configclass
class G1SkatingPhase4EnvCfg(G1SkatingFullEnvCfg):
    """Phase 4: straight-line skating mastery — no turning."""

    commands: Phase4CommandsCfg = Phase4CommandsCfg()
    rewards: Phase4RewardsCfg = Phase4RewardsCfg()
    curriculum: Phase4CurriculumCfg = Phase4CurriculumCfg()


@configclass
class G1SkatingPhase4PlayEnvCfg(G1SkatingPhase4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
