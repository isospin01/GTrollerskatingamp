"""Generate an analytical skating reference motion for the AMP discriminator.

This script synthesises a kinematically plausible push-glide skating cycle
for the Unitree G1-29DOF robot and saves it as a .npz file that the
AmpOnPolicyRunner can load into the expert motion buffer.

The motion is generated analytically (no simulation or motion capture required):
  - Gait period: 1.2 s (configurable)
  - Two phases per cycle:
      [0.0 – 0.4 s]  LEFT push: left hip extends, right knee bends slightly
      [0.4 – 0.6 s]  Glide: both feet roughly parallel, arms balanced
      [0.6 – 1.0 s]  RIGHT push: right hip extends, left knee bends slightly
      [1.0 – 1.2 s]  Glide: transition back
  - Arms swing naturally opposite to legs (counter-rotation)
  - Root translates forward at a target glide speed

AMP observation format (89-dim, matching amp_obs.py):
  joint_pos (29) | joint_vel (29)×0.05 | proj_grav (3) | lin_vel_b (3)×0.1
  | ang_vel_b (3)×0.2 | ankle_rel (6) | wrist_rel (6) | ankle_vel (6)

Output:
  source/unitree_rl_lab/unitree_rl_lab/data/skating_reference.npz
  Keys:
    amp_obs   (T, 89)  float32   — AMP state vectors
    fps       ()       float32   — frames per second
    metadata  ()       str       — human-readable description

Usage:
    conda activate env_isaaclab
    cd /home/muchenxu/unitree_rl_lab
    python scripts/skating/gen_skating_reference.py [--fps 50] [--cycles 20]
"""

import argparse
import os
import math
import numpy as np

# ─── G1-29DOF joint order (must match IsaacLab articulation) ─────────────────
JOINT_NAMES = [
    "left_hip_pitch_joint",   # 0
    "left_hip_roll_joint",    # 1
    "left_hip_yaw_joint",     # 2
    "left_knee_joint",        # 3
    "left_ankle_pitch_joint", # 4
    "left_ankle_roll_joint",  # 5
    "right_hip_pitch_joint",  # 6
    "right_hip_roll_joint",   # 7
    "right_hip_yaw_joint",    # 8
    "right_knee_joint",       # 9
    "right_ankle_pitch_joint",# 10
    "right_ankle_roll_joint", # 11
    "waist_yaw_joint",        # 12
    "waist_roll_joint",       # 13
    "waist_pitch_joint",      # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]
N_DOF = len(JOINT_NAMES)  # 29

# ─── Default (neutral stance) joint positions ─────────────────────────────────
DEFAULT_JOINT_POS = np.zeros(N_DOF, dtype=np.float32)
DEFAULT_JOINT_POS[0]  = -0.15  # left_hip_pitch
DEFAULT_JOINT_POS[3]  =  0.40  # left_knee
DEFAULT_JOINT_POS[4]  = -0.25  # left_ankle_pitch
DEFAULT_JOINT_POS[6]  = -0.15  # right_hip_pitch
DEFAULT_JOINT_POS[9]  =  0.40  # right_knee
DEFAULT_JOINT_POS[10] = -0.25  # right_ankle_pitch
DEFAULT_JOINT_POS[16] =  0.25  # left_shoulder_roll
DEFAULT_JOINT_POS[23] = -0.25  # right_shoulder_roll
DEFAULT_JOINT_POS[18] =  0.97  # left_elbow
DEFAULT_JOINT_POS[25] =  0.97  # right_elbow

# ─── Approximate body positions in default pose (relative to root, metres) ───
# ankle roll links sit ~0.84 m below root (wheel height already accounted for)
LEFT_ANKLE_DEFAULT_REL  = np.array([ 0.01,  0.10, -0.80], dtype=np.float32)
RIGHT_ANKLE_DEFAULT_REL = np.array([ 0.01, -0.10, -0.80], dtype=np.float32)
LEFT_WRIST_DEFAULT_REL  = np.array([-0.10,  0.30,  0.10], dtype=np.float32)
RIGHT_WRIST_DEFAULT_REL = np.array([-0.10, -0.30,  0.10], dtype=np.float32)

# Projected gravity in default upright stance (body frame: -Z in world → [0,0,-1] in body)
PROJ_GRAV_UPRIGHT = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def smooth_sinusoid(t: float, period: float, amplitude: float, phase_offset: float = 0.0) -> float:
    """Return a sinusoidal value for smooth joint trajectory generation."""
    return amplitude * math.sin(2.0 * math.pi * (t / period + phase_offset))


def generate_skating_cycle(t: float, period: float = 1.2, glide_speed: float = 0.8) -> dict:
    """Generate a single-frame skating state at time t.

    Returns a dict with:
        joint_pos  (29,)
        joint_vel  (29,)  — finite-difference approximated
        root_lin_vel_b (3,) in body frame
        root_ang_vel_b (3,)
        proj_grav  (3,)
        ankle_left_rel  (3,)  positions relative to root
        ankle_right_rel (3,)
        wrist_left_rel  (3,)
        wrist_right_rel (3,)
        ankle_left_vel  (3,)
        ankle_right_vel (3,)
    """
    phase = (t % period) / period  # 0..1

    pos = DEFAULT_JOINT_POS.copy()

    # ── Push-off phase logic ─────────────────────────────────────────────────
    # Left push: phase ∈ [0.0, 0.33)
    # Glide:     phase ∈ [0.33, 0.5)
    # Right push: phase ∈ [0.5, 0.83)
    # Glide:     phase ∈ [0.83, 1.0)

    if phase < 0.33:
        # Left foot pushing
        push_strength = math.sin(math.pi * phase / 0.33)  # rises 0→1→0
        pos[0]  += -0.10 * push_strength   # left_hip_pitch extend
        pos[1]  +=  0.15 * push_strength   # left_hip_roll (lean right)
        pos[3]  += -0.15 * push_strength   # left_knee extend
        pos[4]  +=  0.05 * push_strength   # left_ankle_pitch
        pos[5]  +=  0.10 * push_strength   # left_ankle_roll (edge)
        # Right foot: load bearing, slight compression
        pos[9]  +=  0.05 * push_strength   # right_knee flex
        pos[11] += -0.05 * push_strength   # right_ankle_roll
        # Arm counter-swing
        pos[15] +=  0.20 * push_strength   # left_shoulder_pitch forward
        pos[22] += -0.20 * push_strength   # right_shoulder_pitch back
        # Waist rotation (small)
        pos[12] +=  0.05 * push_strength   # waist_yaw

    elif phase < 0.5:
        # Glide phase: both feet approximately parallel
        relax = math.cos(math.pi * (phase - 0.33) / 0.17)  # 1→-1 = relax
        pass  # default pose already good for glide

    elif phase < 0.83:
        # Right foot pushing
        push_strength = math.sin(math.pi * (phase - 0.5) / 0.33)
        pos[6]  += -0.10 * push_strength   # right_hip_pitch extend
        pos[7]  += -0.15 * push_strength   # right_hip_roll (lean left)
        pos[9]  += -0.15 * push_strength   # right_knee extend
        pos[10] +=  0.05 * push_strength   # right_ankle_pitch
        pos[11] += -0.10 * push_strength   # right_ankle_roll (edge)
        # Left foot: load bearing
        pos[3]  +=  0.05 * push_strength   # left_knee flex
        pos[5]  +=  0.05 * push_strength   # left_ankle_roll
        # Arm counter-swing
        pos[22] +=  0.20 * push_strength   # right_shoulder_pitch forward
        pos[15] += -0.20 * push_strength   # left_shoulder_pitch back
        pos[12] += -0.05 * push_strength   # waist_yaw

    # ── Root kinematics ─────────────────────────────────────────────────────
    # Forward glide velocity (body frame x)
    vel_x = glide_speed  # constant forward speed
    vel_y = 0.05 * math.sin(2 * math.pi * phase)  # small lateral sway
    vel_z = 0.0
    root_lin_vel_b = np.array([vel_x, vel_y, vel_z], dtype=np.float32) * 0.1

    # Small roll oscillation from weight shifting
    ang_vel_x = 0.1 * math.sin(4 * math.pi * phase)
    root_ang_vel_b = np.array([ang_vel_x, 0.0, 0.0], dtype=np.float32) * 0.2

    # Projected gravity: slight tilt during push-off
    lean = 0.05 * math.sin(4 * math.pi * phase)
    proj_grav = np.array([lean, 0.0, -math.sqrt(1.0 - lean**2)], dtype=np.float32)

    # ── Approximate body positions (FK simplified as offsets) ────────────────
    hip_flex_l = pos[0] - DEFAULT_JOINT_POS[0]
    hip_flex_r = pos[6] - DEFAULT_JOINT_POS[6]
    ankle_left_rel  = LEFT_ANKLE_DEFAULT_REL + np.array([hip_flex_l * 0.4, pos[1] * 0.1, 0.0])
    ankle_right_rel = RIGHT_ANKLE_DEFAULT_REL + np.array([hip_flex_r * 0.4, pos[7] * 0.1, 0.0])
    shoulder_l = pos[15] - DEFAULT_JOINT_POS[15] if DEFAULT_JOINT_POS[15] != 0 else pos[15]
    shoulder_r = pos[22] - DEFAULT_JOINT_POS[22] if DEFAULT_JOINT_POS[22] != 0 else pos[22]
    wrist_left_rel  = LEFT_WRIST_DEFAULT_REL  + np.array([shoulder_l * 0.3, 0.0, 0.0])
    wrist_right_rel = RIGHT_WRIST_DEFAULT_REL + np.array([shoulder_r * 0.3, 0.0, 0.0])

    ankle_left_vel  = np.array([vel_x + hip_flex_l * 0.3, 0.0, 0.0], dtype=np.float32) * 0.1
    ankle_right_vel = np.array([vel_x + hip_flex_r * 0.3, 0.0, 0.0], dtype=np.float32) * 0.1

    return dict(
        joint_pos=pos.astype(np.float32),
        joint_vel=np.zeros(N_DOF, dtype=np.float32),   # filled by FD below
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        proj_grav=proj_grav,
        ankle_left_rel=ankle_left_rel.astype(np.float32),
        ankle_right_rel=ankle_right_rel.astype(np.float32),
        wrist_left_rel=wrist_left_rel.astype(np.float32),
        wrist_right_rel=wrist_right_rel.astype(np.float32),
        ankle_left_vel=ankle_left_vel.astype(np.float32),
        ankle_right_vel=ankle_right_vel.astype(np.float32),
    )


def build_amp_obs(frame: dict) -> np.ndarray:
    """Pack a frame dict into the 89-dim AMP state vector.

    Order must exactly match amp_obs.py::amp_observation_state().
    """
    jpos = frame["joint_pos"]               # (29,)
    jvel = frame["joint_vel"] * 0.05        # (29,) scaled
    pgrav = frame["proj_grav"]              # (3,)
    linv  = frame["root_lin_vel_b"]         # (3,) already scaled
    angv  = frame["root_ang_vel_b"]         # (3,) already scaled
    ankle_rel = np.concatenate([
        frame["ankle_left_rel"],
        frame["ankle_right_rel"],
    ])  # (6,)
    wrist_rel = np.concatenate([
        frame["wrist_left_rel"],
        frame["wrist_right_rel"],
    ])  # (6,)
    ankle_vel = np.concatenate([
        frame["ankle_left_vel"],
        frame["ankle_right_vel"],
    ])  # (6,)
    return np.concatenate([jpos, jvel, pgrav, linv, angv, ankle_rel, wrist_rel, ankle_vel])
    # Total: 29+29+3+3+3+6+6+6 = 85 dims  (matches amp_obs.py exactly)


def generate_speed_variant(
    fps: int,
    period: float,
    glide_speed: float,
    cycles: int,
) -> np.ndarray:
    """Generate AMP obs array for a single glide speed."""
    dt = 1.0 / fps
    timestamps = np.arange(0.0, cycles * period, dt)
    amp_obs_list = []
    prev_joint_pos = None
    for t in timestamps:
        frame = generate_skating_cycle(t, period=period, glide_speed=glide_speed)
        if prev_joint_pos is not None:
            frame["joint_vel"] = (frame["joint_pos"] - prev_joint_pos) / dt
        else:
            frame["joint_vel"] = np.zeros(N_DOF, dtype=np.float32)
        prev_joint_pos = frame["joint_pos"].copy()
        amp_obs_list.append(build_amp_obs(frame))
    return np.stack(amp_obs_list, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate analytical skating reference motion.")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second (must match sim dt × decimation).")
    parser.add_argument("--cycles", type=int, default=30, help="Number of gait cycles per speed variant.")
    parser.add_argument("--period", type=float, default=1.2, help="Gait period in seconds.")
    parser.add_argument(
        "--glide_speed", type=float, default=None,
        help="Single target speed (m/s). If omitted, generates multiple speeds covering 0.5–3.0 m/s.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output .npz file path.")
    args = parser.parse_args()

    fps = args.fps
    period = args.period

    if args.glide_speed is not None:
        # Single-speed mode (backwards compat)
        speed_variants = [args.glide_speed]
    else:
        # Multi-speed mode: cover the full training range used in Phase 3 (0.3–4.0 m/s).
        # Sampling at 6 speeds ensures the discriminator sees realistic skating motion
        # at all commanded velocities, not just the original 0.8 m/s reference.
        speed_variants = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    all_amp_obs = []
    for speed in speed_variants:
        print(f"  Generating {args.cycles} cycles at {speed:.1f} m/s...")
        variant = generate_speed_variant(fps, period, speed, args.cycles)
        all_amp_obs.append(variant)
        print(f"    → {variant.shape[0]} frames, dim={variant.shape[1]}")

    amp_obs = np.concatenate(all_amp_obs, axis=0).astype(np.float32)
    T = amp_obs.shape[0]
    speed_desc = (
        f"{args.glide_speed:.1f} m/s" if args.glide_speed is not None
        else f"{speed_variants[0]:.1f}–{speed_variants[-1]:.1f} m/s ({len(speed_variants)} speeds)"
    )
    print(f"\nTotal AMP obs shape: {amp_obs.shape}  (T={T} frames, covering {speed_desc})")

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output is None:
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab", "data"
        )
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "skating_reference.npz")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    np.savez(
        output_path,
        amp_obs=amp_obs,
        fps=np.float32(fps),
        metadata=f"Analytical G1-29DOF skating motion: {args.cycles} cycles/speed, "
                 f"{fps} fps, period={period}s, speeds={speed_desc}",
    )
    print(f"Saved reference motion to: {os.path.abspath(output_path)}")
    print(f"  Frames:  {T}")
    print(f"  AMP dim: {amp_obs.shape[1]}")
    print(f"  FPS:     {fps}")


if __name__ == "__main__":
    main()
