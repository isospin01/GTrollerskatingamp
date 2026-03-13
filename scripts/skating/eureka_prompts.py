"""LLM prompt templates for the Eureka roller-skating reward search.

Two prompts are used:
  1. initial_prompt()    — first iteration: full task description + env context
  2. feedback_prompt()   — subsequent iterations: previous code + metrics + critique
"""

from __future__ import annotations

# ── Shared environment context injected into every prompt ─────────────────────

_ENV_CONTEXT = """\
=== ENVIRONMENT CONTEXT ===

Robot: Unitree G1 humanoid, 29 DOF
Simulation: NVIDIA Isaac Lab (Isaac Sim 5.1.0), 200 Hz physics / 50 Hz policy
Terrain: Flat rink, floor friction μ_static=0.05, μ_dynamic=0.02 (polished surface)
Skates: Roller skates attached to both ankles, wheel friction μ=0.01–0.02
Episode length: 20 seconds maximum
Number of parallel envs: 2048

--- env attributes you may access ---

  # Robot kinematics  (all tensors shape [num_envs, ...])
  robot = env.scene["robot"]                          # Articulation
  robot.data.root_pos_w          # [N, 3] world position of base link
  robot.data.root_lin_vel_b      # [N, 3] base linear velocity in body frame
                                 #   [:, 0] = forward (X), [:, 1] = lateral (Y)
  robot.data.root_ang_vel_b      # [N, 3] base angular velocity in body frame
                                 #   [:, 2] = yaw rate
  robot.data.root_quat_w         # [N, 4] base orientation quaternion (w, x, y, z)
  robot.data.projected_gravity_b # [N, 3] gravity vector in body frame
                                 #   upright → ≈(0, 0, -1); tilted → XY nonzero
  robot.data.joint_pos           # [N, 29] joint positions (radians)
  robot.data.joint_vel           # [N, 29] joint velocities (rad/s)
  robot.data.applied_torque      # [N, 29] applied joint torques (Nm)
  robot.data.joint_names         # list of 29 joint name strings

  # Commanded velocity  (shape [N, 3]: vx_cmd, vy_cmd, wz_cmd)
  cmd = env.command_manager.get_command("base_velocity")
  # cmd[:, 0] = forward velocity command (m/s)
  # cmd[:, 2] = yaw rate command (rad/s)

  # Contact sensor  (both ankle_roll links)
  contact_sensor = env.scene.sensors["contact_forces"]
  # contact_sensor.data.current_contact_time  shape [N, num_bodies]
  # contact_sensor.data.net_forces_w          shape [N, num_bodies, 3]

  # Phase / timing
  env.episode_length_buf   # [N] integer step counter per env
  env.step_dt              # scalar: policy dt = 0.02 s (50 Hz)

  # Misc
  env.num_envs             # int: number of parallel environments
  env.device               # torch.device

--- joint names (29 DOF, in order) ---
  left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
  left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
  right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
  right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint,
  waist_yaw_joint, waist_roll_joint, waist_pitch_joint,
  left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint,
  left_elbow_joint, left_wrist_roll_joint, left_wrist_pitch_joint, left_wrist_yaw_joint,
  right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint,
  right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint

--- existing reward helpers you MAY import and call ---
  from unitree_rl_lab.tasks.skating.mdp import (
      forward_velocity_tracking_exp,   # exp(-||vx - vx_cmd||^2 / std^2)
      lateral_velocity_penalty,        # vy^2  (penalty)
      skate_glide_continuity,          # 1 if vx > min_speed and cmd active
      push_off_rhythm,                 # alternating foot contact phase-clock reward
      skate_foot_edge_contact,         # ankle roll aligned with turn command
      upright_orientation_l2,          # gravity_xy^2  (penalty)
      base_height_skating_l2,          # (height - target)^2  (penalty)
      skate_energy,                    # |torque| * |vel|  (penalty)
      skate_action_rate,               # action change rate  (penalty)
  )
  from isaaclab.managers import SceneEntityCfg

=== END CONTEXT ===
"""

_REQUIRED_SIGNATURE = """\
=== REQUIRED FUNCTION SIGNATURE ===

Your output MUST contain exactly one Python function with this signature:

    def compute_reward(env) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

Rules:
  1. Import torch and any other modules at MODULE LEVEL (before the function definition),
     NOT inside the function body.  Example:
       import torch
       def compute_reward(env) -> ...:
           ...
  2. Return (total_reward, components_dict) where:
       total_reward    : torch.Tensor shape (env.num_envs,)  — per-env scalar
       components_dict : dict[str, Tensor]  — named components for logging
  3. Reward scales: individual positive components should be O(1)–O(5).
     Negative penalties should be O(-0.1)–O(-5).
     NEVER return values larger than ±10 for a single component.
  4. All tensors must be on env.device.
  5. Do NOT call functions that modify env state (e.g. apply forces).
  6. The function runs at every policy step (50 Hz) across 2048 envs — keep it efficient.
  7. Use torch operations only; no Python loops over num_envs.

Wrap your code in a ```python ... ``` block so it can be extracted automatically.

=== END SIGNATURE ===
"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert reward-function engineer for deep reinforcement learning on legged robots.
Your task is to design reward functions that teach a 29-DOF Unitree G1 humanoid robot
to roller-skate inside NVIDIA Isaac Lab.

You write Python reward functions using PyTorch.  The environment already provides
structural safety rewards (upright posture, base height, energy, smoothness).
Your function provides the TASK-SPECIFIC rewards only:
  - Forward speed tracking (match the commanded speed)
  - Push-off rhythm (alternating foot contact to propel forward)
  - Turning (follow commanded angular velocity, optional)
  - Any other skating-relevant signals you judge useful

Be creative, precise, and physically grounded.  Good skating reward functions
explicitly address the bio-mechanical challenges of roller skating:
  * Low floor friction → balance is unstable
  * Must push off diagonally to accelerate (no passive propulsion)
  * Glide phase: one foot on floor, other foot pushed sideways and lifted
  * Arms counter-swing for balance during push-off
"""


def initial_prompt(env_source_excerpt: str) -> str:
    """First Eureka iteration: full task description with env context."""
    return f"""\
{_ENV_CONTEXT}

=== TASK DESCRIPTION ===

The Unitree G1 robot has completed Phase 1 training, where it learned to:
  ✓ Balance upright on roller skates (low-friction wheels on a polished rink)
  ✓ Passively glide forward at 0.5–1.5 m/s with a pre-injected initial velocity
  ✓ Maintain ~0.85 m base height (bent-knee skating posture)
  ✓ Resist small random pushes

Phase 2 goal (YOUR reward function):
  → Teach the robot to ACTIVELY PUSH OFF from near-rest (init vel 0–0.5 m/s)
  → Accelerate to the commanded speed (up to 3.5 m/s with curriculum)
  → Maintain a natural alternating push-off gait (left/right foot alternating)
  → Follow turning commands (angular velocity up to ±0.8 rad/s)
  → Exhibit natural arm counter-swing for balance

The most important new skill is the push-off: the robot must learn to
push sideways-backward with one foot while gliding on the other, then switch.
Without a reward for alternating foot contact and forward momentum, the robot
tends to stand still (no penalty) or shuffle without real push-off.

=== RELEVANT ENVIRONMENT CODE EXCERPT ===

{env_source_excerpt}

=== END CODE EXCERPT ===

{_REQUIRED_SIGNATURE}

Design a `compute_reward` function for Phase 2 roller skating.
Focus on: (1) push-off rhythm, (2) forward speed tracking, (3) optional turning.
"""


def feedback_prompt(
    prev_code: str,
    iteration: int,
    train_summary: dict,
    eval_metrics: dict | None,
) -> str:
    """Subsequent Eureka iterations: previous code + metrics + critique."""

    # Build human-readable training summary
    reward_trend = train_summary.get("reward_trend", "unknown")
    mean_ep_len = train_summary.get("mean_episode_length_s", "unknown")
    converged = train_summary.get("converged", False)
    stdout_tail = train_summary.get("stdout_tail", "")

    train_block = f"""\
Training summary (iteration {iteration - 1} → {iteration}):
  Reward trend          : {reward_trend}
  Mean episode length   : {mean_ep_len} s  (max = 20 s)
  Converged             : {converged}
  Last training output  :
{stdout_tail}"""

    # Build eval block
    if eval_metrics:
        fwd = eval_metrics.get("mean_forward_speed", "N/A")
        lat = eval_metrics.get("mean_lateral_drift", "N/A")
        eplen = eval_metrics.get("mean_episode_length", "N/A")
        fall = eval_metrics.get("fall_rate_pct", "N/A")
        fitness = eval_metrics.get("fitness_score", "N/A")
        eval_block = f"""\
Evaluation metrics (100 episodes, headless):
  Mean forward speed    : {fwd} m/s   (target: ≥ 2.0 m/s)
  Mean lateral drift    : {lat} m/s   (target: < 0.15 m/s)
  Mean episode length   : {eplen} s  (target: ≥ 15 s)
  Fall rate             : {fall}%      (target: < 5%)
  Composite fitness     : {fitness}    (higher is better; range ~0–1)"""
    else:
        eval_block = "Evaluation: not available (training may have crashed)"

    return f"""\
{_ENV_CONTEXT}

=== PREVIOUS REWARD FUNCTION (Eureka iteration {iteration - 1}) ===

```python
{prev_code}
```

=== RESULTS ===

{train_block}

{eval_block}

=== ANALYSIS REQUEST ===

Based on the results above, identify what is working and what is not:
  - Is the robot learning to push off? (episode length > 5 s suggests survival)
  - Is it tracking speed? (forward speed close to commanded?)
  - Are there signs of reward hacking? (e.g. high reward but low eval speed)
  - Is the reward too sparse, too dense, or poorly scaled?

Then produce an IMPROVED `compute_reward` function that fixes the issues.
Common improvements to consider:
  1. If the robot is not pushing off: strengthen the push-off rhythm reward
     and/or add a reward for foot air-time (one foot off the floor)
  2. If the robot is falling: the structural rewards already penalise tilting;
     consider adding a small bonus for low angular velocity magnitude
  3. If speed tracking is poor: increase the forward velocity reward weight
     or use a tighter std in the exponential kernel
  4. If the robot shuffles without gliding: reward sustained momentum
     (vx > min_speed) rather than just instantaneous speed matching
  5. If turning is commanded but not followed: add track_ang_vel_z_exp

{_REQUIRED_SIGNATURE}
"""
