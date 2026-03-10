"""Roller-skating task module for Unitree humanoids.

Provides three phase-progressive training environments for the Unitree G1-29DOF:

  Unitree-G1-Skating-Phase1-v0  — Gliding balance on low-friction rink
  Unitree-G1-Skating-Phase2-v0  — Push-off learning with AMP reference motion
  Unitree-G1-Skating-Phase3-v0  — Full velocity + turning commands (lean-to-steer)

Usage:
    python scripts/rsl_rl/train.py --task Unitree-G1-Skating-Phase1-v0 \\
        --num_envs 4096 --headless --logger wandb --log_project_name g1_skating
"""

# Import submodules to trigger gym registration
import unitree_rl_lab.tasks.skating.robots.g1_29dof  # noqa: F401
