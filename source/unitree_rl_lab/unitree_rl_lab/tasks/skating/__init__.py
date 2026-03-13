"""Roller-skating task module for Unitree humanoids.

Two registered environments:

  Unitree-G1-Skating-Phase1-v0
    Gliding balance on low-friction rink. Hand-designed reward, no AMP.
    Run: python scripts/skating/train.py --task Unitree-G1-Skating-Phase1-v0

  Unitree-G1-Skating-Eureka-v0
    Full skating (push-off, speed, turning) with reward proposed by Eureka
    (LLM-driven reward search). Reward function injected via EUREKA_REWARD_FN_PATH.
    Run: python scripts/skating/eureka_phase2.py --phase1_run skating_phase1
"""

# Import submodules to trigger gym registration
import unitree_rl_lab.tasks.skating.robots.g1_29dof  # noqa: F401
