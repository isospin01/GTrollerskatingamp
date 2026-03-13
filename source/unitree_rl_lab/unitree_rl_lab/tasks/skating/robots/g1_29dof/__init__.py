"""Gym environment registrations for G1-29DOF roller-skating tasks."""

import gymnasium as gym

# Phase 1: gliding balance (hand-designed reward, no AMP)
gym.register(
    id="Unitree-G1-Skating-Phase1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.skating_env_cfg:G1SkatingGlideEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingGlidePlayEnvCfg",
        "rsl_rl_cfg_entry_point":   "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase1PPORunnerCfg",
    },
)

# Eureka Phase 2+: full skating with LLM-generated reward (EUREKA_REWARD_FN_PATH)
gym.register(
    id="Unitree-G1-Skating-Eureka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.eureka_env_cfg:G1SkatingEurekaEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.eureka_env_cfg:G1SkatingEurekaPlayEnvCfg",
        "rsl_rl_cfg_entry_point":   "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingEurekaPPORunnerCfg",
    },
)

# AMP Phase: human-video style guidance via discriminator reward (GENMO retargeting pipeline)
# Prerequisites:
#   1. Run: python scripts/skating/retarget_video_to_amp.py \
#               --video_url https://www.youtube.com/shorts/4b-bjwTWPRA
#   2. Confirm: source/unitree_rl_lab/unitree_rl_lab/data/skating_reference_human.npz exists
gym.register(
    id="Unitree-G1-Skating-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.amp_env_cfg:G1SkatingAMPEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.amp_env_cfg:G1SkatingAMPPlayEnvCfg",
        "rsl_rl_cfg_entry_point":   "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingAMPPPORunnerCfg",
    },
)
