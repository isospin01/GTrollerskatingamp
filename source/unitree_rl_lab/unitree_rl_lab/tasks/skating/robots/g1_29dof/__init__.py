"""Gym environment registrations for G1-29DOF roller-skating tasks."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-Skating-Phase1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingGlideEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingGlidePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase1PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Skating-Phase2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPushEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPushPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase2PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Skating-Phase3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingFullEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingFullPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase3PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Skating-Phase3b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPhase3bEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPhase3bPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase3bPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-Skating-Phase4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPhase4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.skating_env_cfg:G1SkatingPhase4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.skating.agents.rsl_rl_ppo_cfg:SkatingPhase4PPORunnerCfg",
    },
)
