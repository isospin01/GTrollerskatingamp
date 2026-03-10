"""RSL-RL PPO runner configurations for roller-skating phases.

Three configs are provided, one per training phase:
  - SkatingPhase1PPORunnerCfg: gliding balance, no AMP
  - SkatingPhase2PPORunnerCfg: push-off with AMP discriminator
  - SkatingPhase3PPORunnerCfg: full velocity + turning, AMP enabled

AMP discriminator is disabled in Phase 1 because the reference NPZ
file only contains push-glide cycles (not static balance poses).
"""

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Path to the pre-generated skating reference motion NPZ
_REFERENCE_MOTION_NPZ = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "skating_reference.npz",
)
_REFERENCE_MOTION_NPZ = os.path.normpath(_REFERENCE_MOTION_NPZ)


@configclass
class SkatingPhase1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1: Gliding balance — pure PPO, no AMP (robot just needs to stay upright)."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "skating_phase1"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP disabled for Phase 1
    amp: dict = {
        "enabled": False,
    }


@configclass
class SkatingPhase2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2: Push-off learning — PPO + AMP discriminator active.

    Resume from Phase 1 checkpoint before running:
        --resume --load_run skating_phase1
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "skating_phase2"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,   # lower noise std — policy is already somewhat trained
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP configuration
    amp: dict = {
        "enabled": True,
        "motion_file": _REFERENCE_MOTION_NPZ,
        "amp_obs_dim": 0,           # auto-detected from env extras at runtime
        "hidden_dims": [1024, 512],
        "activation": "leaky_relu",
        "learning_rate": 1e-4,
        "replay_buffer_size": 100_000,
        "task_reward_lerp": 0.5,    # 50% task, 50% style
        "disc_grad_penalty": 5.0,
        "disc_update_steps": 3,
        "disc_mini_batch_size": 4096,
    }


@configclass
class SkatingPhase3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 3: Full velocity + turning — PPO + AMP, larger network.

    Resume from Phase 2 checkpoint:
        --resume --load_run skating_phase2
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "skating_phase3"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP: slightly reduced blend in Phase 3 to allow more task focus
    amp: dict = {
        "enabled": True,
        "motion_file": _REFERENCE_MOTION_NPZ,
        "amp_obs_dim": 0,
        "hidden_dims": [1024, 512],
        "activation": "leaky_relu",
        "learning_rate": 5e-5,
        "replay_buffer_size": 100_000,
        "task_reward_lerp": 0.3,    # 70% task, 30% style in final phase
        "disc_grad_penalty": 5.0,
        "disc_update_steps": 2,
        "disc_mini_batch_size": 4096,
    }


@configclass
class SkatingPhase3bPPORunnerCfg(SkatingPhase3PPORunnerCfg):
    """Phase 3b: curriculum fix — resume from Phase 3 checkpoint.

    Resume from Phase 3 checkpoint:
        --resume --load_run skating_phase3 --load_checkpoint model_8000.pt
    """

    max_iterations = 5000
    save_interval = 500
    experiment_name = "skating_phase3b"


@configclass
class SkatingPhase4PPORunnerCfg(SkatingPhase3PPORunnerCfg):
    """Phase 4: straight-line skating — resume from Phase 3b checkpoint.

    Resume from Phase 3b checkpoint:
        --resume --load_run <run_dir> --checkpoint model_10000.pt
        --load_experiment skating_phase3b
    """

    max_iterations = 5000
    save_interval = 500
    experiment_name = "skating_phase4"
