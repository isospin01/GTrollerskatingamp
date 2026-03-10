# GT Roller Skating: Teaching a Humanoid Robot to Roller-Skate with Deep RL

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Train a **Unitree G1 29-DOF humanoid** to roller-skate from rest to 3.5 m/s using multi-phase reinforcement learning with Adversarial Motion Priors (AMP) in NVIDIA Isaac Sim.

## Overview

This project uses a progressive, phase-based curriculum to teach a simulated humanoid robot how to roller-skate. Starting from basic balance on wheels, the agent learns to push off, glide, and eventually sustain high-speed straight-line skating -- all driven by reward shaping, curriculum learning, and style guidance from an AMP discriminator.

### Key Results

| Metric | Value |
|---|---|
| Max commanded speed | **3.5 m/s** |
| Fall rate | **< 0.1%** |
| Episode survival | **~20s / 20s** (full episodes) |
| Undesired contacts | **0.000** across all phases |
| Training time (total) | ~10 hours across 4 phases |

## Architecture

```
                 ┌──────────────┐
                 │  PPO Policy  │  (RSL-RL AmpOnPolicyRunner)
                 │  Actor-Critic│
                 └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
   ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
   │   Reward    │ │  AMP   │ │ Curriculum  │
   │  Functions  │ │ Discrim│ │  Manager    │
   │ (12 terms)  │ │(85-dim)│ │(vel levels) │
   └─────────────┘ └────────┘ └────────────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
              ┌─────────▼─────────┐
              │   Isaac Sim Env   │
              │  G1 + Roller Skates│
              │  (4096 parallel)  │
              └───────────────────┘
```

## Training Phases

The robot learns progressively through four phases, each building on the previous checkpoint:

### Phase 1 -- Gliding Balance
- Robot starts with initial velocity on low-friction wheels
- Learns to stay upright and maintain forward glide
- No push-off required; pure balance training
- **Result**: Stable gliding at 0.5--1.5 m/s, <1% fall rate

### Phase 2 -- Push-Off with AMP
- Robot starts from rest and must self-propel
- Alternating push-off stride via `push_off_rhythm` reward
- AMP discriminator trained on analytical skating reference (18,000 frames, 6 speed variants)
- **Result**: Learned to self-propel from rest with natural skating motion

### Phase 3/3b -- Velocity Curriculum + Turning
- Curriculum gradually increases commanded speed range
- Added turning commands and edge-contact rewards
- Phase 3b fixed a curriculum stall by lowering the advancement threshold
- **Result**: Robot reached 2.1 m/s with turning capability up to 0.8 rad/s

### Phase 4 -- Straight-Line Mastery
- Removed all turning to focus on high-speed forward skating
- Boosted forward velocity reward, added yaw-rate penalty
- **Result**: Curriculum reached **3.5 m/s** maximum, near-perfect episode survival

## Repository Structure

```
├── scripts/skating/
│   ├── train.py                     # Training entry point (AMP + PPO)
│   ├── play.py                      # Evaluation + video recording
│   ├── record_demo.py               # Camera-tracked demo footage
│   ├── gen_skating_reference.py     # Generate AMP reference data
│   └── run_phase*.sh                # Shell scripts for each phase
│
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── tasks/skating/
│   │   ├── robots/g1_29dof/
│   │   │   └── skating_env_cfg.py   # Phase 1-4 environment configs
│   │   ├── mdp/
│   │   │   ├── rewards.py           # 12 custom skating reward functions
│   │   │   ├── curriculums.py       # Skating-specific curriculum (threshold 0.3)
│   │   │   ├── amp_obs.py           # 85-dim AMP observation vector
│   │   │   ├── skate_attachment.py  # USD skate geometry + collision physics
│   │   │   ├── events.py            # Reset pose with velocity injection
│   │   │   └── observations.py      # Skating-specific obs terms
│   │   └── agents/
│   │       ├── amp_runner.py         # AMP-aware PPO runner
│   │       └── rsl_rl_ppo_cfg.py     # Per-phase hyperparameter configs
│   ├── tasks/locomotion/            # Base locomotion MDP (inherited)
│   ├── assets/robots/unitree.py     # Robot config with skate init height
│   └── data/skating_reference.npz   # AMP reference motion data
│
├── tests/skating/                   # 36 unit tests
│   ├── test_skate_physics.py        # Wheel geometry + collision
│   ├── test_config_consistency.py   # Phase clock, critic cfg, heights
│   ├── test_amp_and_reference.py    # AMP obs dim + NPZ validity
│   └── test_reset_and_rewards.py    # Velocity rotation + reward logic
│
├── PROJECT_STATUS.md                # Detailed training history and metrics
└── deploy/                          # Sim2Sim (MuJoCo) and Sim2Real tools
```

## Installation

### Prerequisites

- NVIDIA Isaac Sim >= 5.0
- NVIDIA Isaac Lab >= 2.3.0
- NVIDIA GPU with >= 24 GB VRAM (48 GB recommended for 4096 envs)
- Conda / Mamba

### Setup

1. **Install Isaac Lab** following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. **Clone this repository** (outside the IsaacLab directory):

   ```bash
   git clone https://github.com/isospin01/GTrollerskating.git
   cd GTrollerskating
   ```

3. **Install the package** in editable mode:

   ```bash
   conda activate env_isaaclab
   ./unitree_rl_lab.sh -i
   ```

4. **Download robot model** -- choose one method:

   *Method A: URDF (recommended for Isaac Sim >= 5.0)*
   ```bash
   git clone https://github.com/unitreerobotics/unitree_ros.git
   ```
   Then set `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

   *Method B: USD*
   ```bash
   git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
   ```
   Then set `UNITREE_MODEL_DIR` in the same file.

5. **Generate AMP reference data** (if `data/skating_reference.npz` is missing):

   ```bash
   python scripts/skating/gen_skating_reference.py
   ```

## Usage

### Training

Each phase resumes from the previous phase's best checkpoint:

```bash
cd GTrollerskating
conda activate env_isaaclab

# Phase 1: Gliding balance (from scratch)
python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase1-v0 \
    --num_envs 4096 --headless \
    --logger wandb --log_project_name g1_skating \
    --experiment_name skating_phase1

# Phase 2: Push-off + AMP (resume from Phase 1)
python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase2-v0 \
    --num_envs 4096 --headless \
    --resume --load_experiment skating_phase1 --load_run "2026-.*" \
    --logger wandb --log_project_name g1_skating \
    --experiment_name skating_phase2

# Phase 3: Full velocity + turning (resume from Phase 2)
python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase3-v0 \
    --num_envs 4096 --headless \
    --resume --load_experiment skating_phase2 --load_run "2026-.*" \
    --logger wandb --log_project_name g1_skating \
    --experiment_name skating_phase3

# Phase 4: Straight-line mastery (resume from Phase 3b)
python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase4-v0 \
    --num_envs 4096 --headless \
    --resume --load_experiment skating_phase3b --load_run "2026-.*" \
    --logger wandb --log_project_name g1_skating \
    --experiment_name skating_phase4
```

### Evaluation & Recording

```bash
# Record demo footage (side-tracking camera)
python scripts/skating/record_demo.py \
    --vx 3.0 --wz 0.0 --duration 40 \
    --name demo_3ms \
    --checkpoint model_14000.pt \
    --log_dir skating_phase4 \
    --headless

# Quantitative evaluation
python scripts/skating/play.py \
    --task Unitree-G1-Skating-Phase4-v0 \
    --num_envs 64 --headless \
    --load_experiment skating_phase4 --load_run "2026-.*" \
    --eval_episodes 100
```

### Running Tests

```bash
python -m pytest tests/skating/ -v
```

## Reward Design

| Reward Term | Weight | Phase | Purpose |
|---|---|---|---|
| `forward_velocity_tracking_exp` | 5.0--8.0 | 1--4 | Track commanded forward speed |
| `push_off_rhythm` | 1.5 | 2--4 | Alternating push-off stride (1.2s period) |
| `glide_continuity` | 2.0 | 1--4 | Reward sustained foot contact during glide |
| `upright_orientation_l2` | -5.0 | 1--4 | Penalize torso tilt |
| `base_height_skating_l2` | -10.0 | 1--4 | Maintain 0.85m CoM height |
| `lateral_velocity_penalty` | -1.0 | 1--4 | Penalize sideways drift |
| `track_ang_vel_z_exp` | 2.5 | 3 | Track turning commands |
| `skate_foot_edge_contact` | 1.0 | 3 | Reward ankle roll for carving |
| `ang_vel_z_l2` | -1.0 | 4 | Keep robot heading straight |
| `skate_energy` | -0.01 | 1--4 | Energy efficiency |
| `skate_action_rate` | -0.03--0.05 | 1--4 | Smooth actions |
| `AMP discriminator` | 0.5 | 2--4 | Natural skating style |

## Skate Physics

The robot wears procedurally-generated roller skates attached via USD:

- **4 wheel cylinders** per foot (front/back collision, 4 visual)
- **Low-friction material**: static=0.02, dynamic=0.01, combine_mode="min"
- **Init height**: 0.876m (raised by 0.076m to account for wheel radius vs sphere feet)
- **Proof**: `undesired_contacts = 0.000` across all 14,000+ training iterations

## Curriculum Strategy

The skating-specific curriculum (`skating/mdp/curriculums.py`) advances the velocity command range as the agent improves:

- **Threshold**: `reward > weight * 0.3` (lowered from generic 0.8, since from-rest acceleration drags down episode averages)
- **Step size**: 0.2 m/s per advancement (doubled from generic 0.1)
- **Progression**: 1.2 m/s (Phase 3 stuck) -> 2.1 m/s (Phase 3b) -> **3.5 m/s** (Phase 4)

## Tech Stack

| Component | Details |
|---|---|
| Simulator | NVIDIA Isaac Lab / Isaac Sim 5.0 |
| Robot | Unitree G1 29-DOF |
| RL Algorithm | RSL-RL PPO with AMP (`AmpOnPolicyRunner`) |
| Style Guidance | Adversarial Motion Priors (85-dim, 18K frames) |
| Logging | Weights & Biases |
| Training | 4096 parallel environments, single GPU |

## Acknowledgements

This project is built on top of:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) -- simulation and training framework
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) -- base RL environments for Unitree robots
- [MuJoCo](https://github.com/google-deepmind/mujoco) -- sim2sim validation
- [HUSKY](https://arxiv.org/abs/2602.03205) -- phase-wise training strategy inspiration

## License

Apache 2.0. See [LICENCE](LICENCE) for details.
