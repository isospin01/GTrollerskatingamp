# Unitree G1 Roller-Skating RL — Project Status

**Last updated:** 2026-03-12  
**Current phase:** AMP PPO (human-video style guidance) — **training in progress** 🏃

---

## 1. Project Goal

Train a **Unitree G1 29-DOF humanoid robot** to roller-skate using deep reinforcement learning in NVIDIA Isaac Sim. Target behaviours:

1. Stay upright on roller skates while gliding ✅ **DONE (Phase 1)**
2. Self-propel from rest using alternating push-off strides ← **AMP target (in training)**
3. Skate in a straight line at up to ~3.5 m/s ← **AMP target**
4. Track turning commands using lean-to-steer edge control ← **AMP target**

---

## 2. Architecture

### Original design (Phases 1–4, hand-designed rewards)
Phase 1 → Phase 2 (AMP push-off) → Phase 3/3b (turning) → Phase 4 (straight-line speed)

### Current design (as of 2026-03-12)
**Phase 1 (complete) → AMP PPO (human-video style, active) ‖ Eureka loop (paused)**

Two parallel tracks have been developed:

- **AMP track** — Human skating motion extracted from a YouTube video via GVHMR (GENMO's video backbone), retargeted to G1 kinematics, and used as a discriminator reference for AMP PPO. Currently training.
- **Eureka track** — LLM-driven reward search (GPT-5.4 proposes `compute_reward()` functions). Infrastructure complete; paused while AMP track runs.

---

## 3. Stack

| Component | Details |
|---|---|
| Simulator | NVIDIA Isaac Lab / Isaac Sim 5.1.0 |
| Robot | Unitree G1 29-DOF (`g1_29dof_nohand-feet_sphere.usd`) |
| RL algorithm | RSL-RL PPO (`AmpOnPolicyRunner`) |
| Reward search | NVIDIA Eureka — LLM proposes `compute_reward()` functions |
| LLM model | GPT-5.4 (OpenAI, released 2026-03-05) — default; also supports Claude, Gemini |
| Logging | WandB project `g1_skating` |
| GPUs | 8× NVIDIA RTX 6000 Ada (48 GB each) — shared server |
| Python env | `conda env: env_isaaclab` |
| Repo | `/home/muchenxu/rollerskating/` |

---

## 4. Key Source Files

```
rollerskating/
├── scripts/skating/
│   ├── train.py                    # Training entry point (Phase 1 + Eureka + AMP)
│   ├── play.py                     # Eval + video recording
│   ├── record_demo.py              # Demo footage with camera tracking
│   ├── gen_skating_reference.py    # Analytical AMP reference motion generator
│   ├── smpl_to_g1_retarget.py      # ★ SMPL→G1 kinematic retargeter
│   ├── retarget_video_to_amp.py    # ★ End-to-end pipeline: YouTube URL → AMP .npz
│   ├── eureka_phase2.py            # Eureka orchestration loop (LLM → train → eval)
│   ├── eureka_prompts.py           # LLM prompt templates
│   ├── run_eureka.sh               # One-command Eureka launcher
│   └── run_phase1.sh
│
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/unitree.py               # UNITREE_G1_29DOF_SKATE_CFG
│   ├── tasks/skating/
│   │   ├── robots/g1_29dof/
│   │   │   ├── skating_env_cfg.py             # Phase 1 env config (shared components)
│   │   │   ├── eureka_env_cfg.py              # Eureka env (injectable reward)
│   │   │   └── amp_env_cfg.py                 # ★ AMP env (fixed task rewards + discriminator)
│   │   ├── mdp/
│   │   │   ├── rewards.py                     # Hand-crafted reward helpers
│   │   │   ├── eureka_rewards.py              # Dynamic reward injection
│   │   │   ├── skate_attachment.py
│   │   │   ├── curriculums.py
│   │   │   ├── events.py
│   │   │   ├── amp_obs.py                     # 85-dim AMP state vector
│   │   │   └── observations.py
│   │   └── agents/
│   │       ├── amp_runner.py                  # AmpOnPolicyRunner (PPO + discriminator)
│   │       └── rsl_rl_ppo_cfg.py              # Phase 1 + Eureka + AMP PPO configs
│   └── data/
│       ├── skating_reference.npz              # Analytical AMP reference (18000 × 85)
│       └── skating_reference_human.npz        # ★ Human-video AMP reference (T × 85)
│
├── logs/rsl_rl/
│   └── skating_amp/2026-03-11_21-04-26/       # ★ Active AMP training run
│       ├── model_2100.pt                      # Latest checkpoint (iter 2100/3000)
│       └── events.out.tfevents.*              # TensorBoard (also mirrored to W&B)
│
└── eureka_output/                             # Eureka run outputs
    ├── iter_0/
    │   ├── reward_fn.py                       # GPT-5.4-generated reward function
    │   ├── llm_response.txt
    │   ├── training_stdout.txt
    │   └── train_summary.json
    └── eureka_summary.json
```

★ = new files added in the AMP integration (2026-03-12)

---

## 5. Phase 1 — Gliding Balance (COMPLETE ✅)

**Checkpoint:** `/home/muchenxu/unitree_rl_lab/logs/rsl_rl/skating_phase1/2026-03-08_13-46-21/model_1999.pt`

| Metric | Value |
|---|---|
| Iterations | 2,000 / 2,000 |
| Wall time | 1h 22m |
| Final reward | 58.5 |
| Episode length | 701 steps (~14s) |
| forward_velocity reward | 4.19 |
| Fall rate | ~0.6% |
| Undesired contacts | **0.000** (wheels only) |
| Speed range trained | 0.5–1.5 m/s (glide, pre-injected velocity) |

**What the robot learned:** Balance upright on roller skates, maintain ~0.85m bent-knee posture, track forward velocity passively, resist random pushes. Does NOT push off from rest.

---

## 6. AMP Phase — Human-Video Style Guidance (IN TRAINING 🔄)

### Concept

```
r_total = 0.5 × r_task  +  0.5 × r_AMP
r_AMP   = −log(1 − σ(D(s_AMP)))   ∈ [0, ~9]
```

The AMP discriminator is trained **simultaneously** with the PPO policy (GAN-style). It learns to distinguish real human skating frames from robot-generated frames, and its output is used as a reward signal that teaches the robot *how* to skate (style) while task rewards define *what* to achieve (speed/balance).

### Pipeline: YouTube → AMP Reference

```
YouTube URL (human skating video)
    │
    ▼  yt-dlp + ffmpeg
Video frames (1280×720, 30fps)
    │
    ▼  GVHMR (gvhmr conda env) — GENMO's video backbone
hmr4d_results.pt  (SMPL body_pose, global_orient, transl per frame)
    │
    ▼  SMPLtoG1Retargeter  (scripts/skating/smpl_to_g1_retarget.py)
G1 joint angles (29-DOF) + base pose + velocities
    │
    ▼  build_amp_obs()
skating_reference_human.npz  (T × 85 AMP observation array)
```

**Video source:** [https://www.youtube.com/shorts/4b-bjwTWPRA](https://www.youtube.com/shorts/4b-bjwTWPRA)

### AMP Environment (`Unitree-G1-Skating-AMP-v0`)
- **Commands:** 0.3–3.5 m/s forward + ±0.8 rad/s turning (curriculum-expanded, same as Eureka)
- **Task rewards:** alive, forward_velocity (exp kernel), upright, height, energy, smoothness, contacts
- **Style reward:** AMP discriminator on 85-dim kinematic state
- **Reset:** near-rest (0–0.5 m/s) — robot must push off, exactly like Eureka

### Active Training Run

| Property | Value |
|---|---|
| Log dir | `logs/rsl_rl/skating_amp/2026-03-11_21-04-26/` |
| Resumed from | Phase 1 `model_1999.pt` |
| Current iteration | ~2100 / 3000 |
| GPU | `cuda:2` (RTX 6000 Ada, ~22 GB used, 100% util) |
| W&B run | [g1_skating / 6q9t7yuf](https://wandb.ai/xupeter-university-of-southern-california/g1_skating/runs/6q9t7yuf) |
| `task_reward_lerp` | 0.5 (equal blend) |
| Discriminator | 2-layer MLP [1024, 512], leaky ReLU, lr=1e-4 |

---

## 7. Eureka Phase 2+ — LLM Reward Search (PAUSED ⏸)

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  eureka_phase2.py  (runs in base Python, no Isaac Sim needed)   │
│                                                                  │
│  for iter in range(N):                                           │
│    1. Call GPT-5.4 → generate compute_reward(env) function      │
│    2. Write reward_fn.py, set EUREKA_REWARD_FN_PATH env var     │
│    3. Launch train.py subprocess (env_isaaclab Python)           │
│       └─ Isaac Lab loads reward fn via eureka_rewards.py         │
│       └─ Resumes from Phase 1 model_1999.pt                      │
│    4. Wait 30s → launch play.py on best-free GPU for eval        │
│    5. Feed metrics back to GPT-5.4 for next iteration            │
│  Keep best policy across all iterations                          │
└─────────────────────────────────────────────────────────────────┘
```

### Eureka Environment (`Unitree-G1-Skating-Eureka-v0`)
- **Commands:** 0.3–3.5 m/s forward + ±0.8 rad/s turning (curriculum-expanded)
- **Fixed structural rewards:** alive, upright, height, energy, smoothness, contacts
- **Task reward:** `eureka_task` — calls LLM-generated `compute_reward()` dynamically
- **Reset:** near-rest (0–0.5 m/s) — robot must push off to earn task reward

### Fitness Function
```
fitness = 0.40 × survival_rate
        + 0.35 × min(speed / 3.5, 1.0)
        + 0.15 × lateral_quality
        + 0.10 × episode_length / 20s
```

---

## 8. Eureka Run History

### Run 1 — 5 iterations, GPT-4o (2026-03-11, FAILED ❌)
All 5 iterations crashed: `ModuleNotFoundError: No module named 'gymnasium'`  
**Root cause:** `_conda_python()` returned `sys.executable` (base Python) instead of `env_isaaclab` Python.  
**Fix:** Auto-detect `env_isaaclab` Python via `conda run` or known miniconda paths.

### Run 2 — GPT-5.4, 10 iterations (2026-03-11, PARTIAL ⚠️)

| Issue | Description | Fix |
|---|---|---|
| `max_tokens` API error | GPT-5.4 requires `max_completion_tokens`, not `max_tokens` | Auto-detect by model name prefix (`gpt-5.*` → `max_completion_tokens`) |
| Temperature error | GPT-5.4 doesn't accept `temperature` parameter | Removed for reasoning models |
| GPU OOM on eval | After training (2048 envs), only 1.5 GB VRAM free on GPU 0 | Auto-select GPU with most free memory; 30s cooldown; reduce eval_envs 64→32 |
| Package not registered | `Unitree-G1-Skating-Eureka-v0` not found by `env_isaaclab` | **OPEN — see section 8** |

**Iter 0 LLM call:** GPT-5.4 generated reward function in **23.1s** (2741 input tokens, 1694 output tokens). The reward function covers: forward speed tracking, push-off rhythm, arm counter-swing, turning, glide continuity, and air-time incentive.

**Iter 0 training:** Failed with `gymnasium.error.NameNotFound: Environment 'Unitree-G1-Skating-Eureka' doesn't exist`. The rollerskating package is not installed in `env_isaaclab`.

---

## 9. Package Install Note

The `env_isaaclab` conda environment has an editable install that points to `~/unitree_rl_lab/source/unitree_rl_lab/` (not the rollerskating workspace). New task files must be synced there manually, or the editable install should be redirected.

**Sync command (run after modifying any task file in rollerskating):**
```bash
DEST=~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/skating
SRC=~/rollerskating/source/unitree_rl_lab/unitree_rl_lab/tasks/skating
rsync -av --exclude="__pycache__" --exclude="*.pyc" $SRC/ $DEST/
rsync -av $SRC/../../../data/ ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/data/
```

All AMP files (`amp_env_cfg.py`, updated `__init__.py`, `rsl_rl_ppo_cfg.py`, `skating_reference_human.npz`) have already been synced as of 2026-03-12.

---

## 10. Reward Engineering Design

### Fixed Structural Rewards (always active, not touched by LLM)
| Term | Weight | Purpose |
|---|---|---|
| `is_alive` | +0.5 | Prevent instant-fall strategy |
| `upright_orientation_l2` | -5.0 | Keep torso vertical |
| `base_height_skating_l2` | -10.0 | Maintain 0.85m bent-knee posture |
| `lateral_velocity_penalty` | -1.0 | Prevent sideways drift |
| `skate_energy` | -2e-5 | Efficient joint usage |
| `skate_action_rate` | -0.05 | Smooth joint trajectories |
| `joint_acc_l2` | -2.5e-7 | Reduce jerk |
| `dof_pos_limits` | -5.0 | Stay in joint limits |
| `joint_deviation_arms` | -0.1 | Natural arm posture |
| `joint_deviation_waist` | -0.5 | Natural waist posture |
| `undesired_contacts` | -1.0 | Only wheels touch floor |

### LLM-Generated Task Reward (`eureka_task`, weight=1.0)
The LLM designs the task-specific component. GPT-5.4 iteration 0 proposed:
- Forward speed tracking (exponential kernel + acceleration bonus)
- Push-off rhythm (phase-clock based alternating contact)
- Arm counter-swing reward (shoulders opposite to stance leg)
- Turning command tracking
- Glide continuity (momentum reward)
- Air-time incentive (one foot off ground during push)

### LLM Prompt Summary
- **System:** Expert reward-function engineer for legged robot RL
- **Context given:** All `env` Python attributes, all 29 joint names, contact sensor API, existing helpers available to import
- **Task description:** What Phase 1 taught; what Phase 2 must achieve (push-off from rest)
- **Feedback loop:** Previous reward code + training reward curve + eval metrics (speed, fall%, episode length)

---

## 11. Completed Bug Fixes (Pre-Eureka, original codebase)

| # | Severity | File | Issue | Fix |
|---|---|---|---|---|
| 1 | **Critical** | `skate_attachment.py` | Visual-only geometry, no collision | Added `CollisionAPI` + wheel physics material |
| 2 | **Critical** | `skating_env_cfg.py` | Phase clock period mismatch (1.0s vs 1.2s) | Set `push_off_rhythm` period to 1.2s |
| 3 | **Critical** | `events.py` | Reset velocity in world-X ignoring robot yaw | Rotate `vx_body` by `dyaw` into world frame |
| 4 | Significant | `amp_obs.py` | Docstring claimed 89-dim, actual 85-dim | Corrected to 85-dim |
| 5 | Significant | `skating_env_cfg.py` | Phase 2 `glide_continuity` min_speed=0.8 with init_vel 0–0.3 | Override min_speed=0.3 |
| 6 | Moderate | `rewards.py` | `edge_contact` only checked contact, not ankle roll angle | Added ankle roll joint position |
| 7 | Moderate | `skating_env_cfg.py` | `CriticCfg` missing `concatenate_terms = True` | Added in `__post_init__` |
| 8 | Moderate | `gen_skating_reference.py` | Single-speed AMP reference (0.8 m/s) | Multi-speed: 0.5–3.0 m/s, 18000×85 frames |
| 9 | **Critical** | `unitree.py` | Init height too low (0.84m), wheels penetrate ground | Raised to 0.876m |
| 10 | Runtime | `skate_attachment.py` | `AddTranslateOp` crash from USD instancing | Idempotency check + safe xform helper |

---

## 12. Bugs Fixed During Eureka Integration (2026-03-11)

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `eureka_phase2.py` | `_conda_python()` returned base Python (no Isaac Lab) | Auto-detect `env_isaaclab` via `conda run` |
| 2 | `eureka_phase2.py` | GPT-5.4 doesn't accept `max_tokens` | Use `max_completion_tokens` for `gpt-5.*`, `o1*`, `o3*`, `o4*` |
| 3 | `eureka_phase2.py` | GPT-5.4 doesn't accept `temperature` | Remove `temperature` for reasoning model prefixes |
| 4 | `eureka_phase2.py` | Eval OOM: training consumes entire GPU, eval launches immediately | 30s sleep + auto-select GPU with most free VRAM |
| 5 | `train.py` | No way to pass absolute checkpoint path across repos | Added `--resume_path` flag |
| 6 | `eureka_phase2.py` | Checkpoint discovery picks `videos/` dir as latest run | Filter to `YYYY-MM-DD_HH-MM-SS` timestamp dirs only |
| 7 | `source/.../robots/g1_29dof/__init__.py` | Old Phase 2–4 gym registrations still present | Replaced with `Eureka-v0` registration only |

---

## 13. Bugs Fixed During AMP Integration (2026-03-12)

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `retarget_video_to_amp.py` | `infer_smpl_genmo` only caught `ImportError`, missing broader API failures | Broadened to `except Exception` |
| 2 | `retarget_video_to_amp.py` | GENMO video estimation scripts not released; used GVHMR (GENMO's video backbone) instead | Subprocess call to GVHMR `demo.py` in dedicated `gvhmr` conda env |
| 3 | `retarget_video_to_amp.py` | GVHMR output `hmr4d_results.pt` parsing needed axis-angle→numpy conversion | Added `torch.load` + rotation conversion logic |
| 4 | `amp_env_cfg.py` | `forward_velocity_tracking_exp` missing `command_name` param | Added `command_name="base_velocity"` |
| 5 | `amp_env_cfg.py` | Used `track_ang_vel_z_exp` which doesn't exist | Replaced with `ang_vel_z_l2` |
| 6 | `robots/g1_29dof/__init__.py` | AMP env registration not visible to `env_isaaclab` (editable install points to different repo) | Manually synced all modified files to `~/unitree_rl_lab/source/` |

---

## 14. Original Phase Training History (Reference)

### Phases 2–4 (superseded by Eureka approach)

| Phase | Run dir | Best ckpt | Lin vel cmd | Key metric |
|---|---|---|---|---|
| Phase 2 | `skating_phase2/2026-03-08_15-39-53/` | `model_3400.pt` | 0.8–2.0 m/s | push_off=4.06, fall=0.1% |
| Phase 3 | `skating_phase3/2026-03-08_17-50-00/` | `model_8000.pt` | stuck at 1.2 m/s | curriculum stall |
| Phase 3b | `skating_phase3b/2026-03-09_11-39-02/` | `model_10000.pt` | 2.1 m/s | curriculum fix |
| Phase 4 | `skating_phase4/2026-03-09_13-38-33/` | `model_14000.pt` | **3.5 m/s (max)** | full speed reached |

Phase 4 achieved 3.5 m/s curriculum maximum with <0.1% fall rate. These checkpoints remain available but are superseded by the Eureka approach which will re-learn the same behaviors using LLM-designed rewards.

---

## 15. How to Run

### AMP training (current)
```bash
cd /home/muchenxu/rollerskating
conda run -n env_isaaclab python scripts/skating/train.py \
  --task Unitree-G1-Skating-AMP-v0 \
  --num_envs 2048 --headless \
  --device cuda:2 \
  --resume_path logs/rsl_rl/skating_amp/2026-03-11_21-04-26/model_2100.pt \
  --experiment_name skating_amp \
  --max_iterations 3000 \
  --logger wandb --log_project_name g1_skating
```

### Generate human reference from a new video
```bash
# Step 1: extract SMPL from video (uses gvhmr conda env)
conda run -n env_isaaclab python scripts/skating/retarget_video_to_amp.py \
  --video_url https://www.youtube.com/shorts/4b-bjwTWPRA \
  --backend gvhmr \
  --gvhmr_repo ~/gvhmr \
  --out source/unitree_rl_lab/unitree_rl_lab/data/skating_reference_human.npz

# Step 2: sync to installed location
rsync -av source/unitree_rl_lab/unitree_rl_lab/data/ \
  ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/data/
```

### Launch Eureka (paused)
```bash
cd /home/muchenxu/rollerskating
export OPENAI_API_KEY=<your_key>
bash scripts/skating/run_eureka.sh
# Custom: MODEL=gpt-5.4 NUM_EUREKA_ITERS=10 TRAIN_ITERS=1500 bash scripts/skating/run_eureka.sh
```

### Phase 1 eval
```bash
cd /home/muchenxu/unitree_rl_lab
conda run -n env_isaaclab python scripts/skating/play.py \
  --task Unitree-G1-Skating-Phase1-v0 \
  --num_envs 16 --headless --video --video_length 600 \
  --load_experiment skating_phase1 \
  --init_vel_x 1.5 --fixed_cmd_vel_x 1.5 \
  --eval_episodes 50
```

---

## 16. Next Steps

1. **[ACTIVE]** Wait for AMP training to complete (iter 3000) — monitor via [W&B](https://wandb.ai/xupeter-university-of-southern-california/g1_skating/runs/6q9t7yuf)
2. **Evaluate AMP policy** — run `play.py` on `Unitree-G1-Skating-AMP-v0` with `model_3000.pt`; check push-off gait emergence vs Phase 1 passive glide
3. **If AMP succeeds** — record demo video via `record_demo.py`; compare skating style to human reference video
4. **If AMP stalls** — tune `task_reward_lerp` (try 0.3 to weight AMP more) or extend to 5000 iterations
5. **Resume Eureka** — run Eureka loop on top of the best AMP checkpoint for speed/turning refinement (LLM reward on top of style reward)

---

## 17. Hardware

- Server: `phe108c-yuewang-02`
- CPU: AMD EPYC 7763, 128 cores (256 logical)
- RAM: 515 GB
- GPUs: **8× NVIDIA RTX 6000 Ada Generation (48 GB each)**
- OS: Ubuntu 22.04, Kernel 5.15.0-131

**Note on GPU sharing:** Server is shared; GPU memory is often nearly full across all 8 GPUs. The Eureka runner auto-selects the GPU with the most free VRAM for evaluation and waits 30s after training to allow memory release. Training uses `cuda:0` by default; override with `DEVICE=cuda:N`.
