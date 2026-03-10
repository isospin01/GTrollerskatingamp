#!/usr/bin/env bash
# Phase 2: Push-off learning with AMP discriminator
# ──────────────────────────────────────────────────
# REQUIRES: Phase 1 checkpoint in logs/rsl_rl/skating_phase1/
# Duration: ~5 hours on A100/H100 with 4096 envs
# Expected outcome: robot learns alternating foot push-off to accelerate
#
# AMP discriminator is trained online to match the reference skating motion.
# WandB logs: reward/task_reward, reward/amp_reward, disc_loss

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==================================================================="
echo "  Humanoid Roller Skating — Phase 2: Push-Off + AMP"
echo "  Repository: ${REPO_ROOT}"
echo "==================================================================="

conda activate env_isaaclab
cd "${REPO_ROOT}"

# Verify Phase 1 checkpoint exists
PHASE1_DIR="logs/rsl_rl/skating_phase1"
if [ ! -d "${PHASE1_DIR}" ]; then
    echo "[ERROR] Phase 1 checkpoint not found at ${PHASE1_DIR}"
    echo "        Run scripts/skating/run_phase1.sh first."
    exit 1
fi

python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase2-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run skating_phase1 \
    --logger wandb \
    --log_project_name g1_skating \
    --experiment_name skating_phase2 \
    --max_iterations 5000 \
    "$@"

echo "[INFO] Phase 2 training complete."
echo "       Checkpoint: logs/rsl_rl/skating_phase2/<latest_run>/model_5000.pt"
