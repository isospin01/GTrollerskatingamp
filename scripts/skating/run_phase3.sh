#!/usr/bin/env bash
# Phase 3: Full velocity + turning commands (lean-to-steer)
# ──────────────────────────────────────────────────────────
# REQUIRES: Phase 2 checkpoint in logs/rsl_rl/skating_phase2/
# Duration: ~10 hours on A100/H100 with 4096 envs
# Expected outcome: robot tracks arbitrary velocity + angular velocity commands
#                   using body lean and ankle roll edge control.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==================================================================="
echo "  Humanoid Roller Skating — Phase 3: Full Velocity + Turning"
echo "  Repository: ${REPO_ROOT}"
echo "==================================================================="

conda activate env_isaaclab
cd "${REPO_ROOT}"

# Verify Phase 2 checkpoint exists
PHASE2_DIR="logs/rsl_rl/skating_phase2"
if [ ! -d "${PHASE2_DIR}" ]; then
    echo "[ERROR] Phase 2 checkpoint not found at ${PHASE2_DIR}"
    echo "        Run scripts/skating/run_phase2.sh first."
    exit 1
fi

python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase3-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run skating_phase2 \
    --logger wandb \
    --log_project_name g1_skating \
    --experiment_name skating_phase3 \
    --max_iterations 10000 \
    "$@"

echo "[INFO] Phase 3 training complete."
echo "       Run evaluation: bash scripts/skating/run_eval.sh"
