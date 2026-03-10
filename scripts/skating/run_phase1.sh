#!/usr/bin/env bash
# Phase 1: Gliding balance on roller skates (no AMP, pure PPO)
# ─────────────────────────────────────────────────────────────
# Duration: ~2 hours on a single A100/H100 GPU with 4096 envs
# Expected outcome: robot stays upright and glides without falling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==================================================================="
echo "  Humanoid Roller Skating — Phase 1: Gliding Balance"
echo "  Repository: ${REPO_ROOT}"
echo "==================================================================="

conda activate env_isaaclab

cd "${REPO_ROOT}"

# Generate reference motion if not yet done
if [ ! -f "source/unitree_rl_lab/unitree_rl_lab/data/skating_reference.npz" ]; then
    echo "[INFO] Generating skating reference motion..."
    python scripts/skating/gen_skating_reference.py --fps 50 --cycles 50
fi

# Train Phase 1
python scripts/skating/train.py \
    --task Unitree-G1-Skating-Phase1-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name g1_skating \
    --experiment_name skating_phase1 \
    --max_iterations 2000 \
    "$@"

echo "[INFO] Phase 1 training complete."
echo "       Check WandB project 'g1_skating' for training curves."
echo "       Checkpoint: logs/rsl_rl/skating_phase1/<latest_run>/model_2000.pt"
