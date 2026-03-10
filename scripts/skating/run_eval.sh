#!/usr/bin/env bash
# Evaluate the final trained skating policy and record a demo video.
# ──────────────────────────────────────────────────────────────────
# Produces:
#   - logs/rsl_rl/skating_phase3/eval_metrics.csv     (quantitative metrics)
#   - logs/rsl_rl/skating_phase3/videos/eval/*.mp4    (demo video)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

conda activate env_isaaclab
cd "${REPO_ROOT}"

# ── 1. Visual evaluation with video recording (16 envs) ──────────────────────
echo "[INFO] Running visual evaluation (16 envs, video recording)..."
python scripts/skating/play.py \
    --task Unitree-G1-Skating-Phase3-v0 \
    --num_envs 16 \
    --load_run skating_phase3 \
    --video \
    --video_length 600 \
    --eval_episodes 20

# ── 2. Quantitative evaluation (256 envs, headless) ──────────────────────────
echo "[INFO] Running headless quantitative evaluation (256 envs)..."
python scripts/skating/play.py \
    --task Unitree-G1-Skating-Phase3-v0 \
    --num_envs 256 \
    --headless \
    --load_run skating_phase3 \
    --eval_episodes 200

echo ""
echo "[INFO] Evaluation complete."
echo "  Video:   logs/rsl_rl/skating_phase3/videos/eval/"
echo "  Metrics: logs/rsl_rl/skating_phase3/eval_metrics.csv"
