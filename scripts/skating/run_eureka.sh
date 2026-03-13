#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eureka.sh — Launch the Eureka roller-skating reward search (Phase 2+)
#
# Supported LLM models (set MODEL below):
#   OpenAI (OPENAI_API_KEY):
#     MODEL=gpt-5.4-pro                 →  GPT-5.4 Thinking [DEFAULT — best for code]
#     MODEL=gpt-5.4                     →  GPT-5.4 standard  (faster, cheaper)
#     MODEL=o3-mini                     →  o3-mini
#     MODEL=gpt-4o                      →  GPT-4o
#   Anthropic (ANTHROPIC_API_KEY):
#     MODEL=claude-3-7-sonnet-20250219  →  Claude 3.7 Sonnet
#   Google Gemini (GOOGLE_API_KEY):
#     MODEL=gemini-2.0-flash            →  Gemini 2.0 Flash
#
# Prerequisites:
#   1.  pip install anthropic           (or openai / google-generativeai)
#   2.  export ANTHROPIC_API_KEY=<key>  (or OPENAI_API_KEY / GOOGLE_API_KEY)
#   3.  Phase 1 checkpoint must exist   (auto-detected from ~/unitree_rl_lab/)
#
# Usage:
#   bash scripts/skating/run_eureka.sh
#   bash scripts/skating/run_eureka.sh --num_eureka_iters 8 --train_iters 2000
#   MODEL=gpt-4o bash scripts/skating/run_eureka.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Configurable defaults ─────────────────────────────────────────────────────
PHASE1_RUN="${PHASE1_RUN:-latest}"
NUM_EUREKA_ITERS="${NUM_EUREKA_ITERS:-5}"
TRAIN_ITERS="${TRAIN_ITERS:-1500}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
NUM_ENVS="${NUM_ENVS:-2048}"
EVAL_ENVS="${EVAL_ENVS:-64}"
MODEL="${MODEL:-gpt-5.4-pro}"   # GPT-5.4 Thinking (best for code)
DEVICE="${DEVICE:-cuda:0}"      # GPU for training
EVAL_DEVICE="${EVAL_DEVICE:-}"  # GPU for eval (empty = auto-select most-free GPU)
EVAL_SLEEP="${EVAL_SLEEP:-30}"  # seconds to wait after training before eval

# Pass any extra CLI args through (e.g. --api_key, --resume_iter)
EXTRA_ARGS=("$@")

# ── Infer provider and check API key ─────────────────────────────────────────
if [[ "$MODEL" == claude* ]]; then
    PROVIDER="anthropic"
    KEY_VAR="ANTHROPIC_API_KEY"
    PKG="anthropic"
elif [[ "$MODEL" == gemini* ]]; then
    PROVIDER="google"
    KEY_VAR="GOOGLE_API_KEY"
    PKG="google-generativeai"
else
    # OpenAI: gpt-4o, gpt-5.4, gpt-5.4-pro, o3-mini, o1, etc.
    PROVIDER="openai"
    KEY_VAR="OPENAI_API_KEY"
    PKG="openai"
fi

if [[ -z "${!KEY_VAR:-}" ]]; then
    echo "ERROR: ${KEY_VAR} environment variable is not set."
    echo "  export ${KEY_VAR}=<your_api_key>"
    exit 1
fi

if ! python -c "import ${PKG//-/_}" 2>/dev/null; then
    echo "ERROR: '${PKG}' Python package not found."
    echo "  pip install ${PKG}"
    exit 1
fi

# ── Log header ────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "  Eureka Roller-Skating Reward Search"
echo "  $(date)"
echo "  Phase 1 run  : $PHASE1_RUN"
echo "  Eureka iters : $NUM_EUREKA_ITERS"
echo "  Train iters  : $TRAIN_ITERS per Eureka iter"
echo "  Eval episodes: $EVAL_EPISODES"
echo "  LLM model    : $MODEL"
echo "  Device       : $DEVICE"
echo "======================================================================"

# ── Run Eureka ────────────────────────────────────────────────────────────────
EVAL_DEVICE_ARGS=()
[[ -n "$EVAL_DEVICE" ]] && EVAL_DEVICE_ARGS=(--eval_device "$EVAL_DEVICE")

python scripts/skating/eureka_phase2.py \
    --phase1_run       "$PHASE1_RUN" \
    --num_eureka_iters "$NUM_EUREKA_ITERS" \
    --train_iters      "$TRAIN_ITERS" \
    --eval_episodes    "$EVAL_EPISODES" \
    --num_envs         "$NUM_ENVS" \
    --eval_envs        "$EVAL_ENVS" \
    --model            "$MODEL" \
    --device           "$DEVICE" \
    --eval_sleep       "$EVAL_SLEEP" \
    --headless \
    --record_best \
    "${EVAL_DEVICE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "======================================================================"
echo "  Eureka search finished.  Best policy: eureka_output/best/reward_fn.py"
echo "  Demo video:  eureka_output/iter_*/  (recorded by record_demo.py)"
echo "======================================================================"
