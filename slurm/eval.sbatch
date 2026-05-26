#!/bin/bash
#SBATCH --job-name=moshivis-rag-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Held-out perplexity evaluation. Runs after training in the
# orchestrated pipeline (``scripts/run_pipeline.sh``). Single-GPU is
# enough -- no gradients, no optimizer state, just a forward pass.
#
# ============================================================== CONFIGURE
REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
EVAL_DATA="${EVAL_DATA:-$REPO_ROOT/data/eval.jsonl}"
EVAL_AUDIO_CODES_DIR="${EVAL_AUDIO_CODES_DIR:-$REPO_ROOT/data/eval_audio_codes}"
PRECOMPUTED_IMAGE_KV_DIR="${PRECOMPUTED_IMAGE_KV_DIR:-$REPO_ROOT/data/image_kv}"
CHECKPOINT="${CHECKPOINT:-}"  # required -- path to ckpt.pt
KYUTEYE_CONFIG="${KYUTEYE_CONFIG:-$REPO_ROOT/kyuteye_pt/configs/moshika-vis.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-}"
# ===========================================================================

set -euo pipefail

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: set CHECKPOINT to the trained ckpt.pt path"
    exit 1
fi

source "$REPO_ROOT/kyuteye_pt/.venv/bin/activate" 2>/dev/null || \
    echo "WARN: .venv not found; relying on ambient Python"
mkdir -p logs

cd "$REPO_ROOT/kyuteye_pt"
nvidia-smi -L

MAX_BATCHES_ARG=""
if [[ -n "$MAX_BATCHES" ]]; then
    MAX_BATCHES_ARG="--max-batches $MAX_BATCHES"
fi

python scripts/eval.py \
    --kyuteye-config "$KYUTEYE_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data "$EVAL_DATA" \
    --audio-codes-dir "$EVAL_AUDIO_CODES_DIR" \
    --precomputed-image-kv-dir "$PRECOMPUTED_IMAGE_KV_DIR" \
    --batch-size $BATCH_SIZE \
    --dtype bfloat16 \
    $MAX_BATCHES_ARG

echo "[$(date)] eval complete"
