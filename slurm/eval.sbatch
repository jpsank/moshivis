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
    echo "ERROR: set CHECKPOINT to the trained ckpt.pt path (or to a save_dir to auto-resolve)"
    exit 1
fi

# Resolve CHECKPOINT if it points at a save_dir rather than a ckpt.pt.
# The training trainer writes either a ``latest`` symlink (preferred)
# or a ``latest.txt`` fallback for filesystems that don't permit
# symlinks. The orchestration script (``scripts/run_pipeline.sh``)
# passes ``$SAVE_DIR/latest/ckpt.pt`` -- works under symlinks, breaks
# under the .txt fallback. Resolve both cases here so eval doesn't
# fail on Lustre / GPFS clusters.
if [[ ! -f "$CHECKPOINT" ]]; then
    # Try to interpret CHECKPOINT as a save_dir.
    POTENTIAL_SAVE_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
    [[ -d "$POTENTIAL_SAVE_DIR" ]] || POTENTIAL_SAVE_DIR="$CHECKPOINT"
    if [[ -L "$POTENTIAL_SAVE_DIR/latest" ]]; then
        CHECKPOINT="$POTENTIAL_SAVE_DIR/latest/ckpt.pt"
    elif [[ -f "$POTENTIAL_SAVE_DIR/latest.txt" ]]; then
        TARGET_NAME=$(cat "$POTENTIAL_SAVE_DIR/latest.txt" | tr -d '[:space:]')
        CHECKPOINT="$POTENTIAL_SAVE_DIR/$TARGET_NAME/ckpt.pt"
    else
        LATEST_STEP_DIR=$(find "$POTENTIAL_SAVE_DIR" -maxdepth 1 -type d -name "step_*" | sort | tail -1)
        if [[ -n "$LATEST_STEP_DIR" ]]; then
            CHECKPOINT="$LATEST_STEP_DIR/ckpt.pt"
        fi
    fi
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: could not resolve checkpoint -- looked at:"
    echo "  - the literal path"
    echo "  - <parent>/latest symlink"
    echo "  - <parent>/latest.txt fallback"
    echo "  - highest <parent>/step_*/"
    exit 1
fi
echo "[eval] using checkpoint: $CHECKPOINT"

# Offline HF mode: match the training sbatch so the ARC encoder (if
# the eval config has rag.enabled=true) doesn't try to fetch the
# tokenizer / weights from the network. Run ``scripts/hpc_setup.sh``
# from a login node once to populate HF_HOME.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

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
