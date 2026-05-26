#!/bin/bash
#SBATCH --job-name=moshivis-rag-preprocess
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One-shot audio preprocessing job: synthesizes TTS audio for every
# turn in the training JSONL, encodes via Mimi, saves per-example .pt
# files in $AUDIO_CODES_DIR.
#
# Single GPU is enough -- the model loaded is just the Mimi codec
# (a few hundred MB) plus the TTS. Use array jobs (`#SBATCH --array=0-N`)
# to parallelize across dataset shards by setting --start-idx /
# --end-idx from $SLURM_ARRAY_TASK_ID.
#
# ============================================================== CONFIGURE
REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
DATA_JSONL="${DATA_JSONL:-$REPO_ROOT/data/augmented.jsonl}"
AUDIO_CODES_DIR="${AUDIO_CODES_DIR:-$REPO_ROOT/data/audio_codes}"
MIMI_WEIGHT="${MIMI_WEIGHT:-}"  # required -- path to mimi safetensors
TTS_BACKEND="${TTS_BACKEND:-silence}"  # silence|coqui_xtts
TTS_USER_REF="${TTS_USER_REF:-}"  # required when TTS_BACKEND=coqui_xtts
TTS_MOSHI_REF="${TTS_MOSHI_REF:-}"
# ===========================================================================

set -euo pipefail
mkdir -p logs "$AUDIO_CODES_DIR"

if [[ -z "$MIMI_WEIGHT" ]]; then
    echo "ERROR: set MIMI_WEIGHT to the path of your mimi safetensors file"
    echo "  (e.g. \$HF_HOME/.../tokenizer-e351c8d8-checkpoint125.safetensors)"
    exit 1
fi

source "$REPO_ROOT/kyuteye_pt/.venv/bin/activate" 2>/dev/null || \
    echo "WARN: .venv not found; relying on ambient Python"

cd "$REPO_ROOT/kyuteye_pt"

# Optional sharding for array jobs.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ -n "${SHARDS:-}" ]]; then
    # User must set SHARDS and TOTAL_EXAMPLES env vars when using array jobs.
    SHARD_SIZE=$(( TOTAL_EXAMPLES / SHARDS + 1 ))
    START=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
    END=$(( START + SHARD_SIZE ))
    echo "[$(date)] shard $SLURM_ARRAY_TASK_ID/$SHARDS -> [$START, $END)"
    EXTRA_ARGS="--start-idx $START --end-idx $END"
else
    EXTRA_ARGS=""
fi

# Build the TTS argument set.
if [[ "$TTS_BACKEND" == "coqui_xtts" ]]; then
    if [[ -z "$TTS_USER_REF" || -z "$TTS_MOSHI_REF" ]]; then
        echo "ERROR: coqui_xtts needs TTS_USER_REF and TTS_MOSHI_REF"
        exit 1
    fi
    TTS_ARGS="--tts coqui_xtts --tts-user-ref $TTS_USER_REF --tts-moshi-ref $TTS_MOSHI_REF"
else
    TTS_ARGS="--tts $TTS_BACKEND"
fi

nvidia-smi -L

python -m kyuteye.training.audio_preprocess \
    --input "$DATA_JSONL" \
    --output-dir "$AUDIO_CODES_DIR" \
    --mimi-weight "$MIMI_WEIGHT" \
    --device cuda \
    --skip-existing \
    $TTS_ARGS \
    $EXTRA_ARGS

echo "[$(date)] preprocessing complete; codes in $AUDIO_CODES_DIR"
