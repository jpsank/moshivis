#!/bin/bash
# One-command pipeline launcher: submit preprocessing → training → eval
# as Slurm jobs with afterok dependencies. Each downstream job starts
# only if its predecessor finishes successfully.
#
# Usage:
#
#     # From your cluster's login node:
#     cd $REPO_ROOT
#     MIMI_WEIGHT=/path/to/mimi.safetensors \
#     DATA_JSONL=$REPO_ROOT/data/augmented.jsonl \
#     EVAL_DATA=$REPO_ROOT/data/eval.jsonl \
#     ./scripts/run_pipeline.sh
#
# Inputs (env vars):
#   MIMI_WEIGHT (required)        Path to Mimi safetensors checkpoint
#   DATA_JSONL (required)         Training JSONL from ssvd/rag_augment.py
#   EVAL_DATA (required)          Eval split JSONL (smaller held-out)
#   REPO_ROOT (default $HOME/moshivis)
#   AUDIO_CODES_DIR               Where preprocessing writes audio codes
#   EVAL_AUDIO_CODES_DIR          Same, for the eval split
#   SAVE_DIR                      Training checkpoint destination
#   TTS_BACKEND (default silence) silence | coqui_xtts
#   TTS_USER_REF / TTS_MOSHI_REF  Required when TTS_BACKEND=coqui_xtts
#   NUM_STEPS, BATCH_SIZE, LEARNING_RATE, FREEZE_RECIPE  (training knobs)
#   SKIP_PREPROCESS=1             Skip job 1 (audio codes already exist)
#   SKIP_EVAL=1                   Skip job 3 (just train)
#
# Outputs:
#   $SAVE_DIR/                    Training checkpoints
#   logs/                         Per-job stdout/stderr (one file per job)
#   $REPO_ROOT/.pipeline_jobs     Most recent job id chain for monitoring
#
# Monitor with:  squeue -u $USER
# Cancel chain:  scancel $(cat $REPO_ROOT/.pipeline_jobs)

set -euo pipefail

# Defaults.
REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
AUDIO_CODES_DIR="${AUDIO_CODES_DIR:-$REPO_ROOT/data/audio_codes}"
EVAL_AUDIO_CODES_DIR="${EVAL_AUDIO_CODES_DIR:-$REPO_ROOT/data/eval_audio_codes}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints/run_$(date +%Y%m%d_%H%M%S)}"

# Required inputs.
: "${MIMI_WEIGHT:?set MIMI_WEIGHT to the path of your mimi safetensors file}"
: "${DATA_JSONL:?set DATA_JSONL to the training data JSONL}"
: "${EVAL_DATA:?set EVAL_DATA to the eval split JSONL (or pass SKIP_EVAL=1)}"

cd "$REPO_ROOT"
mkdir -p logs "$AUDIO_CODES_DIR" "$EVAL_AUDIO_CODES_DIR" "$SAVE_DIR"

JOB_IDS=()
echo "[pipeline] repo_root=$REPO_ROOT"
echo "[pipeline] save_dir=$SAVE_DIR"

# ---- Job 1: training data audio preprocessing ---------------------------
if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
    echo "[pipeline] submitting preprocess (training data) ..."
    PREPROCESS_JOB=$(
        MIMI_WEIGHT="$MIMI_WEIGHT" \
        DATA_JSONL="$DATA_JSONL" \
        AUDIO_CODES_DIR="$AUDIO_CODES_DIR" \
        TTS_BACKEND="${TTS_BACKEND:-silence}" \
        TTS_USER_REF="${TTS_USER_REF:-}" \
        TTS_MOSHI_REF="${TTS_MOSHI_REF:-}" \
        sbatch --parsable --job-name=mv-rag-prep-train slurm/preprocess_audio.sbatch
    )
    JOB_IDS+=("$PREPROCESS_JOB")
    DEPENDENCY="--dependency=afterok:$PREPROCESS_JOB"
    echo "[pipeline]   training preprocess job: $PREPROCESS_JOB"
else
    DEPENDENCY=""
    echo "[pipeline] skipping training preprocess (SKIP_PREPROCESS=1)"
fi

# ---- Job 1b: eval data audio preprocessing -----------------------------
# Eval preprocessing runs in parallel with training preprocessing (no
# dependency between them) but training waits for both via the chain.
if [[ "${SKIP_PREPROCESS:-0}" != "1" && "${SKIP_EVAL:-0}" != "1" ]]; then
    echo "[pipeline] submitting preprocess (eval data) ..."
    EVAL_PREPROCESS_JOB=$(
        MIMI_WEIGHT="$MIMI_WEIGHT" \
        DATA_JSONL="$EVAL_DATA" \
        AUDIO_CODES_DIR="$EVAL_AUDIO_CODES_DIR" \
        TTS_BACKEND="${TTS_BACKEND:-silence}" \
        TTS_USER_REF="${TTS_USER_REF:-}" \
        TTS_MOSHI_REF="${TTS_MOSHI_REF:-}" \
        sbatch --parsable --job-name=mv-rag-prep-eval slurm/preprocess_audio.sbatch
    )
    JOB_IDS+=("$EVAL_PREPROCESS_JOB")
    # Training waits on BOTH training and eval preprocessing.
    DEPENDENCY="--dependency=afterok:$PREPROCESS_JOB:$EVAL_PREPROCESS_JOB"
    echo "[pipeline]   eval preprocess job: $EVAL_PREPROCESS_JOB"
fi

# ---- Job 2: training ----------------------------------------------------
echo "[pipeline] submitting training ..."
TRAIN_JOB=$(
    DATA_JSONL="$DATA_JSONL" \
    AUDIO_CODES_DIR="$AUDIO_CODES_DIR" \
    SAVE_DIR="$SAVE_DIR" \
    NUM_STEPS="${NUM_STEPS:-10000}" \
    BATCH_SIZE="${BATCH_SIZE:-8}" \
    LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
    FREEZE_RECIPE="${FREEZE_RECIPE:-adapters_only}" \
    sbatch --parsable --job-name=mv-rag-train $DEPENDENCY slurm/train_adapter.sbatch
)
JOB_IDS+=("$TRAIN_JOB")
echo "[pipeline]   train job: $TRAIN_JOB"

# ---- Job 3: eval --------------------------------------------------------
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    echo "[pipeline] submitting eval ..."
    EVAL_JOB=$(
        EVAL_DATA="$EVAL_DATA" \
        EVAL_AUDIO_CODES_DIR="$EVAL_AUDIO_CODES_DIR" \
        CHECKPOINT="$SAVE_DIR/latest/ckpt.pt" \
        BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
        sbatch --parsable --job-name=mv-rag-eval \
               --dependency=afterok:$TRAIN_JOB slurm/eval.sbatch
    )
    JOB_IDS+=("$EVAL_JOB")
    echo "[pipeline]   eval job: $EVAL_JOB"
fi

# Persist the job chain so it can be cancelled atomically if needed.
printf '%s\n' "${JOB_IDS[@]}" > "$REPO_ROOT/.pipeline_jobs"

echo ""
echo "[pipeline] all jobs submitted ($(date))"
echo "[pipeline] chain: ${JOB_IDS[*]}"
echo "[pipeline] monitor:    squeue -u \$USER"
echo "[pipeline] cancel all: scancel \$(cat $REPO_ROOT/.pipeline_jobs)"
echo "[pipeline] training output: $SAVE_DIR/"
echo "[pipeline] per-job logs: logs/"
