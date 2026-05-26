#!/bin/bash
#SBATCH --job-name=moshivis-rag-adapter
#SBATCH --nodes=1
#SBATCH --gres=gpu:4               # 4× A100/H100 on one node
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00            # adjust to your cluster's max
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Slurm batch template: single-node multi-GPU adapter fine-tune of
# MoshiVis + MoshiRAG. Edit the CONFIGURE block below to point at your
# data + checkpoint paths, then submit with `sbatch train_adapter.sbatch`.
#
# Memory footprint (rough): 7B model in bf16 ≈ 14 GB weights + ~5-10 GB
# activations + ~2 GB adapter grad/opt state ≈ ~25 GB per GPU at
# batch_size=8. Comfortably fits on 4× A100 80GB; tight on 4× A100 40GB
# (drop batch_size to 4 and increase grad_accum_steps to compensate).
#
# Multi-node training: this template is single-node. For multi-node DDP,
# duplicate this file and add the standard rendezvous env vars
# (MASTER_ADDR from the first allocated node, MASTER_PORT free,
# torchrun --nnodes=N --node_rank=$SLURM_NODEID). See
# https://pytorch.org/docs/stable/elastic/run.html
#
# ============================================================== CONFIGURE
REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
DATA_JSONL="${DATA_JSONL:-$REPO_ROOT/data/augmented.jsonl}"
AUDIO_CODES_DIR="${AUDIO_CODES_DIR:-$REPO_ROOT/data/audio_codes}"
PRECOMPUTED_IMAGE_KV_DIR="${PRECOMPUTED_IMAGE_KV_DIR:-$REPO_ROOT/data/image_kv}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints/adapter_$(date +%Y%m%d_%H%M%S)}"
KYUTEYE_CONFIG="${KYUTEYE_CONFIG:-$REPO_ROOT/kyuteye_pt/configs/moshika-vis.yaml}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
FREEZE_RECIPE="${FREEZE_RECIPE:-adapters_only}"
# ===========================================================================

set -euo pipefail

# Activate the project's venv. ``uv sync`` should have created
# ``$REPO_ROOT/kyuteye_pt/.venv``; if your cluster uses module load instead,
# replace this with the equivalent module commands.
source "$REPO_ROOT/kyuteye_pt/.venv/bin/activate" 2>/dev/null || {
    echo "WARNING: .venv not found at $REPO_ROOT/kyuteye_pt/.venv -- "
    echo "         relying on the ambient Python environment"
}

mkdir -p logs "$SAVE_DIR"

# torchrun configuration. NPROC is the number of GPUs per node.
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
echo "[$(date)] launching torchrun with NPROC=$NPROC GPUs"
echo "[$(date)] save_dir=$SAVE_DIR"

cd "$REPO_ROOT/kyuteye_pt"

# Useful diagnostics in case the job fails before producing checkpoints.
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}')"

# Single-node launch. ``--standalone`` picks a free rendezvous port.
torchrun \
    --standalone \
    --nproc_per_node=$NPROC \
    scripts/train.py \
    --kyuteye-config "$KYUTEYE_CONFIG" \
    --data "$DATA_JSONL" \
    --audio-codes-dir "$AUDIO_CODES_DIR" \
    --precomputed-image-kv-dir "$PRECOMPUTED_IMAGE_KV_DIR" \
    --save-dir "$SAVE_DIR" \
    --freeze-recipe "$FREEZE_RECIPE" \
    --num-steps $NUM_STEPS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --dtype bfloat16

echo "[$(date)] training complete; checkpoints in $SAVE_DIR"
