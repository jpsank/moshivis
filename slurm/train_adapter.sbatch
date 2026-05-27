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
# Prerequisites:
#   1. Run ``scripts/hpc_setup.sh`` ONCE on a login node with internet
#      access. It installs the ``[arc]`` extra (xformers -- required by
#      the ARC encoder), pre-caches the ARC tokenizer (a GATED repo
#      requiring HF login), and optionally pre-caches MoshiVis + ARC
#      weights via ``HF_HOME``.
#   2. Point ``HF_HOME`` at shared storage so compute nodes see the
#      cache populated by step 1. Set in the env block below.
#
# Memory footprint per GPU (rough, bf16 throughout):
#   * Frozen 7B LM:                14 GB
#   * Vision encoder (frozen):      1 GB
#   * ARC encoder (3 GB weights):
#       - frozen:                   3 GB
#       - finetuned (default):    ~15 GB (weights + Adam state + activations)
#   * Activations / grad state:    5-10 GB at batch_size=8
#   * Adapter Adam state:          ~2 GB
#   Totals at batch_size=8:
#     adapters_only, ARC frozen:                 ~25-30 GB (40GB OK)
#     adapters_only, ARC trainable (default):    ~40-45 GB (tight on 40GB)
#     full_lm:                                   ~70 GB+  (80GB only)
#
# If you OOM on 40GB cards: drop batch_size to 4 and bump
# grad_accum_steps to compensate, or set the ARC conditioner's
# ``finetune: false`` in the YAML to keep it frozen during the adapter
# phase.
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

# Offline-friendly HuggingFace mode: rely entirely on the cache populated
# by ``scripts/hpc_setup.sh`` from the login node. Avoids network calls
# from compute nodes that don't have internet (the ARC encoder's
# AutoTokenizer + load_weights would otherwise hang / fail at startup).
# ``HF_HOME`` must point at a directory the compute node can read
# (typically shared scratch or project storage).
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Activate the project's venv. ``scripts/hpc_setup.sh`` creates this with
# the ``[arc]`` extra installed (xformers); without that, the ARC
# encoder will raise ImportError at training startup.
source "$REPO_ROOT/kyuteye_pt/.venv/bin/activate" 2>/dev/null || {
    echo "WARNING: .venv not found at $REPO_ROOT/kyuteye_pt/.venv. Run"
    echo "         ``scripts/hpc_setup.sh`` from a login node first."
    echo "         Falling back to ambient Python -- xformers may be missing."
}

# Verify the ARC encoder's xformers dependency is actually available. The
# import takes ~1s but failing here is a much better diagnostic than the
# torchrun stack trace 30s later.
python -c "import xformers" 2>/dev/null || {
    echo "WARNING: xformers not importable. The ARC encoder requires it."
    echo "         Run ``scripts/hpc_setup.sh`` or ``uv sync --extra arc``."
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
