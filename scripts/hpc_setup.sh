#!/bin/bash
# One-time HPC setup: install deps + pre-cache HuggingFace assets so
# compute nodes (which usually have no internet) can train and serve
# offline.
#
# Run this ONCE from a login node with internet access. The compute
# nodes pick up the resulting venv + cache via shared storage.
#
# What it does:
#   1. ``uv sync --extra arc --extra omni`` -- installs xformers (ARC
#      encoder needs it) and the omni-stack optional deps.
#   2. Pre-downloads the ARC encoder tokenizer (Llama-3.2-3B-Instruct
#      by default) so ``AutoTokenizer.from_pretrained`` doesn't hit the
#      network at training startup.
#   3. Pre-downloads ARC encoder weights if ``ARC_HF_REPO`` is set
#      (e.g. ``kyutai/moshika-rag-pytorch-bf16``).
#   4. Pre-downloads the MoshiVis weights from ``MOSHIVIS_HF_REPO``
#      (e.g. ``kyutai/moshika-vis-pytorch-bf16``) and the Mimi codec.
#
# Inputs (env vars):
#   REPO_ROOT            (default $HOME/moshivis)
#   HF_HOME              (recommended: set to shared storage on the
#                        cluster so compute nodes see the cache)
#   HF_TOKEN             (required for gated repos like meta-llama/*)
#   MOSHIVIS_HF_REPO     (e.g. kyutai/moshika-vis-pytorch-bf16)
#   ARC_HF_REPO          (e.g. kyutai/moshika-rag-pytorch-bf16)
#   ARC_TOKENIZER        (default meta-llama/Llama-3.2-3B-Instruct)
#
# After running this script, your training sbatch can rely on:
#   * The venv at $REPO_ROOT/kyuteye_pt/.venv being complete
#     (including xformers).
#   * Setting HF_HUB_OFFLINE=1 on the compute node (the sbatches do
#     this) -- all weights load from the shared cache.
#
# Usage:
#   HF_HOME=/scratch/$USER/hf_cache \
#   HF_TOKEN=$(cat ~/.huggingface_token) \
#   MOSHIVIS_HF_REPO=kyutai/moshika-vis-pytorch-bf16 \
#   ARC_HF_REPO=kyutai/moshika-rag-pytorch-bf16 \
#   ./scripts/hpc_setup.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
ARC_TOKENIZER="${ARC_TOKENIZER:-meta-llama/Llama-3.2-3B-Instruct}"

if [[ -z "${HF_HOME:-}" ]]; then
    echo "WARNING: HF_HOME is not set; defaulting to ~/.cache/huggingface."
    echo "         For multi-node HPC, set HF_HOME to shared storage so"
    echo "         compute nodes see the same cache."
fi

echo "[$(date)] hpc_setup: installing deps via uv"
cd "$REPO_ROOT/kyuteye_pt"
# Idempotent: re-run is fast, only pulls deltas. ``--extra arc`` is the
# critical bit -- without it xformers won't be installed and the ARC
# encoder will raise ImportError at training startup.
uv sync --extra arc --extra omni

echo "[$(date)] hpc_setup: pre-caching ARC tokenizer (${ARC_TOKENIZER})"
# The ARC encoder calls ``AutoTokenizer.from_pretrained`` at
# ``ArcEncoderConditioner.__init__`` time, which hits HF if not cached.
# Pre-download here so compute nodes only read from the local cache.
#
# NOTE: ``meta-llama/Llama-3.2-3B-Instruct`` is a GATED repo. You must
# (a) accept its license on the HuggingFace web UI, and
# (b) ``huggingface-cli login`` (or set HF_TOKEN) on this login node.
uv run python -c "
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        '${ARC_TOKENIZER}',
        allow_patterns=['tokenizer*', 'special_tokens*', 'tokenizer.model'],
    )
    print('[hpc_setup] tokenizer cached')
except Exception as e:
    print(f'[hpc_setup] tokenizer cache FAILED: {e}', file=sys.stderr)
    print('Did you accept the license on https://huggingface.co/${ARC_TOKENIZER}', file=sys.stderr)
    print('and run ``huggingface-cli login`` (or set HF_TOKEN)?', file=sys.stderr)
    sys.exit(1)
"

if [[ -n "${ARC_HF_REPO:-}" ]]; then
    echo "[$(date)] hpc_setup: pre-caching ARC encoder weights (${ARC_HF_REPO})"
    uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('${ARC_HF_REPO}', 'model.safetensors')
print('[hpc_setup] ARC encoder weights cached')
"
else
    echo "[$(date)] hpc_setup: ARC_HF_REPO not set; skipping ARC weight pre-cache."
    echo "    Set ARC_HF_REPO (e.g. kyutai/moshika-rag-pytorch-bf16) or"
    echo "    point the YAML's rag.conditioners.*.weights_path at a local file."
fi

if [[ -n "${MOSHIVIS_HF_REPO:-}" ]]; then
    echo "[$(date)] hpc_setup: pre-caching MoshiVis weights (${MOSHIVIS_HF_REPO})"
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MOSHIVIS_HF_REPO}',
    allow_patterns=['*.safetensors', '*.json', 'tokenizer*'],
)
print('[hpc_setup] MoshiVis weights cached')
"
fi

echo "[$(date)] hpc_setup: complete."
echo "Compute nodes should now set ``HF_HUB_OFFLINE=1`` to avoid network calls."
