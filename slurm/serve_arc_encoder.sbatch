#!/bin/bash
#SBATCH --job-name=arc-encoder-service
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # 1 GPU is enough -- encoder is small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00            # long-running service; adjust per cluster
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Run the ARC encoder as a standalone HTTP service. The main inference
# server points at this with ``--omni-arc-encoder-url`` /
# ``REFERENCE_ENCODER_URL``. Use this when you want the LM and encoder
# on different GPUs (or different hosts), e.g. for batched inference
# scaling. For single-process inference, the default
# ``--omni-arc-encoder-mode=auto`` already calls the encoder in-process
# via ``kyuteye.omni.arc_encoder_local`` -- no separate service needed.
#
# Prerequisites: same as train_adapter.sbatch. Run
# ``scripts/hpc_setup.sh`` once on a login node to install xformers and
# pre-cache the ARC tokenizer and (optionally) ARC weights.
#
# ============================================================== CONFIGURE
REPO_ROOT="${REPO_ROOT:-$HOME/moshivis}"
KYUTEYE_CONFIG="${KYUTEYE_CONFIG:-$REPO_ROOT/kyuteye_pt/configs/moshika-vis.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8089}"
CONDITIONER_NAME="${CONDITIONER_NAME:-reference_with_time}"
# ===========================================================================

set -euo pipefail

# Offline HF mode -- relies on pre-populated HF_HOME on shared storage.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source "$REPO_ROOT/kyuteye_pt/.venv/bin/activate" 2>/dev/null || {
    echo "WARNING: .venv not found at $REPO_ROOT/kyuteye_pt/.venv. Run"
    echo "         ``scripts/hpc_setup.sh`` from a login node first."
}
mkdir -p logs

cd "$REPO_ROOT/kyuteye_pt"
nvidia-smi -L

# Surface the URL clients should point at -- the assigned node's
# hostname + chosen port. Useful in the log so a client job started
# under the same allocation knows where to send requests.
HOSTNAME_FQDN="$(hostname -f)"
echo "[$(date)] ARC encoder service starting on http://${HOSTNAME_FQDN}:${PORT}/embed"
echo "[$(date)] clients: export REFERENCE_ENCODER_URL=http://${HOSTNAME_FQDN}:${PORT}"

python scripts/serve_arc_encoder.py \
    --kyuteye-config "$KYUTEYE_CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --conditioner-name "$CONDITIONER_NAME"
