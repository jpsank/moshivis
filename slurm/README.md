# Slurm training templates

This directory contains batch script templates for running a combined
MoshiVis + MoshiRAG fine-tune on a Slurm-managed HPC cluster.

## `train_adapter.sbatch`

Single-node multi-GPU adapter fine-tune. Defaults to 4 GPUs, 24-hour
wall time, the `adapters_only` freeze recipe (trains only the new RAG
conditioners + ARC encoder + bridge projector while the 7B LM backbone
stays frozen). Output checkpoints to `$REPO_ROOT/checkpoints/adapter_<timestamp>/`.

Submit:

```bash
cd /path/to/moshivis
sbatch slurm/train_adapter.sbatch
```

Override defaults via environment variables (recognized by the script):

| Variable | Default | Purpose |
| --- | --- | --- |
| `REPO_ROOT` | `$HOME/moshivis` | Where the repo is cloned |
| `DATA_JSONL` | `$REPO_ROOT/data/augmented.jsonl` | Training data |
| `AUDIO_CODES_DIR` | `$REPO_ROOT/data/audio_codes` | Per-example Mimi codes |
| `PRECOMPUTED_IMAGE_KV_DIR` | `$REPO_ROOT/data/image_kv` | Cached image cross-attention KV |
| `SAVE_DIR` | timestamped under `checkpoints/` | Checkpoint destination |
| `KYUTEYE_CONFIG` | `kyuteye_pt/configs/moshika-vis.yaml` | Model config YAML |
| `NUM_STEPS` | `10000` | Training steps |
| `BATCH_SIZE` | `8` | Per-GPU batch size |
| `LEARNING_RATE` | `1e-4` | AdamW peak LR |
| `FREEZE_RECIPE` | `adapters_only` | One of `adapters_only`, `adapters_plus_xa`, `full_lm` |
| `NPROC` | auto (`nvidia-smi -L \| wc -l`) | GPUs per node |

Example with overrides:

```bash
DATA_JSONL=/scratch/$USER/rag_data/v3.jsonl \
NUM_STEPS=20000 \
BATCH_SIZE=4 \
sbatch --gres=gpu:8 --time=48:00:00 slurm/train_adapter.sbatch
```

## Multi-node training

The single-node template is a starting point. For multi-node DDP across
several nodes, edit a copy with the standard torchrun multi-node
rendezvous:

```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1

# First allocated node hosts the rendezvous.
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --label \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=8 \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/train.py ...
```

The trainer's `init_distributed()` reads `WORLD_SIZE`, `RANK`,
`LOCAL_RANK` from the environment, so torchrun's standard setup works
unchanged.

## Common gotchas

* **Module not found errors at job start**: the venv path in
  `train_adapter.sbatch` assumes `uv sync` was run inside `kyuteye_pt/`.
  If your cluster uses `module load python/...` + `pip install`
  instead, replace the `source .venv/bin/activate` line.
* **`OSError: ... .safetensors`**: the trainer downloads model weights
  from HuggingFace via `huggingface_hub`. On clusters with no direct
  internet access, pre-download to a shared filesystem and pass
  `--moshi-weight /path/to/model_pt.safetensors`.
* **NCCL hangs at startup**: usually a network config issue. Try
  `export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` to surface the
  rendezvous diagnostics. Many clusters need `NCCL_SOCKET_IFNAME` set
  to the right interface.
* **OOM at first step**: drop `BATCH_SIZE` (e.g. to 4) and bump
  `GRAD_ACCUM_STEPS` to keep effective batch the same. The collator
  doesn't yet support sequence packing; long examples may be the cause
  -- try lowering `--max-seq-len 768`.
* **Checkpoint resume failing**: the trainer writes a `latest` symlink
  pointing to the most recent `step_*` directory. If your filesystem
  doesn't support symlinks (rare on HPC, sometimes seen on Lustre),
  the trainer falls back to scanning for the highest-numbered
  `step_*` directory -- but disable any preemption that interrupts a
  checkpoint write half-way.
* **Smoke-test mode (no audio data yet)**: omit `--audio-codes-dir`.
  The collator zero-fills the audio codebooks. The model trains
  end-to-end (text loss decreases), but it isn't learning anything
  meaningful about audio. Useful only for confirming the pipeline
  works on your cluster before investing in audio preprocessing.

## Audio preprocessing -- still a gap

The training data from `ssvd/rag_augment.py` is text-only. Before a
useful fine-tune you need to:

1. Synthesize TTS audio for every turn (Coqui XTTS, Bark, internal
   Kyutai TTS, etc. -- choice is yours).
2. Encode each turn's audio through Mimi (`moshi.models.loaders.get_mimi`
   then `mimi.encode(pcm)`).
3. Save the result as `.pt` files keyed by example index in
   `$AUDIO_CODES_DIR/{idx}.pt`. Shape: `[n_audio_codebooks, T]` long.

A preprocessing script for this isn't shipped because the TTS choice is
deployment-specific; the collator's audio-loading contract is small
enough to write your own in ~100 lines.
