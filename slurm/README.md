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

## `preprocess_audio.sbatch`

One-shot audio preprocessing job. Synthesizes TTS audio for every turn
in the training JSONL, encodes through Mimi, writes per-example
`.pt` files in `$AUDIO_CODES_DIR`. Single-GPU job (the Mimi codec is
small and the TTS is the bottleneck).

Submit:

```bash
MIMI_WEIGHT=/path/to/tokenizer-...safetensors \
TTS_BACKEND=coqui_xtts \
TTS_USER_REF=/path/to/user_voice.wav \
TTS_MOSHI_REF=/path/to/moshi_voice.wav \
sbatch slurm/preprocess_audio.sbatch
```

For initial pipeline validation without a real TTS dep:

```bash
MIMI_WEIGHT=/path/to/tokenizer.safetensors \
TTS_BACKEND=silence \
sbatch slurm/preprocess_audio.sbatch
```

The `silence` backend writes zero-PCM audio (Mimi-encodes to the
"silence" code at every timestep). The training pipeline runs
end-to-end but the model isn't learning audio behavior -- useful only
for confirming the trainer runs on your cluster before investing in
TTS setup.

### Parallelizing across the dataset (array jobs)

For large datasets, run preprocessing as a Slurm array job split into
shards:

```bash
SHARDS=8 TOTAL_EXAMPLES=50000 \
MIMI_WEIGHT=/path/to/mimi.safetensors \
TTS_BACKEND=silence \
sbatch --array=0-7 slurm/preprocess_audio.sbatch
```

Each task processes `TOTAL_EXAMPLES / SHARDS` consecutive examples.
`--skip-existing` (set in the script by default) means re-running an
array re-processes only the missing shards.

### Pluggable TTS

Two TTS backends ship in `kyuteye.training.audio_preprocess`:

* `SilenceTTS` -- zero PCM proportional to text length. Validation only.
* `CoquiXTTS` -- driver for the `TTS` PyPI package (Coqui XTTS v2). Needs `pip install TTS` plus reference audio clips for each speaker.

For other TTS engines (Bark, StyleTTS2, internal models): subclass
`BaseTTS` in your own module and call `AudioPreprocessor.process_example`
directly. The interface is one method (`synthesize(text, speaker) -> Tensor`)
and the docs in `audio_preprocess.py` document the contract.
