# Training: combined MoshiVis + MoshiRAG fine-tune

The architectural pieces a combined fine-tune needs are all in this
branch -- ARC encoder, conditioners, fuser, CFG plumbing, dropout
utilities, per-slot multi-batch. What's not here is the *training loop
driver*: dataset/loader wiring, optimizer, LR schedule, mixed precision,
checkpointing, distributed setup. This README explains the recommended
approach and the scaffolding pieces (`freeze.py`, `dataset.py`,
`loss.py`) we do provide.

Honest framing: neither MoshiVis nor MoshiRAG publish their training
code. Kyutai trained both internally. So a working trainer on this
branch is greenfield engineering on your part, not a port. The pieces
we provide minimize the surface you have to write, but a real fine-tune
still needs an optimizer setup, a data pipeline, and GPU time.

## Recommended phased fine-tune

### Phase 0 -- prepare data

Generate synthetic combined visual + RAG training examples with
`ssvd/rag_augment.py`. Three modes (`augment_visual`, `generate_text`,
`generate_tools`) all emit the JSONL schema that `RagJsonlDataset` reads:

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

# Visual+RAG: augment an existing SSVD dialogue dump.
uv run ssvd/rag_augment.py augment_visual \
    --input ssvd_dialogues.jsonl --output augmented.jsonl --num 5000

# Text-only RAG: generate from seed topics.
uv run ssvd/rag_augment.py generate_text \
    --topics topics.txt --output text_rag.jsonl --per_topic 20

# Tool calling: train the model to emit `[TOOL: name(args)]` and
# consume the result on the following turn. See ssvd/example_tools.json
# for the spec format and ssvd/example_tool_topics.txt for sample
# scenarios.
uv run ssvd/rag_augment.py generate_tools \
    --tools ssvd/example_tools.json \
    --topics ssvd/example_tool_topics.txt \
    --output tools_train.jsonl --per_topic 5
```

Mix them by concatenating the JSONL files in whatever ratio you want
(recommend roughly 50% visual+RAG, 30% text-only RAG, 20% tools as a
starting point -- adjust based on validation perplexity per modality
and your downstream use case). The collator handles all four roles
(`user`, `moshi`, `reference`, `tool`) uniformly.

### Phase 1 -- adapters only (recommended starting point)

Load published MoshiVis weights, attach the new RAG conditioners
(random init for ARC encoder + LUT + `learnt_padding`), freeze the
backbone, train only the new parameters.

```python
from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.loaders import get_moshi_vis
from kyuteye.training import apply_freeze_recipe, RagJsonlDataset, next_token_ce_loss

cfg = KyuteyeConfig.from_yml("configs/moshika-vis-rag.yaml")  # rag.enabled=true
moshi_vis, image_proj = get_moshi_vis(cfg, moshi_weight=..., device="cuda", dtype=torch.bfloat16)
moshi_vis.lm_model.train()
image_proj.train()

report = apply_freeze_recipe("adapters_only", moshi_vis.lm_model, image_proj, moshi_vis)
print(report.pretty())  # sanity-check what's trainable

dataset = RagJsonlDataset("augmented.jsonl")
# ... your DataLoader, AdamW, LR schedule, training loop ...
```

Suggested hyperparameters for the adapter phase (untested, derived from
common practice for similar adapter fine-tunes):

| | Value |
| --- | --- |
| LR | `1e-4` AdamW |
| Batch | 8-16 per GPU, gradient accum to effective 128 |
| Schedule | linear warmup 200 steps, cosine decay |
| Precision | bf16 (training_dtype = torch.bfloat16) |
| Steps | ~5-20k (adapter params are small, converges fast) |
| Dropout | conditioner dropout p=0.1-0.3 for CFG; use `dropout_all_conditions` from `kyuteye.conditioners` |
| Loss | `next_token_ce_loss` on moshi turns only; mask user/reference turns |

Compute estimate: ~200M trainable params on a 7B frozen base. Backward
through the frozen base is the dominant cost. Fits comfortably on
2-4x H100 80GB; runs in hours, not days.

### Phase 2 -- LM partial unfreeze (optional)

If adapter quality plateaus, unfreeze the LM backbone (keep PaliGemma
frozen). Pair with smaller LR (1e-5) and gradient clipping (max norm
1.0). This is more expensive but typically buys another 0.5-1 PPL on
factual benchmarks.

```python
report = apply_freeze_recipe("full_lm", moshi_vis.lm_model, image_proj, moshi_vis)
```

If you don't want to write away the full backbone, attach LoRA adapters
to it instead -- the freeze recipe stays at `adapters_only` and you add
LoRA modules as a separate trainable parameter group.

### Phase 3 -- LoRA on LM (alternative to full unfreeze)

Not implemented in this scaffold. LoRA libraries to consider: `peft`,
`loralib`. The integration point is adding LoRA modules to the
`MoshiVis.llm.transformer.layers[i].attention.*` linears and including
those parameters in the optimizer group.

## Scaffold pieces provided

* **`kyuteye/training/freeze.py`** -- `apply_freeze_recipe(name, ...)`
  with three named recipes:
  * `adapters_only` -- RAG conditioners + ARC encoder + bridge projector.
    Vision pathway frozen, Helium backbone frozen, depformer frozen.
  * `adapters_plus_xa` -- adds the shared cross-attention adapter on
    transformer layer 0 (which all other layers reference via the
    `SharedCrossAttention` metaclass).
  * `full_lm` -- everything trainable except PaliGemma2.
* **`kyuteye/training/dataset.py`** -- `RagJsonlDataset`,
  `RagExample`, `RagTurn`. Loads SSVD-style JSONL, exposes the structured
  records. Helpers like `dataset.rag_fraction()` and
  `dataset.visual_fraction()` for the trainer startup log.
* **`kyuteye/training/loss.py`** -- `next_token_ce_loss(text_logits,
  target_text_tokens, loss_mask=..., ignore_index=-100,
  label_smoothing=0.0)`. Plain masked CE on the model's text codebook.

## What you still have to write

* **Data pipeline** -- tokenize moshi turns into the text codebook,
  synthesize or load audio for each turn, encode it with Mimi, build
  per-example `ConditionAttributes` (with `first_speaker` and
  `reference_with_time` populated), apply random `dropout_all_conditions`
  for the CFG null branch.
* **Optimizer / scheduler** -- AdamW from `torch.optim` is fine.
* **Forward pass** -- assemble inputs and call `MoshiVis.forward_text`
  in training mode (not the streaming `MoshiVisGen.step` -- that's
  inference-only). Feed in the prepended condition via
  `sequence_emb`, the audio + text tokens via `input_ids`, and the
  image cross-attention KV via `cross_attention_src`.
* **Loss masking** -- build the `loss_mask` so only moshi turns
  contribute to the loss. User turns and reference turns should be
  ignored.
* **CFG dropout schedule** -- with probability `p_cfg`, replace the
  example's `ConditionAttributes` with `dropout_all_conditions(...)`.
  Common p_cfg=0.1-0.3.
* **Mixed precision** -- `torch.amp.autocast` wrap around the forward.
* **Gradient accumulation + clipping**.
* **Checkpointing** -- save `MoshiVis.state_dict()` periodically; the
  loaders in this branch handle `strict=False` so partial checkpoints
  are fine.
* **Distributed training** -- DDP if multi-GPU. None of our code
  assumes single-GPU; it just doesn't ship distributed wiring.
* **Evaluation harness** -- pick something off-the-shelf or write a
  small one against a held-out split of the SSVD-augmented JSONL.

## The shipped trainer driver (`scripts/train.py`)

A working trainer is now wired up end-to-end:

* **Entry point**: `kyuteye_pt/scripts/train.py`. Run under torchrun for
  DDP. Driven by `fire`-style CLI flags.
* **Collator**: `kyuteye.training.RagDataCollator` turns a batch of
  `RagExample` into model input tensors. Loads pre-encoded audio codes
  (`.pt` per example) and pre-computed image cross-attention KV (`.pt`
  per image) from disk to keep per-step cost down.
* **Trainer**: `kyuteye.training.Trainer` -- AdamW + linear warmup +
  cosine decay, bf16 autocast, gradient clipping, periodic
  checkpointing with a `latest` symlink, resume from latest, DDP wrap
  with `find_unused_parameters=True` for the partial-freeze case.
* **Slurm batch templates**: see `slurm/train_adapter.sbatch` and
  `slurm/README.md` for single-node and multi-node patterns.

Quick start on a single-node HPC allocation (4 GPUs):

```bash
cd $REPO_ROOT
sbatch slurm/train_adapter.sbatch
```

Or directly via torchrun (skipping the Slurm wrapper):

```bash
cd $REPO_ROOT/kyuteye_pt
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --kyuteye-config configs/moshika-vis.yaml \
    --data ../data/augmented.jsonl \
    --audio-codes-dir ../data/audio_codes \
    --precomputed-image-kv-dir ../data/image_kv \
    --save-dir checkpoints/adapter_v1 \
    --freeze-recipe adapters_only \
    --num-steps 10000 \
    --batch-size 8 \
    --learning-rate 1e-4
```

Honest caveat: I wrote the trainer structurally per best-practice
patterns (DDP via torchrun env vars, bf16 autocast around the forward,
gradient clipping, decoupled weight decay, checkpoint/resume) but it
hasn't been exercised on real GPU + weights from this environment.
First run on your cluster will surface integration bugs. Plan for one
shakeout day before scheduling the long-running fine-tune.

## Audio preprocessing

`ssvd/rag_augment.py` emits text-only JSONL; the trainer's collator
expects per-example Mimi-encoded audio codes in `--audio-codes-dir`.
The audio preprocessing pipeline lives in
`kyuteye.training.audio_preprocess`:

* `BaseTTS` -- abstract TTS interface. One method:
  `synthesize(text, speaker) -> Tensor`.
* `SilenceTTS` -- placeholder that returns zero PCM. Use for end-to-end
  pipeline validation before committing to a TTS dependency.
* `CoquiXTTS` -- driver for Coqui's `TTS` package (XTTS v2). Lazy
  import; install with `pip install TTS` and supply reference audio
  for each speaker voice.
* `AudioPreprocessor.process_example(example, idx)` -- synthesizes the
  user + moshi tracks for one dialogue, encodes through Mimi, saves
  `{idx}.pt` with shape `[n_audio_codebooks, T]` matching the trainer
  collator's expected layout.

CLI entry point:

```bash
python -m kyuteye.training.audio_preprocess \
    --input data/augmented.jsonl \
    --output-dir data/audio_codes \
    --mimi-weight $MIMI_WEIGHT \
    --tts coqui_xtts \
    --tts-user-ref refs/user.wav \
    --tts-moshi-ref refs/moshi.wav
```

Slurm template: `slurm/preprocess_audio.sbatch`. Supports array-job
sharding for parallelizing across large datasets.

For other TTS engines (Bark, StyleTTS2, your in-house model): subclass
`BaseTTS` in your own module, instantiate `AudioPreprocessor` directly.
The interface is one method; pluggability was the explicit goal.

## CFG dropout during training

The trainer's `_forward_one_microbatch` now applies CFG conditioner
dropout when `--cfg-dropout-p > 0`: with that probability the entire
batch's ConditionAttributes are replaced with their null version via
`dropout_all_conditions`, and the resulting condition tensors flow
through the provider/fuser into `sum_condition` for the forward pass.
The model learns both `p(x | condition)` and `p(x | null)`
distributions; inference-time `cfg_coef > 1.0` then interpolates
between them.

Per-batch (not per-example) dropout is a deliberate simplification --
mixing pos+null in a single batch would require splitting the forward
which loses parallelism. Per-example mixing is achieved naturally by
the random dropout decision varying across training steps.

### Streaming-sum vs sum-condition trade-off at training

The conditioner fuser routes `reference_with_time` to `streaming_sum`
in MoshiRAG's stock config. At inference, the streaming-sum queue is
consumed one row per step (`MoshiVisGen.apply_pending_streaming_sum_condition`).
At training the LM's forward runs over the full sequence, where the
`streaming_sum_condition` parameter asserts `seq_len == 1`. The
trainer's `_build_conditioning` resolves this by mean-pooling the
streaming-sum tensor over its sequence dimension into a single
broadcastable conditioning vector that's added to `sum_condition`.
This is a *training-time simplification*: the ARC encoder still gets
gradient signal, but the per-step temporal alignment that
streaming-sum gives at inference is lost during training. For a
faithful per-step streaming-sum training path, override
`_build_conditioning` to align the reference embeddings with the LM's
sequence length on a per-position basis -- not shipped because it
needs design choices that depend on your data layout (how do you
align a variable-length reference with the dialogue timeline?).

## Pluggable logging (WandB / TensorBoard)

`kyuteye.training.logging_hooks` provides a `Logger` abstract base
with three shipped backends:

* `PythonLogger` -- default, prints to stdout.
* `TensorBoardLogger` -- writes scalars + hparams via `SummaryWriter`.
  `pip install tensorboard`.
* `WandbLogger` -- mirrors to a Weights & Biases run. `pip install wandb`.
  Use `mode="offline"` in `init_kwargs` for HPC nodes without internet
  and `wandb sync` from a login node afterwards.

Enable via `--log-backends python,tensorboard,wandb` (comma-separated;
defaults to `python`). The trainer logs train loss, learning rate, and
steps/sec every `--log-every` steps. Eval metrics from the held-out
job land in the same logger backends.

## Per-step streaming-sum training (faithful path)

The `MoshiVis.forward_text` assertion was lifted to accept
`streaming_sum_condition` of either `[B, 1, dim]` (inference per-step)
or `[B, T_lm, dim]` (training per-position). With
`--per-step-streaming-sum`, `Trainer._build_streaming_sum_per_step`
finds the first `<ret>` token position per example and places the ARC
encoder's reference embedding rows at `[ret_pos + 1, ret_pos + 1 + T_ref)`.
The model now learns position-accurate streaming-sum consumption
during training, matching MoshiRAG's inference semantics exactly.

Requires the model to have `rag_token_id` set (in the YAML's `rag:` block).
Without it, the trainer falls back to the mean-pool simplification
documented above.

## Evaluation

`kyuteye.training.eval.evaluate(trainer, dataset)` runs the trainer's
forward over a held-out JSONL split and aggregates token-weighted CE
into a perplexity. Entry point: `scripts/eval.py`; Slurm template:
`slurm/eval.sbatch`. Single-GPU is sufficient.

```bash
python scripts/eval.py \
    --kyuteye-config configs/moshika-vis.yaml \
    --checkpoint checkpoints/run_v1/latest/ckpt.pt \
    --data data/eval.jsonl \
    --audio-codes-dir data/eval_audio_codes/
```

Eval is teacher-forced (not free generation). For real generation
quality you need a separate benchmark; this gives you a fast
training-signal proxy that catches regressions.

## One-command pipeline

`scripts/run_pipeline.sh` submits preprocessing → training → eval as
Slurm jobs with `afterok` dependencies:

```bash
MIMI_WEIGHT=/path/to/mimi.safetensors \
DATA_JSONL=$PWD/data/augmented.jsonl \
EVAL_DATA=$PWD/data/eval.jsonl \
./scripts/run_pipeline.sh
```

Submits 3-4 jobs (training preprocess + eval preprocess in parallel,
then train, then eval). Captures all job IDs to
`$REPO_ROOT/.pipeline_jobs` for atomic cancellation
(`scancel $(cat .pipeline_jobs)`). Knobs: `NUM_STEPS`, `BATCH_SIZE`,
`LEARNING_RATE`, `FREEZE_RECIPE`, `TTS_BACKEND`, `SKIP_PREPROCESS=1`
(reuse existing audio codes), `SKIP_EVAL=1` (train only). Full docs in
the script's header comment.
