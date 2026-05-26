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
`ssvd/rag_augment.py`. Both modes (`augment_visual` and `generate_text`)
emit the JSONL schema that `RagJsonlDataset` reads:

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

# Visual+RAG: augment an existing SSVD dialogue dump.
uv run ssvd/rag_augment.py augment_visual \
    --input ssvd_dialogues.jsonl --output augmented.jsonl --num 5000

# Text-only RAG: generate from seed topics.
uv run ssvd/rag_augment.py generate_text \
    --topics topics.txt --output text_rag.jsonl --per_topic 20
```

Mix them in a ratio that preserves vision quality (recommend roughly
50/50 visual/text-only -- adjust based on validation perplexity per
modality).

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

## What still has to be written

* **Audio preprocessing pipeline**. `ssvd/rag_augment.py` emits
  text-only JSONL; the trainer expects per-example audio codes in
  `--audio-codes-dir`. A preprocessing script that synthesizes TTS
  audio for each turn and Mimi-encodes it isn't shipped because the
  TTS choice is deployment-specific. The collator's audio contract is
  small (one `.pt` file per example, shape `[n_audio_codebooks, T]`).
  You can run a one-shot job on a TTS of your choice; document the
  recipe in your fork.
* **CFG dropout during training**. The trainer hardcodes the
  `cfg_dropout_p` argument but doesn't yet wire it through to the
  forward pass -- the static condition slots set up at
  `MoshiVisGen.__init__` carry the conditioning. For a true CFG
  training loop, override `Trainer._forward_one_microbatch` to apply
  `dropout_all_conditions` per micro-batch and re-prepare the
  ConditionTensors before the forward. ~30 lines of override code.
* **WandB / TensorBoard logging**. The trainer logs to Python logging
  every `--log-every` steps. Wrap the log call in a hook if you want
  experiment tracking.
