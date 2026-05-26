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

## Why we don't ship a `train.py`

Two reasons:

1. **Truthfulness**. We haven't run a training step in this environment
   (no GPU here). Shipping a `train.py` we haven't actually exercised
   on real weights is the worst kind of code -- it looks done but
   needs work the user can't predict. The scaffolding above is the
   honest line between "we built this" and "you build this".
2. **Hyperparameter-specificity**. The right LR, batch size, dropout
   rate, schedule, etc. depend on your hardware, data size, and goal.
   Codifying a single setup would be misleading. Use the suggested
   values above as a starting point and tune from there.
