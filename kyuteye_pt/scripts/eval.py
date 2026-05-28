#!/usr/bin/env python
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate a trained MoshiVis + RAG checkpoint on a held-out JSONL split.

Loads the model + checkpoint, builds the same collator the trainer
uses, and runs :func:`kyuteye.training.eval.evaluate` to compute
held-out perplexity.

Designed to be the third Slurm job in the pipeline (after
preprocessing and training). Single GPU is fine; the model is in eval
mode and gradients aren't tracked.

Usage:

.. code-block:: bash

    python scripts/eval.py \\
        --kyuteye-config configs/moshika-vis.yaml \\
        --checkpoint checkpoints/adapter_v1/latest/ckpt.pt \\
        --data data/eval.jsonl \\
        --audio-codes-dir data/eval_audio_codes/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fire
import sentencepiece
import torch
from huggingface_hub import hf_hub_download

from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.loaders import get_moshi_vis
from kyuteye.training import (
    RagDataCollator,
    RagJsonlDataset,
    Trainer,
    TrainerConfig,
    cleanup_distributed,
    init_distributed,
    is_main_process,
)
from kyuteye.training.eval import evaluate, log_eval


def main(
    kyuteye_config: str,
    checkpoint: str,
    data: str,
    audio_codes_dir: Optional[str] = None,
    precomputed_image_kv_dir: Optional[str] = None,
    batch_size: int = 4,
    max_batches: Optional[int] = None,
    max_seq_len: Optional[int] = 1024,
    dtype: str = "bfloat16",
    moshi_weight: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    """Evaluate ``checkpoint`` on ``data`` and print held-out perplexity.

    :param checkpoint: Path to a ``ckpt.pt`` saved by the trainer.
        Reads ``moshi_vis`` and ``image_proj`` state dicts from it
        (with ``strict=False`` so partial checkpoints are fine).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ctx = init_distributed()
    cfg = KyuteyeConfig.from_yml(kyuteye_config)

    if moshi_weight is None:
        if cfg.hf_repo is None:
            raise ValueError(
                "moshi_weight not given and YAML has no hf_repo"
            )
        moshi_weight = hf_hub_download(cfg.hf_repo, cfg.model)
        pt_variant = moshi_weight.replace(".safetensors", "_pt.safetensors")
        if Path(pt_variant).exists():
            moshi_weight = pt_variant

    tokenizer_path = (
        hf_hub_download(cfg.hf_repo, cfg.text_tokenizer)
        if cfg.hf_repo is not None
        else cfg.text_tokenizer
    )
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore[arg-type]

    torch_dtype = getattr(torch, dtype)
    moshi_vis_gen, image_proj = get_moshi_vis(
        cfg,
        moshi_weight=moshi_weight,
        device=str(ctx.device),
        dtype=torch_dtype,
    )

    # Load the fine-tuned checkpoint on top of the base weights. The
    # state-dict load is ``strict=False`` because the trainer saves only
    # the trainable subset under a partial-freeze recipe; the
    # already-loaded base weights cover the frozen portion. Assert the
    # top-level keys are present so a corrupted / wrong-format
    # checkpoint fails loud instead of silently leaving the base weights
    # unchanged (which produced a misleading "trained but no improvement"
    # signal in eval).
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "moshi_vis" not in state or "image_proj" not in state:
        raise RuntimeError(
            f"checkpoint {checkpoint!r} is missing required top-level keys; "
            f"got {sorted(state.keys())}. Expected at least "
            f"'moshi_vis' and 'image_proj'."
        )
    moshi_vis_gen.lm_model.load_state_dict(state["moshi_vis"], strict=False)
    image_proj.load_state_dict(state["image_proj"], strict=False)
    if is_main_process(ctx):
        logging.info("[eval] loaded checkpoint at step %d", state.get("step", -1))

    # Eval doesn't need a freeze recipe -- there's no backward pass and
    # ``requires_grad`` flags don't affect the forward. We do build a
    # degenerate Trainer instance below purely to reuse its
    # ``training_model`` wrapper (which carries the DDP-correct
    # conditioning + LM forward). The dataset passed in is the eval
    # split; the Trainer's DataLoader isn't iterated here, ``evaluate``
    # builds its own.
    dataset = RagJsonlDataset(data)
    if is_main_process(ctx):
        logging.info(
            "[eval] dataset size=%d  rag_fraction=%.2f  visual_fraction=%.2f",
            len(dataset),
            dataset.rag_fraction(),
            dataset.visual_fraction(),
        )

    collator = RagDataCollator(
        tokenizer,
        num_codebooks=moshi_vis_gen.lm_model.num_codebooks,
        audio_offset=moshi_vis_gen.lm_model.audio_offset,
        max_seq_len=max_seq_len,
        audio_codes_dir=audio_codes_dir,
        precomputed_image_kv_dir=precomputed_image_kv_dir,
    )

    trainer = Trainer(
        moshi_vis=moshi_vis_gen.lm_model,
        image_proj=image_proj,
        dataset=dataset,  # not actually iterated -- just for the trainer init
        collator=collator,
        ctx=ctx,
        config=TrainerConfig(
            batch_size=batch_size,
            num_steps=1,  # unused -- we don't call train()
            log_backends="python",
            dtype=dtype,
        ),
    )

    try:
        result = evaluate(
            trainer, dataset, batch_size=batch_size, max_batches=max_batches
        )
        log_eval(trainer, result, label="held_out")
        if is_main_process(ctx):
            print(
                f"\nEval result: loss={result.loss:.4f} "
                f"ppl={result.perplexity:.2f} tokens={result.tokens}"
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(main)
