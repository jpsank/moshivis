#!/usr/bin/env python
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for combined MoshiVis + MoshiRAG fine-tune.

Designed to be launched under torchrun (single- or multi-node) from a
Slurm batch script. See ``slurm/train_adapter.sbatch`` for an example.

Usage (single-node, 4 GPUs):

.. code-block:: bash

    torchrun --nproc_per_node=4 scripts/train.py \\
        --kyuteye-config configs/moshika-vis.yaml \\
        --data data/augmented.jsonl \\
        --audio-codes-dir data/audio/ \\
        --save-dir checkpoints/adapter_run_01 \\
        --num-steps 10000 \\
        --batch-size 8 \\
        --learning-rate 1e-4 \\
        --freeze-recipe adapters_only

Multi-node (e.g. 2 nodes × 8 GPUs): ``torchrun`` with ``--nnodes 2
--node_rank $SLURM_NODEID --master_addr $MASTER_ADDR
--master_port 29500``. See the Slurm template.

This is a scaffold trainer -- structurally correct per best-practice
patterns but unrun in this environment. First training run on a real
cluster will surface integration bugs; iterate from there. See
``kyuteye/training/README.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

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
    apply_freeze_recipe,
    cleanup_distributed,
    init_distributed,
    is_main_process,
)


def main(
    kyuteye_config: str,
    data: str,
    save_dir: str,
    audio_codes_dir: str | None = None,
    image_dir: str | None = None,
    precomputed_image_kv_dir: str | None = None,
    freeze_recipe: str = "adapters_only",
    num_steps: int = 10_000,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    save_every: int = 500,
    log_every: int = 10,
    cfg_dropout_p: float = 0.0,
    max_seq_len: int | None = 1024,
    dtype: str = "bfloat16",
    seed: int = 42,
    resume_dir: str | None = None,
    moshi_weight: str | None = None,
    log_level: str = "INFO",
) -> None:
    """Train MoshiVis + RAG with the configured freeze recipe.

    :param kyuteye_config: Path to the YAML model config (e.g.
        ``configs/moshika-vis-rag.yaml`` with ``rag.enabled: true``).
    :param data: JSONL file produced by ``ssvd/rag_augment.py``.
    :param save_dir: Where to write checkpoints + the ``latest`` symlink.
    :param audio_codes_dir: Per-example pre-encoded audio codes (.pt
        files indexed by example index). Omit for smoke-test mode
        (zero-filled audio; gradient flows but model learns nothing
        meaningful from audio).
    :param image_dir / precomputed_image_kv_dir: see collator docstring.
    :param freeze_recipe: ``adapters_only`` | ``adapters_plus_xa`` | ``full_lm``.
    :param resume_dir: If set, load the latest checkpoint in this dir
        and resume from that step.
    :param moshi_weight: Local override for the model weights; defaults
        to HF download driven by the YAML ``hf_repo`` field.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ctx = init_distributed()
    if is_main_process(ctx):
        logging.info("[train] starting on world_size=%d", ctx.world_size)

    cfg = KyuteyeConfig.from_yml(kyuteye_config)

    # Resolve moshi weights from HF or local path.
    if moshi_weight is None:
        if cfg.hf_repo is None:
            raise ValueError(
                "moshi_weight not given and YAML has no hf_repo to download from"
            )
        moshi_weight = hf_hub_download(cfg.hf_repo, cfg.model)
        if not moshi_weight.endswith("_pt.safetensors"):
            # Match the convention in server.py.
            pt_variant = moshi_weight.replace(".safetensors", "_pt.safetensors")
            if Path(pt_variant).exists():
                moshi_weight = pt_variant

    # Tokenizer (used by the collator).
    if cfg.hf_repo is None:
        tokenizer_path = cfg.text_tokenizer
    else:
        tokenizer_path = hf_hub_download(cfg.hf_repo, cfg.text_tokenizer)
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore[arg-type]

    torch_dtype = getattr(torch, dtype)
    moshi_vis_gen, image_proj = get_moshi_vis(
        cfg,
        moshi_weight=moshi_weight,
        device=str(ctx.device),
        dtype=torch_dtype,
    )

    # Apply freeze recipe. Note ``moshi_vis_gen.lm_model`` is the
    # MoshiVis backbone -- recipes operate on that.
    moshi_vis = moshi_vis_gen.lm_model
    moshi_vis.train()
    image_proj.train()
    report = apply_freeze_recipe(freeze_recipe, moshi_vis, image_proj, moshi_vis_gen)
    if is_main_process(ctx):
        logging.info("\n%s", report.pretty())

    # Dataset + collator.
    dataset = RagJsonlDataset(data)
    if is_main_process(ctx):
        logging.info(
            "[train] dataset size=%d  rag_fraction=%.2f  visual_fraction=%.2f",
            len(dataset),
            dataset.rag_fraction(),
            dataset.visual_fraction(),
        )

    collator = RagDataCollator(
        tokenizer,
        num_codebooks=moshi_vis.num_codebooks,
        audio_offset=moshi_vis.audio_offset,
        max_seq_len=max_seq_len,
        audio_codes_dir=audio_codes_dir,
        image_dir=image_dir,
        precomputed_image_kv_dir=precomputed_image_kv_dir,
    )

    trainer_cfg = TrainerConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        save_every=save_every,
        log_every=log_every,
        cfg_dropout_p=cfg_dropout_p,
        save_dir=save_dir,
        seed=seed,
        dtype=dtype,
        resume_dir=resume_dir,
    )
    trainer = Trainer(
        moshi_vis=moshi_vis,
        image_proj=image_proj,
        dataset=dataset,
        collator=collator,
        ctx=ctx,
        config=trainer_cfg,
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(main)
