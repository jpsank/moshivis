"""Training scaffold for a combined MoshiVis + MoshiRAG fine-tune.

This package provides the pieces a training loop needs that the inference
codebase doesn't have:

* :mod:`freeze` -- parameter-group selection (freeze the LM backbone, train
  the new RAG conditioners + ARC encoder, etc.). The "partial freeze"
  pattern recommended in ``training/README.md``.
* :mod:`dataset` -- ``RagJsonlDataset`` that consumes the JSONL produced by
  ``ssvd/rag_augment.py``.
* :mod:`loss` -- ``next_token_ce_loss`` over the model's predicted text
  codebook, masked to the moshi turns.

What is intentionally NOT in this package: the trainer driver (optimizer,
LR schedule, mixed precision, distributed setup, checkpointing). Those
are deployment-specific and need a real training run + budget to
calibrate; we don't ship code we haven't run. See
``training/README.md`` for what to wire on top of these primitives.

Importing this package does NOT require a GPU. The trainer itself, once
written, will.
"""

from kyuteye.training.collator import CollatedBatch, RagDataCollator
from kyuteye.training.dataset import RagExample, RagJsonlDataset, RagTurn
from kyuteye.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
)
from kyuteye.training.freeze import (
    FreezeReport,
    apply_freeze_recipe,
    freeze_recipes,
    summarize_freeze,
)
from kyuteye.training.loss import next_token_ce_loss
from kyuteye.training.trainer import Trainer, TrainerConfig

__all__ = [
    "CollatedBatch",
    "DistributedContext",
    "FreezeReport",
    "RagDataCollator",
    "RagExample",
    "RagJsonlDataset",
    "RagTurn",
    "Trainer",
    "TrainerConfig",
    "apply_freeze_recipe",
    "barrier",
    "cleanup_distributed",
    "freeze_recipes",
    "init_distributed",
    "is_main_process",
    "next_token_ce_loss",
    "summarize_freeze",
]
