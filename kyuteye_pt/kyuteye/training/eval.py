"""Evaluation harness for trained MoshiVis + RAG checkpoints.

Computes held-out perplexity over a JSONL eval split using the same
collator + loss as the trainer. Reuses :meth:`Trainer._build_conditioning`
so the eval path exercises the exact same conditioning logic
(including per-step streaming-sum if you trained with it) -- the only
difference is no backprop and a mean over the full eval set.

Designed to run as a small follow-on Slurm job after training. Single
GPU is fine -- the model is held in eval mode, no gradients tracked,
batch can be large.

Metrics reported:

* ``eval/loss``  mean CE on moshi-turn positions, token-weighted.
* ``eval/ppl``   exp(loss) -- standard speech-LM benchmark metric.
* ``eval/tokens``  total scored tokens (sanity check).

Caveat: this is a teacher-forced evaluation, not free generation.
Generation quality has to be measured separately (e.g. against a
spoken-QA benchmark) -- not shipped here because the metric choice is
deployment-specific.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader

from kyuteye.training.collator import RagDataCollator
from kyuteye.training.dataset import RagJsonlDataset
from kyuteye.training.distributed import DistributedContext, is_main_process

if TYPE_CHECKING:
    from kyuteye.models.image_projection import ImageProjection
    from kyuteye.models.moshivis import MoshiVis

    from kyuteye.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    tokens: int
    examples: int

    def as_dict(self) -> dict[str, float]:
        return {
            "eval/loss": self.loss,
            "eval/ppl": self.perplexity,
            "eval/tokens": float(self.tokens),
            "eval/examples": float(self.examples),
        }


@torch.no_grad()
def evaluate(
    trainer: "Trainer",
    dataset: RagJsonlDataset,
    *,
    batch_size: int = 4,
    max_batches: Optional[int] = None,
) -> EvalResult:
    """Run the trainer's forward pass over ``dataset`` and aggregate CE loss.

    Calls ``trainer.training_model`` directly under ``no_grad`` so the
    exact same conditioning + LM forward + masked loss runs in eval as
    in training -- including per-step streaming-sum if you trained with
    it. CFG dropout is forced off during eval. The trainer's
    optimizer/scheduler state is left untouched (this is purely
    read-only).

    :param max_batches: Cap on number of eval batches (useful for fast
        validation passes during long training runs). ``None`` runs
        the full dataset.
    """
    trainer.training_model.eval()
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=trainer.collator,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        total_loss_x_tokens = 0.0
        total_tokens = 0
        total_examples = 0

        device = trainer.ctx.device
        # Share the trainer's autocast dtype so eval and train measure
        # losses in the same precision (no accidental float32 advantage).
        dtype = trainer._dtype  # pylint: disable=protected-access

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=dtype != torch.float32,
            ):
                loss = trainer.training_model(
                    batch,
                    cfg_dropout=False,
                    per_step_streaming_sum=trainer.config.per_step_streaming_sum,
                )
            n_tokens = int(batch.loss_mask.sum().item())
            total_loss_x_tokens += loss.item() * n_tokens
            total_tokens += n_tokens
            total_examples += int(batch.loss_mask.shape[0])

        avg_loss = (
            total_loss_x_tokens / total_tokens if total_tokens > 0 else float("nan")
        )
        ppl = math.exp(avg_loss) if total_tokens > 0 else float("nan")
        return EvalResult(
            loss=avg_loss,
            perplexity=ppl,
            tokens=total_tokens,
            examples=total_examples,
        )
    finally:
        trainer.training_model.train()


def log_eval(
    trainer: "Trainer",
    result: EvalResult,
    *,
    label: str = "final",
) -> None:
    """Push eval metrics through the trainer's logger backends."""
    metrics = {
        f"eval/{label}/loss": result.loss,
        f"eval/{label}/ppl": result.perplexity,
        f"eval/{label}/tokens": float(result.tokens),
        f"eval/{label}/examples": float(result.examples),
    }
    if is_main_process(trainer.ctx):
        for lg in trainer._loggers:  # pylint: disable=protected-access
            lg.log_scalars(trainer.step, metrics)
        logger.info(
            "[eval/%s] loss=%.4f ppl=%.2f tokens=%d examples=%d",
            label,
            result.loss,
            result.perplexity,
            result.tokens,
            result.examples,
        )
