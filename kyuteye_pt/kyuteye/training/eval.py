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
from kyuteye.training.loss import next_token_ce_loss

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

    Operates through the trainer instance so all of its plumbing
    (conditioning builder, mixed precision dtype, DDP module unwrap,
    per-step streaming-sum) is reused. The trainer is left at the same
    step + optimizer state on return -- this is purely read-only.

    :param max_batches: Cap on number of eval batches (useful for fast
        validation passes during long training runs). ``None`` runs
        the full dataset.
    """
    # Switch to eval mode and back so dropout / batchnorm semantics are
    # correct. ``DistributedDataParallel`` proxies ``.eval()`` /
    # ``.train()`` to the wrapped module.
    trainer.moshi_vis.eval()
    trainer.image_proj.eval()
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

        inner = (
            trainer.moshi_vis.module
            if hasattr(trainer.moshi_vis, "module")
            else trainer.moshi_vis
        )
        device = trainer.ctx.device
        # Trainer's protected attribute -- accessed because we deliberately
        # share its mixed-precision config here.
        dtype = trainer._dtype  # pylint: disable=protected-access

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch.input_ids.to(device)
            target_text = batch.target_text.to(device)
            loss_mask = batch.loss_mask.to(device)
            cross_attention_src = (
                batch.cross_attention_src.to(device)
                if batch.cross_attention_src is not None
                else None
            )
            sum_condition, fuser_cross, streaming_sum_per_step = (
                trainer._build_conditioning(  # pylint: disable=protected-access
                    batch.condition_attributes,
                    device,
                    inner,
                    text_input_ids=input_ids[:, 0],
                )
            )
            ca_src = cross_attention_src
            if ca_src is None and fuser_cross is not None:
                ca_src = fuser_cross.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=dtype != torch.float32,
            ):
                _, text_logits, _ = inner.forward_text(
                    input_ids=input_ids,
                    cross_attention_src=ca_src,
                    sum_condition=sum_condition.to(device)
                    if sum_condition is not None
                    else None,
                    streaming_sum_condition=streaming_sum_per_step.to(device)
                    if streaming_sum_per_step is not None
                    else None,
                )
                text_logits = text_logits.squeeze(1)
                loss = next_token_ce_loss(
                    text_logits=text_logits,
                    target_text_tokens=target_text,
                    loss_mask=loss_mask,
                )

            # Token-weight the mean so the final figure is comparable
            # across batches of unequal moshi-turn length.
            n_tokens = int(loss_mask.sum().item())
            total_loss_x_tokens += loss.item() * n_tokens
            total_tokens += n_tokens
            total_examples += int(loss_mask.shape[0])

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
        trainer.moshi_vis.train()
        trainer.image_proj.train()


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
