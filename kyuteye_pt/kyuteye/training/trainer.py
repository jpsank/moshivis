"""Minimal training loop driver for combined MoshiVis + MoshiRAG fine-tune.

The :class:`Trainer` wires up: DDP, bf16 mixed precision, gradient
accumulation, optimizer + cosine schedule, periodic checkpointing,
optional CFG conditioner dropout, the masked next-token CE loss. It is
intentionally minimal -- no WandB integration, no fancy curriculum, no
multi-objective loss balancing. Add those on top if needed; the surface
here is deliberately small so a Slurm job can run unattended for a day
and produce a checkpoint.

Honest framing: this code is structurally correct per best-practice
patterns (DDP init via torchrun env vars, bf16 autocast around the
forward, gradient clip, parameter-group LR, decoupled weight decay).
But it has NOT been run on a real GPU + real weights from this
environment. The first run on your cluster will surface integration
bugs no amount of static review catches -- expect to iterate.

Typical usage:

.. code-block:: python

    from kyuteye.training import Trainer, RagJsonlDataset, RagDataCollator, apply_freeze_recipe
    from kyuteye.training.distributed import init_distributed

    ctx = init_distributed()
    moshi_vis_gen, image_proj = ...  # via kyuteye.models.loaders.get_moshi_vis
    apply_freeze_recipe("adapters_only", moshi_vis_gen.lm_model, image_proj, moshi_vis_gen)

    dataset = RagJsonlDataset("data/augmented.jsonl")
    collator = RagDataCollator(tokenizer, num_codebooks=17, audio_codes_dir="data/audio/")
    trainer = Trainer(
        moshi_vis=moshi_vis_gen.lm_model,
        image_proj=image_proj,
        dataset=dataset,
        collator=collator,
        ctx=ctx,
        learning_rate=1e-4,
        batch_size=8,
        grad_accum_steps=4,
        num_steps=10000,
        save_every=500,
        save_dir="checkpoints/",
        cfg_dropout_p=0.1,
    )
    trainer.train()
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler  # noqa: F401  # retained for the float16 fallback
from torch.utils.data import DataLoader

from kyuteye.conditioners import dropout_all_conditions
from kyuteye.training.collator import CollatedBatch, RagDataCollator
from kyuteye.training.dataset import RagJsonlDataset
from kyuteye.training.distributed import (
    DistributedContext,
    barrier,
    is_main_process,
)
from kyuteye.training.loss import next_token_ce_loss

if TYPE_CHECKING:
    from kyuteye.models.image_projection import ImageProjection
    from kyuteye.models.moshivis import MoshiVis

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """All hyperparameters in one struct so the entry point can argparse them cleanly."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    batch_size: int = 8
    grad_accum_steps: int = 1
    num_steps: int = 10_000
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1  # cosine decays to this fraction of peak LR
    save_every: int = 500
    log_every: int = 10
    cfg_dropout_p: float = 0.0  # 0 disables CFG dropout
    save_dir: str = "checkpoints"
    seed: int = 42
    dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"
    # Resume from this checkpoint dir if non-empty (loads latest step inside).
    resume_dir: Optional[str] = None


class Trainer:
    """Single-objective LM trainer for MoshiVis + RAG.

    Loss: masked next-token CE on the text codebook (only positions in
    moshi turns contribute). Optimizer: AdamW with parameter groups
    matching the freeze recipe -- only ``requires_grad=True`` params
    are passed to the optimizer.

    Mixed precision: bf16 autocast around the forward pass; no
    GradScaler since bf16 doesn't need one. The fp16 path is left as a
    TODO (GradScaler integration) -- defaulting to bf16 keeps things
    simple and matches what modern A100 / H100 deployments use.
    """

    def __init__(
        self,
        *,
        moshi_vis: "MoshiVis",
        image_proj: "ImageProjection",
        dataset: RagJsonlDataset,
        collator: RagDataCollator,
        ctx: DistributedContext,
        config: Optional[TrainerConfig] = None,
        **kwargs,
    ) -> None:
        self.config = config or TrainerConfig(**kwargs)
        self.ctx = ctx
        self.moshi_vis = moshi_vis
        self.image_proj = image_proj
        self.dataset = dataset
        self.collator = collator

        torch.manual_seed(self.config.seed + ctx.rank)

        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.dtype]

        # DDP wrapping. We wrap the MoshiVis backbone; the image encoder
        # is not in the gradient graph for the adapter recipe (frozen)
        # but is wrapped too if any of its params are trainable.
        self.moshi_vis = self._maybe_wrap_ddp(self.moshi_vis)
        self.image_proj = self._maybe_wrap_ddp(self.image_proj)

        # Optimizer + scheduler over the trainable params.
        trainable_params = [
            p
            for p in list(self.moshi_vis.parameters())
            + list(self.image_proj.parameters())
            if p.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found. Did you call apply_freeze_recipe()?"
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=self.config.adam_betas,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        self.step = 0
        self.save_dir = Path(self.config.save_dir)
        if is_main_process(ctx):
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader. Distributed sampler when running under DDP so each
        # rank sees a disjoint partition.
        if ctx.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            self.sampler = DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                seed=self.config.seed,
            )
            shuffle = False
        else:
            self.sampler = None
            shuffle = True
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        if self.config.resume_dir:
            self._resume(self.config.resume_dir)

    # ------------------------------------------------------------------ wrap
    def _maybe_wrap_ddp(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.ctx.is_distributed:
            return module.to(self.ctx.device)
        from torch.nn.parallel import DistributedDataParallel as DDP

        module = module.to(self.ctx.device)
        # ``find_unused_parameters=True`` is safer for partial-freeze
        # setups where some params don't receive grads (PaliGemma frozen,
        # depformer frozen, etc.). It costs a bit of perf; flip to False
        # once the freeze recipe is settled.
        return DDP(
            module,
            device_ids=[self.ctx.local_rank],
            output_device=self.ctx.local_rank,
            find_unused_parameters=True,
        )

    # ------------------------------------------------------------------ schedule
    def _lr_lambda(self, step: int) -> float:
        """Linear warmup then cosine decay to ``min_lr_ratio`` * peak."""
        if step < self.config.warmup_steps:
            return float(step) / float(max(1, self.config.warmup_steps))
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.num_steps - self.config.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.config.min_lr_ratio
            + (1.0 - self.config.min_lr_ratio) * cosine
        )

    # ------------------------------------------------------------------ loop
    def train(self) -> None:
        """Run the configured number of optimizer steps."""
        if is_main_process(self.ctx):
            logger.info(
                "[trainer] starting: num_steps=%d batch_size=%d grad_accum=%d "
                "effective_batch=%d world_size=%d",
                self.config.num_steps,
                self.config.batch_size,
                self.config.grad_accum_steps,
                self.config.batch_size
                * self.config.grad_accum_steps
                * self.ctx.world_size,
                self.ctx.world_size,
            )

        data_iter = iter(self._infinite_dataloader())
        last_log_time = time.time()
        last_log_step = self.step

        while self.step < self.config.num_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                batch = next(data_iter)
                loss = self._forward_one_microbatch(batch)
                loss = loss / self.config.grad_accum_steps
                loss.backward()
                accumulated_loss += loss.item()

            # Gradient clip + step.
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p in list(self.moshi_vis.parameters())
                        + list(self.image_proj.parameters())
                        if p.requires_grad
                    ],
                    self.config.grad_clip_norm,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            # Logging.
            if (
                is_main_process(self.ctx)
                and self.step % self.config.log_every == 0
            ):
                now = time.time()
                dt = now - last_log_time
                steps = self.step - last_log_step
                steps_per_sec = steps / dt if dt > 0 else 0.0
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "[trainer] step %d/%d  loss=%.4f  lr=%.2e  %.2f steps/s",
                    self.step,
                    self.config.num_steps,
                    accumulated_loss * self.config.grad_accum_steps,
                    lr,
                    steps_per_sec,
                )
                last_log_time = now
                last_log_step = self.step

            # Checkpoint.
            if self.step % self.config.save_every == 0:
                self._save()
                barrier(self.ctx)

        # Final checkpoint.
        self._save()
        barrier(self.ctx)
        if is_main_process(self.ctx):
            logger.info("[trainer] done. checkpoints in %s", self.save_dir)

    def _infinite_dataloader(self):
        """Loop the dataloader forever, reseeding the distributed sampler each epoch."""
        epoch = 0
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.dataloader:
                yield batch
            epoch += 1

    # ------------------------------------------------------------------ forward
    def _build_conditioning(
        self,
        condition_attributes: list,
        device: torch.device,
        inner_model: "MoshiVis",
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build ``sum_condition`` for the LM's full-sequence training forward.

        Calls the model's ``condition_provider`` to encode the
        :class:`ConditionAttributes` list, then the ``fuser`` to route the
        results into the per-method slots. For training we collapse
        ``streaming_sum`` (which assumes ``seq_len=1`` per step at inference)
        into ``sum_condition`` by mean-pooling the reference embeddings
        over their sequence dimension; this gives the LM a single
        broadcast-applicable conditioning vector that exercises the ARC
        encoder's gradient path. Trade-off vs MoshiRAG's training is
        documented in ``training/README.md``.

        Returns ``(sum_condition, cross_condition_from_fuser)`` -- the
        latter is the fuser's ``cross`` output if any, which the caller
        merges with the image cross-attention KV (or rejects if both are
        set, matching the loader's collision check).
        """
        if (
            inner_model.condition_provider is None
            or inner_model.fuser is None
            or not condition_attributes
        ):
            return None, None

        prepared = inner_model.condition_provider.prepare(condition_attributes)
        condition_tensors = inner_model.condition_provider(prepared)

        sum_cond = inner_model.fuser.get_sum(condition_tensors)
        streaming_sum = inner_model.fuser.get_streaming_sum(condition_tensors)
        cross_cond = inner_model.fuser.get_cross(condition_tensors)

        # Collapse streaming_sum [B, T_ref, dim] → [B, 1, dim] by mean
        # pooling so it broadcasts over the LM's full training sequence
        # exactly like sum_condition does. This is a deliberate training-
        # time simplification of MoshiRAG's per-step semantics; the ARC
        # encoder still gets gradient signal but the per-step temporal
        # alignment is lost. Re-enable per-step at inference time -- the
        # streaming_sum queue in ``MoshiVisGen`` is unchanged.
        if streaming_sum is not None and streaming_sum.shape[1] > 0:
            streaming_sum_pooled = streaming_sum.mean(dim=1, keepdim=True)
            if sum_cond is None:
                sum_cond = streaming_sum_pooled
            else:
                sum_cond = sum_cond + streaming_sum_pooled.to(sum_cond)

        return sum_cond, cross_cond

    def _forward_one_microbatch(self, batch: CollatedBatch) -> torch.Tensor:
        """Forward one micro-batch and return the (unscaled) loss.

        Builds per-batch ``sum_condition`` via the condition provider
        and fuser (collapsing ``streaming_sum`` into ``sum`` for
        training; see :meth:`_build_conditioning`). Applies CFG
        conditioner dropout when ``cfg_dropout_p > 0`` -- with that
        probability, the entire batch's ConditionAttributes are
        replaced with their dropped (all-attributes-nulled) version
        before being passed through the provider/fuser. The model
        therefore learns both ``p(x | condition)`` and ``p(x | null)``
        distributions, which is what makes inference-time CFG sampling
        produce meaningful interpolations.
        """
        device = self.ctx.device
        input_ids = batch.input_ids.to(device)
        target_text = batch.target_text.to(device)
        loss_mask = batch.loss_mask.to(device)
        cross_attention_src = (
            batch.cross_attention_src.to(device)
            if batch.cross_attention_src is not None
            else None
        )

        # CFG dropout: replace the entire batch's ConditionAttributes with
        # their nullified version with probability ``cfg_dropout_p``. The
        # gating is per-batch (not per-example) because the provider's
        # batch handling is collated -- mixing pos+null in one batch
        # would require splitting the forward, which loses parallelism.
        # Per-example mixing can be achieved by simply running two
        # smaller batches in alternation, which the dataloader does
        # naturally over training steps.
        condition_attrs = batch.condition_attributes
        if (
            self.config.cfg_dropout_p > 0
            and torch.rand(()).item() < self.config.cfg_dropout_p
        ):
            condition_attrs = dropout_all_conditions(condition_attrs)

        # Inner model under DDP wrapper: forward_text is on the underlying
        # MoshiVis, not the DDP wrapper directly. ``.module`` accesses it.
        inner = (
            self.moshi_vis.module
            if hasattr(self.moshi_vis, "module")
            else self.moshi_vis
        )

        sum_condition, fuser_cross = self._build_conditioning(
            condition_attrs, device, inner
        )

        # Resolve the cross-attention source: prefer the explicit image
        # KV (vision path) over the fuser's ``cross`` output. The loader
        # rejects the collision at config-load time when vision XA is
        # also enabled, so by the time we get here the two paths are
        # mutually exclusive in practice.
        ca_src: Optional[torch.Tensor] = cross_attention_src
        if ca_src is None and fuser_cross is not None:
            ca_src = fuser_cross.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=self._dtype,
            enabled=self._dtype != torch.float32,
        ):
            transformer_out, text_logits, _gate = inner.forward_text(
                input_ids=input_ids,
                cross_attention_src=ca_src,
                sum_condition=sum_condition.to(device) if sum_condition is not None else None,
            )
            # text_logits comes out as [B, K_text=1, T, card]. Squeeze K_text.
            text_logits = text_logits.squeeze(1)
            loss = next_token_ce_loss(
                text_logits=text_logits,
                target_text_tokens=target_text,
                loss_mask=loss_mask,
            )
        return loss

    # ------------------------------------------------------------------ ckpt
    def _save(self) -> None:
        if not is_main_process(self.ctx):
            return
        path = self.save_dir / f"step_{self.step:08d}"
        path.mkdir(parents=True, exist_ok=True)
        inner_mv = (
            self.moshi_vis.module
            if hasattr(self.moshi_vis, "module")
            else self.moshi_vis
        )
        inner_ip = (
            self.image_proj.module
            if hasattr(self.image_proj, "module")
            else self.image_proj
        )
        torch.save(
            {
                "step": self.step,
                "moshi_vis": inner_mv.state_dict(),
                "image_proj": inner_ip.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.config.__dict__,
            },
            path / "ckpt.pt",
        )
        # Update the "latest" pointer so resume is one file lookup away.
        latest = self.save_dir / "latest"
        latest.unlink(missing_ok=True)
        latest.symlink_to(path.name)
        logger.info("[trainer] saved checkpoint to %s", path)

    def _resume(self, resume_dir: str) -> None:
        """Load the latest checkpoint inside ``resume_dir`` if any."""
        rdir = Path(resume_dir)
        latest = rdir / "latest"
        if latest.exists():
            target = (rdir / latest.readlink()).resolve()
            ckpt_path = target / "ckpt.pt"
        else:
            # Fall back to the highest step_* directory.
            candidates = sorted(rdir.glob("step_*"))
            if not candidates:
                logger.warning("[trainer] resume_dir has no checkpoints, starting fresh")
                return
            ckpt_path = candidates[-1] / "ckpt.pt"

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.step = int(state["step"])
        inner_mv = (
            self.moshi_vis.module
            if hasattr(self.moshi_vis, "module")
            else self.moshi_vis
        )
        inner_ip = (
            self.image_proj.module
            if hasattr(self.image_proj, "module")
            else self.image_proj
        )
        inner_mv.load_state_dict(state["moshi_vis"], strict=False)
        inner_ip.load_state_dict(state["image_proj"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if is_main_process(self.ctx):
            logger.info("[trainer] resumed from %s at step %d", ckpt_path, self.step)
