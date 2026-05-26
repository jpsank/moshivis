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

from kyuteye.training.collator import CollatedBatch, RagDataCollator
from kyuteye.training.dataset import RagJsonlDataset
from kyuteye.training.distributed import (
    DistributedContext,
    barrier,
    is_main_process,
)
from kyuteye.training.logging_hooks import Logger as _RunLogger
from kyuteye.training.logging_hooks import build_loggers
from kyuteye.training.training_module import TrainingForward

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
    # When True, the ARC encoder's reference embeddings are placed at LM
    # positions immediately following each <ret> token (faithful per-step
    # MoshiRAG semantics during training). When False (default), they are
    # mean-pooled and folded into sum_condition (simpler, broadcast-applied
    # across the full sequence). The model must have ``rag_token_id`` set
    # for the per-step path to fire; without it we fall back to mean-pool.
    per_step_streaming_sum: bool = False
    save_dir: str = "checkpoints"
    seed: int = 42
    dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"
    # Resume from this checkpoint dir if non-empty (loads latest step inside).
    resume_dir: Optional[str] = None
    # Comma-separated logger backends. ``python`` (default), ``tensorboard``,
    # ``wandb``. Multiple backends mirror metrics to each.
    log_backends: str = "python"
    tb_log_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


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

        # DDP wrapping: bundle MoshiVis + ImageProjection in a single
        # ``TrainingForward`` wrapper module so DDP's reducer sees the
        # whole forward graph in one ``__call__``. Bypassing
        # ``DDP.__call__`` (by accessing ``.module`` and calling
        # ``forward_text`` directly) skips ``prepare_for_backward``,
        # which under ``find_unused_parameters=True`` causes the next
        # ``backward`` to hang waiting for grads on params that
        # actually didn't participate. The wrapper has a single
        # ``forward(batch)`` method that the trainer calls through DDP.
        self.training_model = TrainingForward(
            moshi_vis=self.moshi_vis,
            image_proj=self.image_proj,
        )
        self.training_model = self._maybe_wrap_ddp(self.training_model)
        # ``moshi_vis`` / ``image_proj`` attributes are still useful for
        # checkpoint save/load; they're just no longer separately wrapped.

        # Optimizer + scheduler over the trainable params. The wrapper
        # holds both submodules so ``parameters()`` yields the union.
        trainable_params = [
            p for p in self.training_model.parameters() if p.requires_grad
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

        # Logging backends. Only rank 0 logs (the others would duplicate
        # everything onto WandB and confuse the metric step).
        if is_main_process(ctx):
            self._loggers: list[_RunLogger] = build_loggers(
                self.config.log_backends,
                tb_log_dir=self.config.tb_log_dir or str(self.save_dir / "tb"),
                wandb_project=self.config.wandb_project,
                wandb_run_name=self.config.wandb_run_name,
            )
            for lg in self._loggers:
                lg.log_hparams(
                    {
                        k: v
                        for k, v in self.config.__dict__.items()
                        if isinstance(v, (int, float, str, bool, type(None)))
                    }
                )
        else:
            self._loggers = []

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

            # Gradient clip + step. ``training_model`` holds both
            # ``moshi_vis`` and ``image_proj`` as submodules so a single
            # parameters() walk covers everything that DDP synced.
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p in self.training_model.parameters()
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
                metrics = {
                    "train/loss": accumulated_loss * self.config.grad_accum_steps,
                    "train/lr": lr,
                    "train/steps_per_sec": steps_per_sec,
                }
                for lg in self._loggers:
                    lg.log_scalars(self.step, metrics)
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
            for lg in self._loggers:
                lg.close()

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
    def _cfg_dropout_decision(self) -> bool:
        """Return a per-step CFG dropout decision that's identical across ranks.

        Each rank's RNG state diverges (we seed ``self.config.seed +
        ctx.rank`` per-rank for data shuffling). Using ``torch.rand(())``
        here would give different decisions per rank -> different
        forward shapes / dropped-conditions -> DDP all-reduce mismatch
        and corrupted gradients. We instead derive the decision
        deterministically from ``(step, seed)`` so every rank agrees.
        """
        if self.config.cfg_dropout_p <= 0:
            return False
        h = hash((self.step, self.config.seed)) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF) < self.config.cfg_dropout_p

    # NB: ``_build_streaming_sum_per_step`` and ``_build_conditioning``
    # used to live here; they've moved to
    # ``kyuteye.training.training_module.TrainingForward`` so they
    # execute inside DDP's ``__call__`` (which fixes the
    # ``find_unused_parameters`` hang the previous design caused).


    def _forward_one_microbatch(self, batch: CollatedBatch) -> torch.Tensor:
        """Forward one micro-batch and return the (unscaled) loss.

        Dispatches to ``self.training_model(...)``, which is the
        DDP-wrapped :class:`TrainingForward` -- a single ``forward``
        method that does conditioning + LM forward + masked CE loss.
        Routing through DDP's ``__call__`` is required for
        ``find_unused_parameters=True`` to correctly prime the reducer;
        bypassing it caused next-step ``backward()`` hangs.

        CFG dropout decision is made here (not inside the wrapper) so
        it can be deterministic across DDP ranks -- see
        :meth:`_cfg_dropout_decision`.
        """
        batch = batch.to(self.ctx.device)
        cfg_dropout = self._cfg_dropout_decision()
        with torch.autocast(
            device_type=self.ctx.device.type,
            dtype=self._dtype,
            enabled=self._dtype != torch.float32,
        ):
            loss = self.training_model(
                batch,
                cfg_dropout=cfg_dropout,
                per_step_streaming_sum=self.config.per_step_streaming_sum,
            )
        return loss

    # ------------------------------------------------------------------ ckpt
    def _training_inner(self) -> "TrainingForward":
        """Unwrap DDP if present and return the underlying ``TrainingForward``."""
        return (
            self.training_model.module
            if hasattr(self.training_model, "module")
            else self.training_model
        )

    def _save(self) -> None:
        if not is_main_process(self.ctx):
            return
        path = self.save_dir / f"step_{self.step:08d}"
        path.mkdir(parents=True, exist_ok=True)
        inner = self._training_inner()
        torch.save(
            {
                "step": self.step,
                "moshi_vis": inner.moshi_vis.state_dict(),
                "image_proj": inner.image_proj.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.config.__dict__,
            },
            path / "ckpt.pt",
        )
        # Update the "latest" pointer. Try a symlink first; fall back to a
        # plain text file naming the target dir (some HPC filesystems --
        # certain Lustre + GPFS configs -- don't permit symlinks). The
        # resume code reads either form.
        latest_link = self.save_dir / "latest"
        latest_link.unlink(missing_ok=True)
        try:
            latest_link.symlink_to(path.name)
        except (OSError, NotImplementedError) as e:
            logger.warning(
                "[trainer] symlink to latest failed (%s); using latest.txt fallback",
                e,
            )
            (self.save_dir / "latest.txt").write_text(path.name + "\n")
        logger.info("[trainer] saved checkpoint to %s", path)

    def _resume(self, resume_dir: str) -> None:
        """Load the latest checkpoint inside ``resume_dir`` if any.

        Resolution order for finding "latest":
          1. ``resume_dir/latest`` symlink (preferred, atomic via
             unlink + symlink_to).
          2. ``resume_dir/latest.txt`` plain text fallback (some
             HPC filesystems don't permit symlinks).
          3. Highest-numbered ``step_*`` subdirectory (sorted by name).
        """
        rdir = Path(resume_dir)
        latest_link = rdir / "latest"
        latest_txt = rdir / "latest.txt"
        ckpt_path: Optional[Path] = None
        if latest_link.is_symlink() or latest_link.exists():
            target = (rdir / latest_link.readlink()).resolve()
            ckpt_path = target / "ckpt.pt"
        elif latest_txt.exists():
            target_name = latest_txt.read_text().strip()
            ckpt_path = rdir / target_name / "ckpt.pt"
        else:
            candidates = sorted(rdir.glob("step_*"))
            if not candidates:
                logger.warning(
                    "[trainer] resume_dir %s has no checkpoints, starting fresh",
                    rdir,
                )
                return
            ckpt_path = candidates[-1] / "ckpt.pt"

        if not ckpt_path.exists():
            logger.warning("[trainer] resolved checkpoint %s does not exist", ckpt_path)
            return

        # ``weights_only=True`` accepts dict[str, Tensor] and the basic
        # numeric/string types we put in the ``config`` dict; safe to use
        # here. If the checkpoint format ever grows non-serializable
        # entries, flip this to False (with the usual security caveat).
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.step = int(state["step"])
        inner = self._training_inner()
        inner.moshi_vis.load_state_dict(state["moshi_vis"], strict=False)
        inner.image_proj.load_state_dict(state["image_proj"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if is_main_process(self.ctx):
            logger.info("[trainer] resumed from %s at step %d", ckpt_path, self.step)
