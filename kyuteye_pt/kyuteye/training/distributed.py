"""Distributed training helpers (DDP + Slurm).

Wraps the small amount of ``torch.distributed`` boilerplate every Slurm
job needs:

* :func:`init_distributed` reads ``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK``
  from the environment (Slurm or torchrun sets these) and calls
  ``torch.distributed.init_process_group``.
* :func:`is_main_process` short-cut for "should I log / save / etc.".
* :func:`barrier` wrap that no-ops outside DDP.

We deliberately don't ship a Slurm-specific launcher script -- ``torchrun``
on each node is the modern recommendation and matches what every HPC
cluster's documentation suggests these days.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """Snapshot of the distributed setup. Returned by :func:`init_distributed`."""

    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(
    backend: str = "nccl",
    *,
    timeout_seconds: int = 1800,
) -> DistributedContext:
    """Initialize ``torch.distributed`` if launched under torchrun/Slurm.

    Single-process callers (no env vars) get a degenerate context with
    ``world_size=1`` and CUDA device 0 if available, CPU otherwise. The
    trainer code path is identical for both cases -- DDP wrapping is a
    no-op at world_size 1.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            from datetime import timedelta

            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
        logger.info(
            "[dist] initialized: world_size=%d rank=%d local_rank=%d device=%s",
            world_size,
            rank,
            local_rank,
            device,
        )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_distributed = False
        logger.info("[dist] single-process mode on %s", device)

    return DistributedContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        is_distributed=is_distributed,
    )


def is_main_process(ctx: Optional[DistributedContext] = None) -> bool:
    """Return True on rank 0 (or always in single-process mode)."""
    if ctx is None:
        return int(os.environ.get("RANK", "0")) == 0
    return ctx.is_main


def barrier(ctx: Optional[DistributedContext] = None) -> None:
    """Synchronize all ranks. No-op in single-process mode."""
    if ctx is not None and not ctx.is_distributed:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    """Tear down the process group on exit. Safe to call unconditionally."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
