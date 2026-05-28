"""Logging backends for the trainer (Python logging / TensorBoard / WandB).

Pluggable per the same pattern as the TTS interface in
``audio_preprocess.py``: an abstract :class:`Logger` defines the
``log_scalars`` / ``close`` contract, and three concrete subclasses
ship out of the box. The trainer accepts a list of loggers, calls
each one's ``log_scalars`` each ``--log-every`` step, and on shutdown
closes each one. Pass multiple to mirror to several backends at once.

Imports of ``torch.utils.tensorboard`` and ``wandb`` are lazy so the
module loads without either installed; the constructor of the
respective subclass raises a clear ImportError if missing.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Logger(abc.ABC):
    """Abstract logger. Override :meth:`log_scalars` and :meth:`close`."""

    @abc.abstractmethod
    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        """Log a flat dict of ``{name: value}`` scalars at ``step``."""

    def log_text(self, step: int, key: str, text: str) -> None:
        """Log a text fragment (defaults to a single info-level log line)."""
        logger.info("[step %d] %s: %s", step, key, text[:200])

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Log run hyperparameters once at training start."""
        logger.info("[hparams] %s", hparams)

    @abc.abstractmethod
    def close(self) -> None:
        """Flush + tear down. Always called from the trainer's ``finally``."""


class PythonLogger(Logger):
    """Default backend: prints a one-line summary to Python's ``logging``.

    Already what the trainer was doing; this subclass just formalizes
    it so the multi-backend dispatch loop works uniformly.
    """

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        msg = " ".join(f"{k}={v:.4f}" for k, v in scalars.items())
        logger.info("[step %d] %s", step, msg)

    def close(self) -> None:
        pass


class TensorBoardLogger(Logger):
    """Write scalars to a TensorBoard ``SummaryWriter`` under ``log_dir``."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "TensorBoardLogger requires the ``tensorboard`` package. "
                "Install with ``pip install tensorboard``."
            ) from e
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        for k, v in scalars.items():
            self._writer.add_scalar(k, v, step)

    def log_text(self, step: int, key: str, text: str) -> None:
        self._writer.add_text(key, text, step)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        # add_hparams needs a metric; we use a sentinel zero so the run
        # shows up in the hparam tab without polluting the scalar pane.
        sanitized = {
            k: v
            for k, v in hparams.items()
            if isinstance(v, (int, float, str, bool))
        }
        self._writer.add_hparams(sanitized, {"hparam/sentinel": 0.0})

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandbLogger(Logger):
    """Mirror metrics to a Weights & Biases run.

    Initialization is intentionally minimal -- pass through any extra
    kwargs to ``wandb.init`` via ``init_kwargs``. The most common one
    is ``mode="offline"`` for HPC nodes without internet, in which
    case you ``wandb sync <run_dir>`` from a login node afterwards.
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        *,
        init_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "WandbLogger requires the ``wandb`` package. "
                "Install with ``pip install wandb``."
            ) from e
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            **(init_kwargs or {}),
        )

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        self._wandb.log(scalars, step=step)

    def log_text(self, step: int, key: str, text: str) -> None:
        self._wandb.log({key: self._wandb.Html(text)}, step=step)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(hparams, allow_val_change=True)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


def build_loggers(spec: str, **kwargs: Any) -> list[Logger]:
    """Parse a comma-separated logger spec into a list of :class:`Logger`.

    Recognized backends: ``python`` (default), ``tensorboard``, ``wandb``.
    Keyword args are routed by name prefix: ``tb_log_dir``, ``wandb_project``,
    ``wandb_run_name``. The trainer entry point passes the relevant ones
    after argparse.

    Example: ``"python,tensorboard"`` enables both. Always includes
    PythonLogger so the Slurm job's stdout has at least one line per
    log step.
    """
    backends = {b.strip().lower() for b in spec.split(",") if b.strip()}
    loggers: list[Logger] = []
    if "python" in backends or not backends:
        loggers.append(PythonLogger())
    if "tensorboard" in backends:
        log_dir = kwargs.get("tb_log_dir") or "runs"
        loggers.append(TensorBoardLogger(log_dir))
    if "wandb" in backends:
        loggers.append(
            WandbLogger(
                project=kwargs.get("wandb_project") or "moshivis-rag",
                run_name=kwargs.get("wandb_run_name"),
            )
        )
    return loggers
