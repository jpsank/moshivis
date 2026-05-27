"""Direct in-process ARC encoder, bypassing the HTTP service.

When the LM is loaded with ``rag.enabled=True``, an
:class:`kyuteye.conditioners.arc_encoder.ArcEncoderConditioner` is already
instantiated inside ``moshi_vis.condition_provider``. This module exposes
a small wrapper that calls it directly, producing the same ``[1, T, dim]``
tensor the HTTP
:func:`kyuteye.omni.arc_encoder_client.encode_reference_async` returns.

Use this for single-process inference where running a separate ARC encoder
HTTP service is unnecessary overhead. Trade-off: the LM and the encoder
share the same GPU memory budget. For production scaling where you want
the LM and encoder on different GPUs (or different hosts), prefer the
HTTP service path -- ``scripts/serve_arc_encoder.py`` wraps the same
conditioner over ``POST /embed``.

Both paths produce identical tensors when given the same input; the
choice between them is purely an operational one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from kyuteye.conditioners.arc_encoder import ArcEncoderConditioner

logger = logging.getLogger(__name__)


def find_arc_conditioner(
    moshi_vis: object,
    *,
    name: str = "reference_with_time",
) -> Optional["ArcEncoderConditioner"]:
    """Look up the ARC encoder conditioner on a loaded model.

    Returns ``None`` if RAG isn't enabled on this model, if no conditioner
    is registered under ``name``, or if xformers isn't installed (the
    :class:`ArcEncoderConditioner` import fails). Callers can use the
    ``None`` return to fall back to the HTTP encoder path.

    :param moshi_vis: A :class:`MoshiVisGen` (preferred) or the raw
        :class:`MoshiVis` LM. We descend via ``.lm_model`` if present.
    :param name: Conditioner registry key, defaults to MoshiRAG's
        ``"reference_with_time"``.
    """
    try:
        from kyuteye.conditioners.arc_encoder import ArcEncoderConditioner
    except ImportError:
        return None
    lm = getattr(moshi_vis, "lm_model", moshi_vis)
    cp = getattr(lm, "condition_provider", None)
    if cp is None:
        return None
    conds = getattr(cp, "conditioners", None)
    if not conds or name not in conds:
        return None
    cond = conds[name]
    if not isinstance(cond, ArcEncoderConditioner):
        logger.warning(
            "[ARC local] conditioner %r is %s, not ArcEncoderConditioner -- "
            "skipping local-encode path",
            name,
            type(cond).__name__,
        )
        return None
    return cond


def encode_reference_local(
    text: str,
    *,
    conditioner: "ArcEncoderConditioner",
) -> torch.Tensor:
    """Synchronous in-process encode: ``text -> [1, T, dim]`` tensor.

    Calls the conditioner's ``prepare`` + ``_get_condition`` pipeline,
    detaches the result (no autograd graph carried back into the runtime),
    and normalizes the shape to a leading batch dim of 1 so the result
    drops directly into :meth:`MoshiVisGen.update_streaming_sum_tensor`.

    The encoder runs on whichever device the conditioner was constructed
    on (typically the same CUDA device as the LM).
    """
    if text is None:
        text = ""
    with torch.no_grad():
        inputs = conditioner.prepare([text])
        # pylint: disable=protected-access -- this is the conditioner's
        # canonical forward path; ConditionProvider invokes it the same way.
        cond = conditioner._get_condition(inputs)
    tensor = cond.condition
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.detach()


async def encode_reference_local_async(
    text: str,
    *,
    conditioner: "ArcEncoderConditioner",
) -> torch.Tensor:
    """Async wrapper around :func:`encode_reference_local`.

    Runs the encoder forward via ``asyncio.to_thread`` so it doesn't block
    the event loop. The forward still happens on the same GPU as the LM;
    this just yields the Python thread so concurrent WebSocket messages
    keep flowing.

    Signature mirrors
    :func:`kyuteye.omni.arc_encoder_client.encode_reference_async` so the
    server's ``push_to_streaming_sum`` can swap one for the other.
    """
    return await asyncio.to_thread(
        encode_reference_local, text, conditioner=conditioner
    )
