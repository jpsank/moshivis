#!/usr/bin/env python
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP server wrapping the in-repo ARC encoder.

Speaks the same protocol the MoshiRAG ARC encoder service does -- a
``POST /embed`` endpoint that takes ``{"text": "..."}`` and returns the
embedding as a safetensors payload with a single ``tensor`` entry shaped
``[1, T, dim]``. The Omni runtime's
:func:`kyuteye.omni.arc_encoder_client.encode_reference_async` is the
matching client.

This is the second of two ways to run the encoder in this branch:

* :func:`kyuteye.omni.arc_encoder_local.encode_reference_local_async`
  bypasses HTTP entirely and calls the conditioner in-process. Use for
  single-process inference (the default ``server`` already auto-detects
  this path when ``rag.enabled=true`` is set in the YAML).
* This script wraps the same conditioner over HTTP. Use it when you
  want to (a) run the LM and encoder on separate GPUs, (b) batch
  encoder requests across multiple model servers, or (c) match the
  exact MoshiRAG deployment topology.

The encoder weights are loaded the same way ``get_moshi_vis`` loads them:
the conditioner spec is parsed from the YAML's ``rag.conditioners`` block,
and if ``hf_repo`` is set on the ARC entry, ``load_weights()`` pulls
``model.safetensors`` from that HuggingFace repo.

Usage:

.. code-block:: bash

    # Standalone, listening on :8089
    python scripts/serve_arc_encoder.py \\
        --kyuteye-config configs/moshika-vis.yaml \\
        --host 0.0.0.0 \\
        --port 8089

    # Point the main server at it
    export REFERENCE_ENCODER_URL=http://localhost:8089
    uv run server configs/moshika-vis.yaml --omni-arc-encoder-mode http
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import fire
import torch
from aiohttp import web
from safetensors.torch import save as save_safetensors

from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.loaders import build_condition_provider

logger = logging.getLogger(__name__)


def _load_arc_conditioner(
    cfg: KyuteyeConfig,
    *,
    device: str,
    name: str = "reference_with_time",
):
    """Build a :class:`ConditionProvider` containing only the ARC encoder.

    Reuses ``build_condition_provider`` so the construction path matches
    what the main server does -- if a config works for ``server``, it
    works here.

    :raises ValueError: if ``rag.enabled`` is False, the conditioner
        registry doesn't contain ``name``, or the named entry isn't of
        ``type: arc``.
    """
    rag = getattr(cfg, "rag", None)
    if rag is None or not getattr(rag, "enabled", False):
        raise ValueError(
            "The YAML config does not enable RAG (rag.enabled=false). "
            "Enable it and configure rag.conditioners.reference_with_time "
            "with type: arc to serve embeddings."
        )
    conditioners = getattr(rag, "conditioners", None) or {}
    if name not in conditioners:
        raise ValueError(
            f"rag.conditioners does not contain {name!r}; "
            f"available: {list(conditioners)}"
        )
    if conditioners[name].get("type") != "arc":
        raise ValueError(
            f"rag.conditioners.{name}.type must be 'arc' to serve "
            f"embeddings; got {conditioners[name].get('type')!r}"
        )
    # Build a minimal provider containing only the ARC entry. The
    # ``output_dim`` matches the LM's hidden dim from the rest of the
    # config so the produced tensors are directly compatible with the
    # client-side ``update_streaming_sum_tensor`` call. ``cfg.moshi.dim``
    # is the LM hidden dimension (lives on the nested ``moshi``
    # subconfig per kyuteye_pt/kyuteye/config/subconfigs.py:89).
    arc_only = {name: conditioners[name]}
    provider = build_condition_provider(
        arc_only,
        output_dim=cfg.moshi.dim,
        device=device,
    )
    cond = provider.conditioners[name]
    if hasattr(cond, "load_weights"):
        cond.load_weights()
    cond.eval()
    return cond


def _encode(cond, text: str) -> torch.Tensor:
    """Run the conditioner forward and normalize the output shape.

    Returns a CPU tensor (``[1, T, dim]``, ``float32``) ready for
    safetensors serialization. CPU + float32 because the client does
    ``load_safetensors`` and then ``update_streaming_sum_tensor`` which
    handles dtype/device conversion itself.
    """
    with torch.no_grad():
        inputs = cond.prepare([text or ""])
        # pylint: disable=protected-access
        result = cond._get_condition(inputs)
    tensor = result.condition
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def main(
    kyuteye_config: str,
    host: str = "0.0.0.0",
    port: int = 8089,
    device: str = "cuda",
    conditioner_name: str = "reference_with_time",
    log_level: str = "INFO",
) -> None:
    """Launch the ARC encoder HTTP server.

    :param kyuteye_config: Path to the same YAML the main ``server``
        uses (e.g. ``configs/moshika-vis.yaml``). The ``rag`` block of
        the config determines which conditioner is served.
    :param host, port: Bind address. Default ``0.0.0.0:8089``.
    :param device: ``cuda`` (default) or ``cpu``. CPU is fine for small
        request rates but the encoder is a 26-layer transformer; expect
        ~hundreds of ms per request on CPU.
    :param conditioner_name: Key under ``rag.conditioners`` to serve.
        Defaults to MoshiRAG's ``reference_with_time``.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = KyuteyeConfig.from_yml(kyuteye_config)
    logger.info("[ARC] loading conditioner %s on %s", conditioner_name, device)
    cond = _load_arc_conditioner(cfg, device=device, name=conditioner_name)
    logger.info("[ARC] ready: out_dim=%d", cond.bridge_module_params["out_dim"])

    async def handle_embed(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": f"invalid JSON: {e}"}, status=400
            )
        text = payload.get("text")
        if not isinstance(text, str):
            return web.json_response(
                {"error": "missing or non-string 'text' field"}, status=400
            )
        try:
            tensor = _encode(cond, text)
        except Exception as e:
            logger.exception("[ARC] encoder forward failed")
            return web.json_response(
                {"error": f"encoder failed: {e}"}, status=500
            )
        body = save_safetensors({"tensor": tensor})
        return web.Response(body=body, content_type="application/octet-stream")

    async def handle_health(request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "status": "ok",
                "out_dim": int(cond.bridge_module_params["out_dim"]),
                "device": str(device),
            }
        )

    app = web.Application()
    app.router.add_post("/embed", handle_embed)
    app.router.add_get("/health", handle_health)
    logger.info("[ARC] listening on http://%s:%d/embed", host, port)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
