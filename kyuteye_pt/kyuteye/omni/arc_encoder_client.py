"""Client for an external ARC reference encoder service.

The ARC encoder is the same HTTP service used by kyutai-labs/moshi-rag: it
takes a reference string and returns a safetensors-encoded ``[1, T, dim]``
tensor where ``dim`` matches the LLM hidden dimension and ``T`` is the
number of per-step rows that will be additively merged into the LM input
embeddings (see ``MoshiVisGen.push_streaming_sum_tensor`` /
``consume_streaming_sum_step``).

The default endpoint and timeout mirror MoshiRAG's defaults. Configure with
``REFERENCE_ENCODER_URL`` env var or ``--omni-arc-encoder-url`` on the
server CLI.

Note: this only ports the *client* half. Running the encoder itself is out
of scope here; see https://github.com/kyutai-labs/moshi-rag for the server
build instructions and the ARC encoder Dockerfile.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_arc_encoder_url(default: Optional[str] = None) -> Optional[str]:
    """Resolve the ARC encoder URL from env or an explicit override."""
    return default or os.environ.get("REFERENCE_ENCODER_URL")


async def encode_reference_async(
    text: str,
    *,
    encoder_url: str,
    timeout: float = 30.0,
) -> torch.Tensor:
    """POST ``text`` to ``{encoder_url}/embed`` and return the embedding tensor.

    Returns a ``[1, T, dim]`` tensor (matches the on-the-wire shape from the
    MoshiRAG reference encoder). Raises on HTTP / decode errors so the caller
    can surface a ``[RET_FAILED]`` to the UI.
    """
    if text is None:
        text = ""

    # Lazy imports so the rest of the omni package keeps working without the
    # ARC encoder being available.
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "The `httpx` package is required to talk to the ARC reference "
            "encoder. Install with `pip install httpx`."
        ) from e
    try:
        from safetensors.torch import load as load_safetensors
    except ImportError as e:
        raise ImportError(
            "The `safetensors` package is required to decode ARC encoder "
            "responses. Install with `pip install safetensors`."
        ) from e

    logger.info(
        "[ARC] POST %s/embed text=%r", encoder_url, text[:80]
    )
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{encoder_url}/embed",
            json={"text": text},
        )
        response.raise_for_status()

    payload = load_safetensors(response.content)
    if "tensor" not in payload:
        raise ValueError("ARC encoder response is missing the `tensor` entry")
    tensor = payload["tensor"]
    logger.info(
        "[ARC] received %s %s in %.3fs",
        tuple(tensor.shape),
        tensor.dtype,
        time.monotonic() - t0,
    )
    return tensor
