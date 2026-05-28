"""Inject retrieved text into MoshiVis' cross-attention input.

MoshiVis is trained with image patch embeddings as the cross-attention
source (``ca_src``) and exposes :meth:`MoshiVisGen.precompte_ca_kv` to
project an embedding tensor to (K, V) pairs that the shared cross-attention
layer can consume directly. We reuse that path for retrieved text: tokenize
the reference with SentencePiece, look up the LLM's text-embedding table,
and project the resulting ``(1, seq_len, llm_dim)`` tensor through
``precompte_ca_kv``.

The result is concatenated with the image (K, V) along the sequence
dimension. From the model's point of view this is just a longer
cross-attention key/value cache.

This is **experimental conditioning**: MoshiVis was not trained to attend
to text embeddings, so the practical effect on generation quality is
limited. We surface it behind a flag (``enable_xa_injection``); even when
it is off the retrieved text is always shown to the user via the
WebSocket text channel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    import sentencepiece

    from kyuteye.models.moshivis import MoshiVisGen

logger = logging.getLogger(__name__)


def encode_text_to_ca_kv(
    text: str,
    *,
    moshi_vis: "MoshiVisGen",
    tokenizer: "sentencepiece.SentencePieceProcessor",
    device: str | torch.device,
    dtype: torch.dtype,
    max_tokens: int = 256,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize ``text`` and project it through the LLM text embedding +
    cross-attention input projection. Returns ``(K, V)`` or ``None`` for
    empty text.
    """
    if not text:
        return None
    ids = tokenizer.encode(text)  # type: ignore[no-untyped-call]
    if not ids:
        return None
    ids = ids[:max_tokens]
    ids_t = torch.tensor([ids], device=device, dtype=torch.long)
    text_emb = moshi_vis.lm_model.llm.text_emb(ids_t)  # [1, seq, llm_dim]
    text_emb = text_emb.to(dtype=dtype)
    k, v = moshi_vis.precompte_ca_kv(text_emb)
    return k.to(dtype), v.to(dtype)


def _cat_kv(
    base: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
    extra: Tuple[torch.Tensor, torch.Tensor] | None,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
    """Concatenate two precomputed-KV pairs along the sequence dimension.

    ``base`` may be either a precomputed ``(K, V)`` tuple, a raw embedding
    tensor, or ``None``. Concatenation is only supported when both sides
    are ``(K, V)`` tuples; otherwise we keep ``base`` unchanged and log a
    warning (this happens if ``base`` is a raw tensor from a code path
    that hasn't been migrated to precomputed KV yet).
    """
    if extra is None:
        return base
    if base is None:
        return extra
    if isinstance(base, torch.Tensor):
        logger.warning(
            "[Omni] cannot concat extra KV onto a raw embedding ca_src; "
            "extra context will be ignored on this step"
        )
        return base
    bk, bv = base
    ek, ev = extra
    # Both are shaped like [B, num_heads * head_dim, seq, ...] depending on
    # whether the projection was packed or not; concatenation along the seq
    # axis works for both layouts because the seq axis is the second-to-last.
    return (torch.cat([bk, ek], dim=-2), torch.cat([bv, ev], dim=-2))


class ContextInjector:
    """Maintains the running cross-attention augmentation for one channel.

    Each channel owns one instance. On every model step the server asks
    :meth:`current_ca_src` for the cross-attention input -- it always
    returns the image KVs concatenated with whatever retrieval / tool
    context has accumulated since the last reset.
    """

    def __init__(
        self,
        *,
        moshi_vis: "MoshiVisGen",
        tokenizer: "sentencepiece.SentencePieceProcessor",
        device: str | torch.device,
        dtype: torch.dtype,
        enabled: bool = True,
        max_tokens_per_chunk: int = 256,
    ) -> None:
        self.moshi_vis = moshi_vis
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.enabled = enabled
        self.max_tokens_per_chunk = max_tokens_per_chunk

        self._image_kv: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None
        self._text_kv: Tuple[torch.Tensor, torch.Tensor] | None = None

    def set_image_kv(
        self, kv: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None
    ) -> None:
        self._image_kv = kv

    def add_text(self, text: str, *, role: str = "context") -> None:
        """Append a text chunk to the running cross-attention augmentation."""
        if not self.enabled or not text:
            return
        framed = f"[{role}] {text}\n" if role else text
        extra = encode_text_to_ca_kv(
            framed,
            moshi_vis=self.moshi_vis,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=self.dtype,
            max_tokens=self.max_tokens_per_chunk,
        )
        if extra is None:
            return
        if self._text_kv is None:
            self._text_kv = extra
        else:
            self._text_kv = (
                torch.cat([self._text_kv[0], extra[0]], dim=-2),
                torch.cat([self._text_kv[1], extra[1]], dim=-2),
            )

    def reset_text(self) -> None:
        self._text_kv = None

    def current_ca_src(self) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
        return _cat_kv(self._image_kv, self._text_kv)
