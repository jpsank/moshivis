"""Single-forward training wrapper so DDP can correctly track used parameters.

Why this exists: MoshiVis exposes ``forward_text`` (training-mode forward
of the LM) and ``forward_depformer`` (per-codebook depformer step) as
separate methods. Neither is named ``forward``. PyTorch ``DistributedDataParallel``
hooks its reducer's ``prepare_for_backward(...)`` into its own ``__call__``,
which calls ``module.forward(...)``. If the trainer bypasses
``__call__`` and invokes ``forward_text`` directly on the wrapped
``module``, the reducer is never primed. With ``find_unused_parameters=True``
(required for our partial-freeze recipes) the reducer can then hang on
the next ``backward()``, waiting for gradients on parameters it didn't
see in this iteration's autograd graph.

This wrapper bundles the conditioning + LM forward + loss into a single
``forward(batch)`` method that the trainer calls through DDP. The
``ImageProjection`` is also a submodule here so its parameters
participate in DDP's reduce when the freeze recipe includes them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from kyuteye.conditioners import dropout_all_conditions
from kyuteye.training.collator import CollatedBatch
from kyuteye.training.loss import next_token_ce_loss

if TYPE_CHECKING:
    from kyuteye.models.image_projection import ImageProjection
    from kyuteye.models.moshivis import MoshiVis

logger = logging.getLogger(__name__)


class TrainingForward(nn.Module):
    """DDP-safe training wrapper around (MoshiVis, ImageProjection).

    Owns both submodules so a single DDP wrap covers all trainable
    parameters across them. ``forward`` accepts a :class:`CollatedBatch`
    plus optional CFG / streaming-sum flags and returns a scalar loss.
    All conditioning machinery (provider, fuser, per-step streaming-sum
    alignment) runs inside this method, ensuring DDP's autograd-graph
    scan in :meth:`DistributedDataParallel.forward` sees every used
    parameter.
    """

    def __init__(
        self,
        moshi_vis: "MoshiVis",
        image_proj: "ImageProjection",
    ) -> None:
        super().__init__()
        self.moshi_vis = moshi_vis
        self.image_proj = image_proj

    def _build_conditioning(
        self,
        condition_attributes: list,
        text_input_ids: Optional[torch.Tensor],
        *,
        per_step_streaming_sum: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the conditioner provider + fuser and return per-method slots.

        See ``Trainer._build_conditioning`` in earlier revisions for the
        algorithm; this is the same logic, moved into the wrapper so it
        executes inside ``DDP.__call__``'s autograd scope.
        """
        if (
            self.moshi_vis.condition_provider is None
            or self.moshi_vis.fuser is None
            or not condition_attributes
        ):
            return None, None, None

        prepared = self.moshi_vis.condition_provider.prepare(condition_attributes)
        condition_tensors = self.moshi_vis.condition_provider(prepared)

        sum_cond = self.moshi_vis.fuser.get_sum(condition_tensors)
        streaming_sum = self.moshi_vis.fuser.get_streaming_sum(condition_tensors)
        cross_cond = self.moshi_vis.fuser.get_cross(condition_tensors)

        streaming_sum_per_step: Optional[torch.Tensor] = None
        if streaming_sum is not None and streaming_sum.shape[1] > 0:
            rag_token_id = getattr(self.moshi_vis, "rag_token_id", None)
            if (
                per_step_streaming_sum
                and text_input_ids is not None
                and rag_token_id is not None
            ):
                streaming_sum_per_step = _build_streaming_sum_per_step(
                    text_input_ids=text_input_ids,
                    reference_embeddings=streaming_sum,
                    rag_token_id=rag_token_id,
                )
            else:
                streaming_sum_pooled = streaming_sum.mean(dim=1, keepdim=True)
                if sum_cond is None:
                    sum_cond = streaming_sum_pooled
                else:
                    sum_cond = sum_cond + streaming_sum_pooled.to(sum_cond)

        return sum_cond, cross_cond, streaming_sum_per_step

    def forward(
        self,
        batch: CollatedBatch,
        *,
        cfg_dropout: bool = False,
        per_step_streaming_sum: bool = False,
    ) -> torch.Tensor:
        """Compute training loss for one batch.

        :param cfg_dropout: When ``True``, the batch's
            ``ConditionAttributes`` are replaced with their nulled
            version before the conditioning is built (CFG null branch).
            The trainer makes this decision rank-deterministically; the
            wrapper itself is stateless.
        :param per_step_streaming_sum: Plumbed through to
            :meth:`_build_conditioning`. When ``True``, reference
            embedding rows are placed at the LM positions after each
            ``<ret>`` token.
        :return: Scalar token-weighted CE loss on the moshi-turn
            positions of the text codebook.
        """
        # CFG conditioner dropout (decision made by caller for DDP-determinism).
        condition_attrs = batch.condition_attributes
        if cfg_dropout:
            condition_attrs = dropout_all_conditions(condition_attrs)

        # Conditioning -- runs INSIDE this DDP-wrapped forward so the
        # reducer sees the condition_provider's parameters in the graph.
        sum_condition, fuser_cross, streaming_sum_per_step = self._build_conditioning(
            condition_attrs,
            batch.input_ids[:, 0],
            per_step_streaming_sum=per_step_streaming_sum,
        )

        # Resolve cross-attention source: image KV wins over fuser-cross
        # (matches loader-level collision check).
        ca_src: Optional[torch.Tensor] = batch.cross_attention_src
        if ca_src is None and fuser_cross is not None:
            ca_src = fuser_cross

        _, text_logits, _gate = self.moshi_vis.forward_text(
            input_ids=batch.input_ids,
            cross_attention_src=ca_src,
            sum_condition=sum_condition,
            streaming_sum_condition=streaming_sum_per_step,
        )
        # text_logits is [B, K_text=1, T, card]; squeeze the codebook dim.
        text_logits = text_logits.squeeze(1)
        return next_token_ce_loss(
            text_logits=text_logits,
            target_text_tokens=batch.target_text,
            loss_mask=batch.loss_mask,
        )


def _build_streaming_sum_per_step(
    *,
    text_input_ids: torch.Tensor,
    reference_embeddings: torch.Tensor,
    rag_token_id: int,
) -> torch.Tensor:
    """Place reference rows at LM positions after each ``<ret>`` token.

    See ``Trainer._build_streaming_sum_per_step`` (earlier revisions) for
    the design rationale; this is the same code, moved here so it lives
    next to its only caller and the wrapper class can call it without a
    cross-module trainer reference.
    """
    B, T_lm = text_input_ids.shape
    _, T_ref, dim = reference_embeddings.shape
    out = torch.zeros(
        B,
        T_lm,
        dim,
        device=reference_embeddings.device,
        dtype=reference_embeddings.dtype,
    )
    ret_mask = text_input_ids == rag_token_id
    for b in range(B):
        positions = ret_mask[b].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        ret_pos = int(positions[0].item())
        start = ret_pos + 1
        end = min(start + T_ref, T_lm)
        length = max(0, end - start)
        if length > 0:
            out[b, start:end] = reference_embeddings[b, :length]
    return out
