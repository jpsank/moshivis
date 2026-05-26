"""Next-token cross-entropy loss for MoshiVis text generation.

The combined fine-tune teaches the model to predict moshi's text turns
(in the model's SentencePiece vocabulary) conditioned on the audio
codes, image, and reference. The trainer flattens each example's
moshi-turn tokens into a target sequence; this loss compares those
targets against the model's predicted text logits position-by-position,
ignoring positions outside the moshi turns (user turns, padding,
reference markers, the special ``<ret>`` token if you mask it out for
training).

The loss is intentionally a plain CE -- no z-loss, no label smoothing
by default. Those are easy to add in the trainer if a particular
fine-tune benefits from them; we don't bake them in here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def next_token_ce_loss(
    text_logits: torch.Tensor,
    target_text_tokens: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    shift_targets: bool = True,
) -> torch.Tensor:
    """Standard masked next-token CE loss.

    :param text_logits: ``[B, T, text_card]`` -- ``MoshiVis.forward_text``
        returns ``[B, K_text=1, T, text_card]``; the caller should squeeze
        the codebook dim before passing in.
    :param target_text_tokens: ``[B, T]`` long tensor of target token ids.
        Use ``ignore_index`` (default ``-100``) for positions to skip
        (user turns, padding, reference text -- whichever the training
        recipe ignores).
    :param loss_mask: Optional ``[B, T]`` bool tensor. When provided,
        positions where ``loss_mask`` is ``False`` are also ignored
        regardless of their target id. This is the simpler alternative
        to setting their target to ``ignore_index``.
    :param ignore_index: Token id treated as "no loss" by
        ``F.cross_entropy``. Default ``-100``.
    :param label_smoothing: Passed through to ``F.cross_entropy``.
    :param shift_targets: When True (default), align logits and targets
        for next-token prediction: ``logits[:, t]`` is scored against
        ``targets[:, t+1]``. Implemented by slicing ``logits[:, :-1]``
        and ``targets[:, 1:]`` (and ``loss_mask[:, 1:]``). When False,
        compare position-by-position -- only correct for tasks where
        the model's output at position ``t`` is already aligned with
        the target at position ``t`` (e.g. classification heads or
        pre-shifted data).

    :return: Scalar tensor, mean CE over the non-ignored positions.
    """
    assert text_logits.dim() == 3, (
        f"expected text_logits [B, T, card], got {tuple(text_logits.shape)}"
    )
    assert target_text_tokens.dim() == 2, (
        f"expected target_text_tokens [B, T], got {tuple(target_text_tokens.shape)}"
    )
    B, T, card = text_logits.shape
    assert target_text_tokens.shape == (B, T), (
        f"target shape {tuple(target_text_tokens.shape)} != logits batch/time {(B, T)}"
    )

    if shift_targets:
        # Next-token alignment: logits[:, t] predicts target[:, t+1].
        if T < 2:
            # Degenerate batch (single position) has no next-token signal.
            return torch.zeros(
                (),
                device=text_logits.device,
                dtype=torch.float32,
                requires_grad=text_logits.requires_grad,
            )
        text_logits = text_logits[:, :-1].contiguous()
        target_text_tokens = target_text_tokens[:, 1:].contiguous()
        if loss_mask is not None:
            assert loss_mask.shape == (B, T), (
                f"loss_mask shape {tuple(loss_mask.shape)} != {(B, T)}"
            )
            loss_mask = loss_mask[:, 1:].contiguous()
        B, T, card = text_logits.shape  # update for downstream asserts

    targets = target_text_tokens
    if loss_mask is not None:
        assert loss_mask.shape == (B, T), (
            f"loss_mask shape {tuple(loss_mask.shape)} != {(B, T)}"
        )
        targets = torch.where(
            loss_mask, targets, torch.full_like(targets, ignore_index)
        )

    # Flatten so F.cross_entropy sees [N, C] vs [N].
    flat_logits = text_logits.reshape(-1, card)
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(
        flat_logits.float(),
        flat_targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
