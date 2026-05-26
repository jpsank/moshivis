# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Load moshi-vis neccessary components."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from kyuteye.conditioners import (
    BaseConditioner,
    ConditionAttributes,
    ConditionFuser,
    ConditionProvider,
    ConditionTensors,
    LUTConditioner,
    TensorConditioner,
)
from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.image_projection import ImageProjection
from kyuteye.models.moshivis import MoshiVis, MoshiVisGen

logger = logging.getLogger(__name__)


_CONDITIONER_CLASSES: Dict[str, type[BaseConditioner]] = {
    "lut": LUTConditioner,
    "tensor": TensorConditioner,
}


def build_condition_provider(
    spec: Dict[str, Dict[str, Any]],
    *,
    output_dim: int,
    device: str | torch.device,
) -> ConditionProvider:
    """Instantiate a :class:`ConditionProvider` from a YAML config dict.

    ``spec`` maps attribute name -> conditioner kwargs (with a ``type`` key
    naming one of :data:`_CONDITIONER_CLASSES`). The conditioner's
    ``output_dim`` is fixed to the LLM hidden size so the fuser tensors are
    directly additive on the input embeddings.
    """
    modules: Dict[str, BaseConditioner] = {}
    for name, raw in spec.items():
        kwargs = dict(raw)
        ctype = kwargs.pop("type", None)
        if ctype not in _CONDITIONER_CLASSES:
            raise ValueError(
                f"Unknown conditioner type {ctype!r} for {name!r}; "
                f"supported: {list(_CONDITIONER_CLASSES)}"
            )
        kwargs.setdefault("output_dim", output_dim)
        kwargs.setdefault("device", device)
        modules[name] = _CONDITIONER_CLASSES[ctype](**kwargs)
    return ConditionProvider(modules, device=device)


def compute_inference_condition_tensors(
    provider: ConditionProvider,
    *,
    initial_reference_text: str = "",
    first_speaker: str = "SPEAKER_MAIN",
) -> ConditionTensors:
    """Build a one-batch :class:`ConditionTensors` for inference.

    Mirrors the MoshiRAG behavior in
    ``moshi-rag/moshi/moshi/inference_utils/utils.py::get_condition_tensors``:
    the ``first_speaker`` LUT slot gets the initial speaker, and the
    ``reference_with_time`` tensor slot starts empty (the runtime ARC encoder
    will swap a real tensor in via
    :meth:`MoshiVisGen.update_streaming_sum_tensor`).
    """
    from kyuteye.conditioners.base import TensorCondition

    text: Dict[str, Optional[str]] = {}
    tensors: Dict[str, TensorCondition] = {}
    device = provider.device

    for name in provider.text_conditions:
        if name == "first_speaker":
            text[name] = first_speaker
        else:
            text[name] = ""
    for name in provider.tensor_conditions:
        cond = provider.conditioners[name]
        dim = cond.output_dim
        # Empty (T=0) tensor; ``learnt_padding`` (if any) will fill in.
        zero_t = torch.zeros(1, 0, dim, device=device)
        zero_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        tensors[name] = TensorCondition(zero_t, zero_mask)

    attrs = [ConditionAttributes(text=text, tensor=tensors)]
    prepared = provider.prepare(attrs)
    return provider(prepared)


def get_moshi_vis(
    kyuteye_config: KyuteyeConfig,
    moshi_weight: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MoshiVisGen, ImageProjection]:
    """Return main Moshi model.

    When ``kyuteye_config.rag.enabled`` is True, also instantiates the
    MoshiRAG-style :class:`ConditionProvider` / :class:`ConditionFuser` so the
    model's state-dict layout matches a combined MoshiVis+RAG fine-tune. The
    conditioner weights are randomly initialized when missing from
    ``moshi_weight``; this is the expected "scaffolding" mode (you need a
    fine-tune that actually trained them for the streaming_sum injection to
    do anything meaningful).
    """
    image_proj_state: Dict[str, torch.Tensor] = {}
    model_state: Dict[str, torch.Tensor] = {}

    if moshi_weight is not None:
        from safetensors.torch import load_file

        for key, v in load_file(moshi_weight, device=device).items():  # type: ignore
            if key.startswith("image_prefix."):
                image_proj_state[key[13:]] = v
            else:
                model_state[key] = v

    rag_cfg = kyuteye_config.rag
    rag_enabled = bool(rag_cfg.enabled)
    moshi_kwargs = dict(kyuteye_config.moshi_constructor_kwargs)
    moshi_kwargs.setdefault("dtype", dtype)

    moshivis = MoshiVis(**moshi_kwargs)

    if rag_enabled:
        logger.info(
            "[RAG] enabled: building ConditionProvider/Fuser with conditioners=%s, "
            "fuse2cond=%s",
            list(rag_cfg.conditioners.keys()),
            rag_cfg.fuse2cond,
        )
        provider = build_condition_provider(
            rag_cfg.conditioners,
            output_dim=moshivis.llm.dim,
            device=device,
        )
        fuse2cond = {str(k): list(v) for k, v in rag_cfg.fuse2cond.items()}
        # Cross-attention conflict check. The image cross-attention path is
        # owned by the vision encoder via ``MoshiVis.forward_text``'s
        # ``cross_attention_src`` parameter. If the fuser's ``cross`` slot is
        # also populated AND vision cross-attention is enabled
        # (``num_crossattended_tokens != 0``), the fuser output would be
        # silently dropped at forward time. We reject only in that explicit
        # collision case -- a "text-only RAG over MoshiVis architecture" use
        # case (no vision, with ``num_crossattended_tokens: 0`` in the YAML)
        # is allowed and the fuser ``cross`` slot routes to
        # ``cross_attention_src`` as MoshiRAG intends. MoshiRAG's own stock
        # config doesn't use ``cross`` for text anyway (it uses
        # ``streaming_sum``), so this collision is rare in practice.
        cross_conditions = fuse2cond.get("cross", [])
        if cross_conditions and bool(kyuteye_config.fuse.num_crossattended_tokens):
            raise ValueError(
                f"rag.fuse2cond routes {cross_conditions!r} to the 'cross' slot, "
                "but vision cross-attention is also enabled "
                f"(num_crossattended_tokens={kyuteye_config.fuse.num_crossattended_tokens}). "
                "Pick one: use the 'streaming_sum' slot for reference text "
                "(MoshiRAG default), or disable vision cross-attention with "
                "``fuse.num_crossattended_tokens: 0`` to repurpose the cross slot "
                "for the fuser output."
            )
        fuser = ConditionFuser(fuse2cond=fuse2cond)
        moshivis.condition_provider = provider
        moshivis.fuser = fuser
        moshivis.rag_token_id = rag_cfg.rag_token_id

    if moshi_weight is not None:
        missing_keys, _ = moshivis.load_state_dict(model_state, strict=False)
        missing_keys = [
            k
            for k in missing_keys
            if ("cross_attention.mha" not in k or "layers.0" in k)
        ]
        # When RAG is enabled but the checkpoint pre-dates the fine-tune,
        # conditioner weights are expected-missing.
        if rag_enabled:
            missing_keys = [
                k
                for k in missing_keys
                if not k.startswith("condition_provider.")
            ]
            if any(k.startswith("condition_provider.") for k in missing_keys):
                logger.warning(
                    "[RAG] condition_provider weights are missing from the checkpoint; "
                    "using random init -- you will need to fine-tune."
                )
        assert len(missing_keys) == 0, missing_keys

    moshivis = moshivis.eval().to(device).to(dtype)

    condition_tensors: Optional[ConditionTensors] = None
    if rag_enabled:
        condition_tensors = compute_inference_condition_tensors(moshivis.condition_provider)  # type: ignore[arg-type]

    moshi_vis = MoshiVisGen(
        moshi_vis=moshivis,
        condition_tensors=condition_tensors,
        force_streaming_sum=rag_enabled and rag_cfg.force_streaming_sum,
        cfg_coef=rag_cfg.cfg_coef if rag_enabled else 1.0,
        **(gen_kwargs or {}),
    )

    image_embedder = ImageProjection.from_config(
        kyuteye_config, moshi_vis.model_dim, image_proj_state, device
    )

    return moshi_vis, image_embedder.to(dtype)
