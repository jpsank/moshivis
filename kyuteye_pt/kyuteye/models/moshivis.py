"""Moshi the little AI"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from kyuteye.conditioners import (
    ConditionFuser,
    ConditionProvider,
    ConditionTensors,
    ConditionType,
)


def _dropped_condition_tensors(condition_tensors: ConditionTensors) -> ConditionTensors:
    """Return a copy of ``condition_tensors`` with all conditions zeroed out.

    Used to build the "null" branch for classifier-free guidance: every
    attribute's condition tensor is replaced with zeros and its mask with
    a zeros mask, which makes the conditioner's ``learnt_padding`` (if any)
    fill in instead. Mirrors what MoshiRAG produces by passing
    :func:`dropout_all_conditions` through the provider.
    """
    return {
        name: ConditionType(
            torch.zeros_like(cond.condition), torch.zeros_like(cond.mask)
        )
        for name, cond in condition_tensors.items()
    }


def _cfg_stack(
    pos: Optional[torch.Tensor], null: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Stack ``[pos; null]`` along the batch dim for CFG. Either side may be ``None``."""
    if pos is None and null is None:
        return None
    if pos is None:
        pos = torch.zeros_like(null)
    if null is None:
        null = torch.zeros_like(pos)
    return torch.cat([pos, null], dim=0)


def _cfg_repeat(
    x: Optional[torch.Tensor | Tuple[torch.Tensor, ...]],
) -> Optional[torch.Tensor | Tuple[torch.Tensor, ...]]:
    """Repeat a tensor (or each tensor in a tuple) along the batch dim for CFG."""
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_cfg_repeat(t) for t in x)  # type: ignore[return-value]
    reps = [2] + [1] * (x.ndim - 1)
    return x.repeat(*reps)
from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.helium import Helium
from kyuteye.modules.streaming_utils import StreamingModule
from kyuteye.modules.transformer import Transformer
from kyuteye.modules.utils import ClampedEmbedding
from moshi.utils.sampling import sample_token


class MoshiVis(StreamingModule):
    """Moshi model derived from Audiocraft with extra stuff for vision conditioninign"""

    # Class attributes; extra special tokens
    end_of_text_padding_id = 0
    zero_token_id = -1
    ungenerated_token_id = -2

    def __init__(
        self,
        hidden_scale: float = 4.125,
        norm: str = "real_rms_norm_f32",
        gating: bool = True,
        activation: str = "silu",
        n_q: int = 8,
        dep_q: Optional[int] = None,
        audio_card: int = 1024,
        audio_context: Optional[int] = None,
        depformer: bool = False,
        depformer_multi_linear: bool = False,
        depformer_pos_emb: Optional[Literal["none", "sin", "rope", "sin_rope"]] = None,
        depformer_dim: Optional[int] = None,
        depformer_dim_feedforward: Optional[int] = None,
        depformer_num_layers: Optional[int] = None,
        depformer_num_heads: Optional[int] = None,
        depformer_weights_per_step: bool = False,
        depformer_context: Optional[int] = 8,
        depformer_gating: Optional[bool] = None,
        depformer_activation: Optional[str] = None,
        delays: Optional[List[int]] = None,
        text_card: int = 32000,
        text_context: Optional[int] = None,
        padding_token_id: int = 3,
        condition_provider: Optional[ConditionProvider] = None,
        fuser: Optional[ConditionFuser] = None,
        rag_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a MoshiVis model.

        :param condition_provider: Optional MoshiRAG-style conditioner registry.
            When provided, the model's state-dict layout matches MoshiRAG so
            a combined fine-tune can be loaded without surgery.
        :param fuser: Optional :class:`ConditionFuser`. Routes named conditions
            into the ``sum`` / ``prepend`` / ``cross`` / ``streaming_sum`` slots
            of :meth:`forward_text`. Must be paired with ``condition_provider``.
        :param rag_token_id: Optional text-vocab id that signals retrieval. When
            the model emits this token, the server's Omni pipeline fires the
            configured retriever. ``None`` (default) disables the token path;
            the substring trigger (``<ret>``) still works.
        """
        super().__init__()
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.rag_token_id = rag_token_id
        if (condition_provider is None) != (fuser is None):
            raise ValueError(
                "condition_provider and fuser must be set together (both or neither)"
            )
        # Set parameter for generation/preprocessing
        self.text_card = text_card
        self.audio_card = audio_card
        self.text_context = text_context
        self.text_padding_token_id = padding_token_id
        self.audio_context = audio_context
        self.n_q = n_q
        self.dep_q = dep_q or self.n_q
        assert delays is not None and len(delays) > 0, "Delays must be non empty"
        assert len(delays) <= self.num_codebooks, "Too many delays"
        if len(delays) < self.num_codebooks:
            delays = delays + [delays[-1]] * (self.num_codebooks - len(delays))
        self.delays = delays

        embeddings_factory = partial(
            ClampedEmbedding, device=device, dtype=dtype, zero_idx=self.zero_token_id
        )

        # LLM backbone (includes text embedding + text linear projection)
        self.llm = Helium(
            hidden_scale=hidden_scale,
            card=text_card
            + 1,  # Add an initial token in the embedding but not in the text linear
            output_card=text_card + int(padding_token_id is None),
            padding_token_id=padding_token_id,
            device=device,
            dtype=dtype,
            zero_token_id=self.zero_token_id,
            **kwargs,
        )

        # Audio input embeddings
        self.audio_emb = torch.nn.ModuleList(
            [
                embeddings_factory(audio_card + 1, self.llm.dim)
                for _ in range(self.num_audio_codebooks_in)
            ]
        )

        # Depformer
        self.depformer: Optional[torch.nn.Module] = None
        self.depformer_multi_linear = depformer_multi_linear
        if depformer:
            assert depformer_dim is not None
            assert depformer_num_heads is not None
            assert depformer_num_layers is not None
            assert depformer_pos_emb is not None
            if depformer_dim_feedforward is None:
                depformer_dim_feedforward = int(hidden_scale * depformer_dim)
            assert depformer_dim_feedforward is not None

            self.depformer_in = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.llm.dim, depformer_dim, bias=False)
                    for _ in range(
                        self.num_audio_codebooks_out if depformer_multi_linear else 1
                    )
                ]
            )
            # Text and audio input embeddings for the depformer
            self.depformer_emb = torch.nn.ModuleList(
                [
                    embeddings_factory(audio_card + 1, depformer_dim)
                    for _ in range(self.num_audio_codebooks_out - 1)
                ]
            )
            self.depformer_text_emb = embeddings_factory(text_card + 1, depformer_dim)

            self.depformer = Transformer(
                d_model=depformer_dim,
                dim_feedforward=depformer_dim_feedforward,
                positional_embedding=depformer_pos_emb,
                num_heads=depformer_num_heads,
                num_layers=depformer_num_layers,
                norm=norm,
                device=device,
                dtype=dtype,
                causal=True,
                cross_attention=False,
                context=depformer_context,
                gating=depformer_gating or gating,
                activation=depformer_activation or activation,
                weights_per_step=dep_q if depformer_weights_per_step else None,
            )
            # Output projection
            self.audio_linears = torch.nn.ModuleList(
                [
                    torch.nn.Linear(depformer_dim, audio_card, bias=False)
                    for _ in range(self.num_audio_codebooks_out)
                ]
            )

    @property
    def cross_attention(self) -> bool:
        """Shortcut for checking whether cross_attention i sused"""
        return self.llm.cross_attention

    @property
    def num_audio_codebooks_in(self) -> int:
        """Number of audio codebooks to model as input"""
        return self.n_q

    @property
    def num_audio_codebooks_out(self) -> int:
        """Number of audio codebooks to model in the depformer"""
        return self.dep_q

    @property
    def num_codebooks(self) -> int:
        """Number codebooks including text"""
        return self.num_audio_codebooks_in + 1

    @property
    def initial_audio_token_id(self) -> int:
        """Initial token for the audio codebooks"""
        return self.audio_card

    @property
    def initial_text_token_id(self) -> int:
        """Initial token for the text; takes into account the "fake/proxy"
        tokens for beginning and end of image if they have been set"""
        return self.text_card

    @property
    def audio_offset(self) -> int:
        """Offset in the audio codebook. Returns 1 because we always generate with text"""
        return 1

    def forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        cross_attention_src: Optional[
            Tuple[torch.Tensor, torch.Tensor] | torch.Tensor
        ] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sum_condition: Optional[torch.Tensor] = None,
        streaming_sum_condition: Optional[torch.Tensor] = None,
        sequence_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass for Moshi

        :param input_ids: Text + audio tokens of shape (batch, codebooks, seq length)
        :param cross_attention_src: Conditioning (image) tokens that can be
            cross-attended to through the cross attention module
        :param cross_attention_mask: Additional mask for the cross_attention_src.
            This is necessary mainly for Pixtral models, as the cross-attended images
            might be of different sizes and therefore padded.
        :param attention_mask: Optional attention mask on input_ids (e.g. used at
            generation for batched inference with left padding)
        :param sum_condition: Optional ``(batch, 1, llm_dim)`` tensor added to
            input embeddings once at every step (MoshiRAG ``sum`` fuser path).
        :param streaming_sum_condition: Optional ``(batch, 1, llm_dim)`` tensor
            added to input embeddings per generation step (MoshiRAG
            ``streaming_sum`` fuser path -- the one used for asynchronous
            retrieval injection). Must be passed only with ``seq_len==1``.
        :param sequence_emb: Optional pre-computed input embeddings of shape
            ``(batch, T, llm_dim)``. When provided, ``input_ids`` is ignored and
            the embeddings are fed straight to the transformer. Used to apply
            the ``prepend`` condition at session start.
        :return: ``(transformer_out, text_logits, gate_weight)``.
        """
        # Embed tokens (or accept caller-provided embeddings for prepend path).
        if sequence_emb is not None:
            assert input_ids is None, "pass either input_ids or sequence_emb, not both"
            inputs_embeds = sequence_emb
        else:
            assert input_ids is not None, "input_ids is required when sequence_emb is None"
            inputs_embeds = torch.zeros((), device=input_ids.device)
            if self.audio_offset > 0:
                inputs_embeds = self.llm.text_emb(input_ids[:, 0, :])
            for cb_index in range(self.num_audio_codebooks_in):
                update = self.audio_emb[cb_index](
                    input_ids[:, cb_index + self.audio_offset, :]
                )
                inputs_embeds += update

        if sum_condition is not None:
            inputs_embeds = inputs_embeds + sum_condition.to(inputs_embeds)

        if streaming_sum_condition is not None:
            assert (
                streaming_sum_condition.shape[1] == inputs_embeds.shape[1] == 1
            ), "streaming_sum_condition is only supported in streaming (seq_len=1) mode"
            inputs_embeds = inputs_embeds + streaming_sum_condition.to(inputs_embeds)

        # Pass through Helium
        transformer_out, gate_weight = self.llm(
            inputs_embeds=inputs_embeds,
            cross_attention_src=cross_attention_src,
            cross_attention_mask=cross_attention_mask,
            attention_mask=attention_mask,
            return_features=True,
        )

        # Output proj
        text_logits = self.llm.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits, gate_weight

    def forward_depformer(
        self,
        depformer_cb_index: int,
        input_ids: torch.Tensor,
        depformer_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one depformer step"""
        _, num_codes, seq_len = input_ids.shape
        assert self.depformer is not None
        assert (
            num_codes == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {num_codes}."
        assert (
            seq_len == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {seq_len}."
        assert (
            depformer_input.shape[1] == 1
        ), "Transformer output should be a for a single step."

        # project transformer out
        depformer_input = self.depformer_in[
            depformer_cb_index if self.depformer_multi_linear else 0
        ](depformer_input)

        # project input ids
        if depformer_cb_index == 0:
            depformer_input += self.depformer_text_emb(input_ids[:, 0])
        else:
            depformer_input += self.depformer_emb[depformer_cb_index - 1](
                input_ids[:, 0]
            )

        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output, _ = self.depformer(depformer_input)
        logits = self.audio_linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    @property
    def device(self) -> torch.device:
        """Torch device"""
        return next(iter(self.parameters())).device

    def get_initial_token(self) -> torch.Tensor:
        """Returns the initial token that will be fed to the model to predict the
        very first timestep. This is akin to a beginning of sentence tokens but
        to handle potentially delayed codebooks

        :param text_or_audio: Whether we are predicting for text, audio, or both

        :return: A Tensor fo shape (B, K, 1)
        """
        zero = torch.full(
            [1, 1, 1], MoshiVis.zero_token_id, device=self.device, dtype=torch.long
        )
        audio_token = torch.full_like(
            zero, self.initial_audio_token_id or MoshiVis.zero_token_id
        )
        text_token = torch.full_like(
            zero, self.initial_text_token_id or MoshiVis.zero_token_id
        )

        return torch.cat(
            [text_token, audio_token.expand(-1, self.num_audio_codebooks_in, -1)], dim=1
        )


class MoshiVisGen(StreamingModule):
    """MoshiVis for autoregressive generation at inference"""

    def __init__(
        self,
        moshi_vis: MoshiVis,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
        condition_tensors: Optional[ConditionTensors] = None,
        force_streaming_sum: bool = False,
        cfg_coef: float = 1.0,
    ):
        """Initialize a streaming-inference wrapper.

        :param cfg_coef: Classifier-free-guidance coefficient. When ``!= 1.0``,
            the model runs internally at double batch on every step (positive
            and null-conditioned branches) and the text + depformer logits
            are interpolated as ``null + (pos - null) * cfg_coef`` before
            sampling. Requires conditioner-dropout training to be useful at
            inference. Memory cost: 2x KV cache. Compute cost: 2x forward.
        """
        assert not moshi_vis.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = moshi_vis
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(
            moshi_vis.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            moshi_vis.delays, device=self.lm_model.device, dtype=torch.long
        )
        self.initial_token = self.lm_model.get_initial_token()
        self.condition_tensors = condition_tensors
        self.force_streaming_sum = force_streaming_sum
        self.cfg_coef = cfg_coef
        self._cfg = cfg_coef != 1.0

        # Pre-compute the static (per-session) condition slots from
        # ``condition_tensors``. MoshiRAG's ``LMGen._init_streaming_state``
        # does the equivalent inside the per-batch state initializer because
        # it owns CFG and per-slot reset; MoshiVis is single-batch / no-CFG so
        # the slots are stable for the lifetime of a ``MoshiVisGen`` instance.
        # Implication: if you ever move the model to a different device or
        # change dtype between sessions, rebuild the ``MoshiVisGen`` rather
        # than reuse the existing one (the static slots are pinned to the
        # dtype/device they were initialized with).
        self._condition_sum: Optional[torch.Tensor] = None
        self._condition_cross: Optional[torch.Tensor] = None
        self._condition_prepend: Optional[torch.Tensor] = None
        self._condition_streaming_sum_init: Optional[torch.Tensor] = None
        if condition_tensors is not None and self.lm_model.fuser is not None:
            fuser = self.lm_model.fuser
            self._condition_sum = fuser.get_sum(condition_tensors)
            self._condition_cross = fuser.get_cross(condition_tensors)
            self._condition_prepend = fuser.get_prepend(condition_tensors)
            self._condition_streaming_sum_init = fuser.get_streaming_sum(condition_tensors)
            # For CFG, stack the positive condition tensors with their null
            # (all-attributes-dropped) counterpart along the batch dim. The
            # streaming cache and per-step inputs are doubled to match; see
            # :meth:`step`. Mirrors MoshiRAG's ``LMGen._init_streaming_state``
            # when ``cfg_coef != 1.0``.
            if self._cfg:
                null_tensors = _dropped_condition_tensors(condition_tensors)
                for name, fuse_method in (
                    ("_condition_sum", "sum"),
                    ("_condition_cross", "cross"),
                    ("_condition_prepend", "prepend"),
                    ("_condition_streaming_sum_init", "streaming_sum"),
                ):
                    pos = getattr(self, name)
                    null = getattr(fuser, f"get_{fuse_method}")(null_tensors)
                    setattr(self, name, _cfg_stack(pos, null))
            target_dtype = self.lm_model.llm.text_emb.weight.dtype
            for name in (
                "_condition_sum",
                "_condition_cross",
                "_condition_prepend",
                "_condition_streaming_sum_init",
            ):
                t = getattr(self, name)
                if t is not None:
                    setattr(self, name, t.to(dtype=target_dtype))
        if self.force_streaming_sum and self._condition_streaming_sum_init is None:
            # Allocate a zero ``[B, 1, dim]`` slot (B=2 under CFG) so the
            # streaming_sum path is always exercised -- matches MoshiRAG's
            # ``force_streaming_sum``.
            B = 2 if self._cfg else 1
            self._condition_streaming_sum_init = torch.zeros(
                B,
                1,
                self.lm_model.llm.dim,
                device=self.lm_model.device,
                dtype=self.lm_model.llm.text_emb.weight.dtype,
            )

    def update_gen_kwargs(
        self,
        temp: Optional[float] = None,
        temp_text: Optional[float] = None,
        top_k: Optional[int] = None,
        top_k_text: Optional[int] = None,
    ) -> None:
        """update params for sampling during generation"""
        self.temp = temp or self.temp
        self.temp_text = temp_text or self.temp_text
        self.top_k = top_k or self.top_k
        self.top_k_text = top_k_text or self.top_k_text

    @property
    def model_dim(self) -> int:
        """Return dimension of the tokens in the model"""
        return self.lm_model.llm.dim

    @property
    def num_audio_codebooks_out(self) -> int:
        """Number of audio codebooks generated by the model"""
        return self.lm_model.num_audio_codebooks_out

    @classmethod
    def from_config(
        cls,
        kyuteye_config: KyuteyeConfig,
        moshi_weight: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        condition_tensors: Optional[ConditionTensors] = None,
        **gen_kwargs: Any,
    ) -> "MoshiVisGen":
        """Instantiate model from a config.

        :param condition_tensors: Optional pre-computed MoshiRAG-style condition
            tensors (one entry per conditioner). When ``None``, the conditioning
            paths stay inactive. See :func:`kyuteye.models.loaders.get_moshi_vis`
            for how these are produced from a fine-tune's conditioner registry.
        """
        moshivis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
        if moshi_weight is not None:
            missing_keys, _ = moshivis.load_state_dict(moshi_weight, strict=False)
            # Cross-attention MHSA is shared across layers (only layers.0 holds it).
            # Conditioner / fuser weights only exist in MoshiRAG-finetuned checkpoints,
            # so they are also expected-missing when loading a vanilla MoshiVis ckpt.
            missing_keys = [
                k
                for k in missing_keys
                if ("cross_attention.mha" not in k or "layers.0" in k)
                and not k.startswith("condition_provider.")
            ]
            assert len(missing_keys) == 0, missing_keys

        return MoshiVisGen(
            moshi_vis=moshivis.eval().to(device),
            condition_tensors=condition_tensors,
            **gen_kwargs,
        )

    @torch.no_grad()
    def precompte_ca_kv(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompte kv proj for cross-attention"""
        ca_layer = self.lm_model.llm.transformer.layers[0].cross_attention.mha
        if hasattr(ca_layer, "in_proj_weight_kv"):
            splits = torch.chunk(ca_layer.in_proj_weight_kv, 2)
        else:
            splits = torch.chunk(ca_layer.in_proj_weight, 3)

        k = torch.nn.functional.linear(  # pylint: disable=not-callable
            embeddings, splits[-2]
        )
        v = torch.nn.functional.linear(  # pylint: disable=not-callable
            embeddings, splits[-1]  # type: ignore
        )
        return k, v

    def update_streaming_sum_tensor(
        self,
        tensor: Optional[torch.Tensor],
        slot_idx: int = 0,
    ) -> None:
        """Set the pending streaming-sum queue for one batch slot.

        Equivalent to MoshiRAG's ``LMGen.update_streaming_sum_tensors`` but
        signed for explicit per-slot updates (vs. the list-of-tensors batch
        API). Call this when a new reference becomes available (e.g. once
        the ARC encoder responds for an Omni RAG retrieval).

        :param tensor: Either ``None`` (clear the queue), a ``[T, dim]`` tensor,
            or a ``[1, T, dim]`` tensor (leading batch dim is squeezed). ``T``
            is the number of streaming-sum rows; one row is consumed per
            :meth:`step` call via :meth:`apply_pending_streaming_sum_condition`.
        :param slot_idx: Which batch slot to update. Defaults to 0 (the only
            slot for single-stream inference). For multi-batch deployments,
            pass the per-channel slot index.
        """
        pending_dict = self.get_streaming_attribute(
            "pending_streaming_sum_per_slot", {}
        )
        if tensor is None:
            pending_dict.pop(slot_idx, None)
        else:
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            assert tensor.dim() == 2, f"expected [T, dim] tensor, got {tuple(tensor.shape)}"
            assert tensor.shape[-1] == self.model_dim, (
                f"streaming_sum dim {tensor.shape[-1]} != model dim {self.model_dim}"
            )
            tensor = tensor.to(
                device=self.lm_model.device,
                dtype=self.lm_model.llm.text_emb.weight.dtype,
            )
            pending_dict[slot_idx] = tensor
        self.add_streaming_attribute("pending_streaming_sum_per_slot", pending_dict)

    def apply_pending_streaming_sum_condition(
        self, batch_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Consume one row of each per-slot queue into the active condition slot.

        Returns a ``[batch_size, 1, dim]`` tensor (or ``[2 * batch_size, 1, dim]``
        under CFG) that should be passed to :meth:`MoshiVis.forward_text` as
        ``streaming_sum_condition`` for the upcoming step, or ``None`` if no
        slot has a pending queue AND the static condition slot is unset.

        Equivalent to MoshiRAG's ``LMGen.apply_pending_streaming_sum_condition``.
        MoshiRAG mutates ``state.condition_streaming_sum[b, 0]`` in place
        (writing either the next pending row or zeros via ``.zero_()`` when
        a slot's queue is empty); we build a fresh tensor each call. When a
        slot's queue drains and ``force_streaming_sum=True``, we fill that
        slot with the static zero tensor, matching MoshiRAG's ``zero_()``.
        When ``force_streaming_sum=False`` and no slot has a queue, we
        return ``None``, matching MoshiRAG's absence of a slot in that case.

        Always called from :meth:`step`; the server does not need to call it
        directly.
        """
        pending_dict: dict[int, torch.Tensor] = self.get_streaming_attribute(
            "pending_streaming_sum_per_slot", {}
        )
        active = self._condition_streaming_sum_init  # [B_internal, 1, dim] or None
        exec_mask = self.get_streaming_attribute("exec_mask", None)

        # Fast path: no queues anywhere, return the static slot.
        if not pending_dict:
            return active

        # Build the per-slot tensor for this step. Under CFG, internal batch is
        # 2 * user batch with pos in [:B] and null in [B:]; we only need to fill
        # the positive half (null half stays at the static zero).
        dim = self.model_dim
        device = self.lm_model.device
        dtype = self.lm_model.llm.text_emb.weight.dtype
        internal_batch = batch_size * (2 if self._cfg else 1)
        if active is not None:
            out = active.clone() if active.shape[0] == internal_batch else active.expand(
                internal_batch, -1, -1
            ).clone()
        else:
            out = torch.zeros(internal_batch, 1, dim, device=device, dtype=dtype)

        # Pop one row per *active* slot that has a queue; update the dict in
        # place. Idle slots keep their queue intact -- silent users shouldn't
        # consume reference context they haven't actually heard.
        new_dict = dict(pending_dict)
        for slot_idx, pending in pending_dict.items():
            if slot_idx >= batch_size:
                # Stale slot from a since-released session; drop it.
                new_dict.pop(slot_idx, None)
                continue
            if pending.shape[0] == 0:
                new_dict.pop(slot_idx, None)
                continue
            if exec_mask is not None and not bool(exec_mask[slot_idx].item()):
                # Slot is idle this step: surface the next row to the forward
                # pass (so the model sees the right offset if it does run for
                # this slot) but do NOT pop it from the queue.
                out[slot_idx, 0] = pending[0]
                continue
            out[slot_idx, 0] = pending[0]
            if pending.shape[0] > 1:
                new_dict[slot_idx] = pending[1:]
            else:
                new_dict.pop(slot_idx, None)
        self.add_streaming_attribute("pending_streaming_sum_per_slot", new_dict)
        return out

    def prime(self) -> None:
        """Apply the (static) prepend condition once at session start.

        MoshiRAG runs this from its ``_reset_callback``. We expose it as an
        explicit method so the server can call it after
        :meth:`reset_streaming`. No-op when there is no prepend condition.
        """
        if self._condition_prepend is None or self._condition_prepend.shape[1] == 0:
            return
        with torch.no_grad():
            self.lm_model.forward_text(sequence_emb=self._condition_prepend)

    @torch.no_grad()
    def step(
        self,
        input_tokens: torch.Tensor,
        ca_src: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
    ) -> Tuple[torch.Tensor | None, float]:
        """One step of generation.

        Automatically pulls per-step conditioning from the configured slots:

        * ``sum_condition`` from the static ``_condition_sum``
        * ``streaming_sum_condition`` from
          :meth:`apply_pending_streaming_sum_condition` (queue or static zeros)

        ``ca_src`` is the explicit cross-attention input -- for MoshiVis this is
        the image KV (and optionally Omni text KV concatenated on top); we
        keep it as a parameter rather than routing through ``_condition_cross``
        because the vision path predates the conditioner machinery.

        Under CFG (``cfg_coef != 1.0``), the user passes single-batch tokens and
        single-batch ``ca_src``; the model runs internally at double batch and
        the returned tokens are still single-batch (sampled from interpolated
        logits). The caller is responsible for ensuring the model's transformer
        streaming state was allocated for the doubled batch (i.e. invoking
        ``moshi_vis.streaming_forever(2)`` instead of ``(1)`` when CFG is on).
        """
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        user_batch_size, num_codes, seq_len = input_tokens.shape
        assert seq_len == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.num_audio_codebooks_out - 1
        assert (
            num_codes == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {num_codes}."

        # CFG: double the input + ca_src so the model sees pos+null in one
        # forward pass. The user-facing batch dim stays at ``user_batch_size``.
        if self._cfg:
            input_tokens = _cfg_repeat(input_tokens)  # type: ignore[assignment]
            ca_src = _cfg_repeat(ca_src)  # type: ignore[assignment]
        batch_size = input_tokens.shape[0]

        current_input_cache = self.get_streaming_attribute(
            "cache",
            torch.full(
                (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
                self.lm_model.ungenerated_token_id,
                device=self.lm_model.device,
                dtype=torch.long,
            ),
        )
        current_offset = self.get_streaming_attribute("offset", 0)
        dcache_len = current_input_cache.shape[2]

        # Multi-batch exec mask: when present, idle slots keep their existing
        # cache values at each write position. The ``current_offset`` itself
        # is a shared logical step counter -- all slots advance together --
        # but the per-slot cache rows only mutate for active slots. The cross-
        # attention KV cache in :class:`MultiheadAttention` honors the same
        # mask via its own ``set_exec_mask`` path; together this gives proper
        # per-slot streaming with no cache corruption for idle slots.
        exec_mask = self.get_streaming_attribute("exec_mask", None)
        if exec_mask is not None and self._cfg:
            # Under CFG the internal batch is 2x; mirror the exec_mask for
            # both pos+null halves so they stay in sync.
            exec_mask = exec_mask.repeat(2)
        if exec_mask is not None and exec_mask.shape != (batch_size,):
            raise ValueError(
                f"exec_mask shape {tuple(exec_mask.shape)} must be ({batch_size},)"
            )

        def _masked_write(dest_slice: torch.Tensor, new_value: torch.Tensor) -> None:
            """In-place cache write that respects exec_mask if set."""
            if exec_mask is None:
                dest_slice.copy_(new_value)
                return
            # exec_mask: [B] bool. dest_slice: [B, *] of any shape.
            keep = exec_mask.view(-1, *([1] * (dest_slice.ndim - 1)))
            dest_slice.copy_(torch.where(keep, new_value, dest_slice))

        # write input_tokens (sent from Mimi) in OTHER codebooks
        for q_other in range(input_tokens.shape[1]):
            k = lm_model.num_audio_codebooks_out + lm_model.audio_offset + q_other
            write_position = (current_offset + lm_model.delays[k]) % dcache_len
            _masked_write(
                current_input_cache[:, k, write_position : write_position + 1],
                input_tokens[:, q_other],
            )

        # Only for the very beginning, we extend the initial token for the acoustic
        # token that are delayed, and thus have no good value to take.
        position = current_offset % dcache_len
        for k, delay in enumerate(lm_model.delays):
            if current_offset <= delay:
                # Initial-token broadcast happens once at session start -- it's
                # safe to always apply (the exec_mask would skip it for idle
                # slots, but those slots have nothing meaningful in their cache
                # at this point anyway, so the broadcast is harmless).
                current_input_cache[:, k, position] = self.initial_token[:, k, 0]

        # Transformer forward
        input_ = current_input_cache[:, :, position : position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                current_offset,
                input_,
            )
            assert (
                input_[:, lm_model.audio_offset :] <= lm_model.audio_card
            ).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        streaming_sum = self.apply_pending_streaming_sum_condition(
            batch_size=user_batch_size
        )
        transformer_out, text_logits, gate_weight = self.lm_model.forward_text(
            input_,
            cross_attention_src=ca_src,
            sum_condition=self._condition_sum,
            streaming_sum_condition=streaming_sum,
        )

        # CFG: split the doubled-batch text logits into positive + null halves
        # and interpolate before sampling. ``transformer_out`` is kept doubled
        # because :meth:`depformer_step` does its own CFG handling.
        if self._cfg:
            pos_logits, null_logits = text_logits.chunk(2, dim=0)
            text_logits = null_logits + (pos_logits - null_logits) * self.cfg_coef

        # Sample text tokens
        # Shape of text_logits should be [user_batch_size, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [user_batch_size]

        # Generate and sample audio tokens
        audio_tokens = self.depformer_step(text_token, transformer_out)

        # Write generated tokens
        current_offset += 1
        position = current_offset % dcache_len
        # Broadcast user-batch tokens to internal-batch cache (under CFG the
        # cache has B=2 user-batch slots; the same generated token goes to
        # both pos+null branches since the cache holds the actual generation).
        if self._cfg:
            text_token_for_cache = text_token.repeat(2)
            audio_tokens_for_cache = audio_tokens.repeat(2, 1)
        else:
            text_token_for_cache = text_token
            audio_tokens_for_cache = audio_tokens
        _masked_write(
            current_input_cache[:, 0, position],
            text_token_for_cache,
        )
        _masked_write(
            current_input_cache[
                :,
                lm_model.audio_offset : lm_model.num_audio_codebooks_out
                + lm_model.audio_offset,
                position,
            ],
            audio_tokens_for_cache,
        )

        # if <= max_delay, we continue partial-generation
        # until removing all ungenerated tokens
        if current_offset <= self.max_delay:
            self.add_streaming_attribute("cache", current_input_cache)
            self.add_streaming_attribute("offset", current_offset)
            return None, 0.0

        # otherwise, retrieve tokens with the correct delay
        gen_delays_cuda = self.delays_cuda[
            : lm_model.num_audio_codebooks_out + lm_model.audio_offset
        ]
        index = (
            ((current_offset - self.max_delay + gen_delays_cuda) % dcache_len)
            .view(1, -1, 1)
            .expand(current_input_cache.shape[0], -1, 1)
        )
        out = current_input_cache.gather(dim=2, index=index)
        self.add_streaming_attribute("offset", current_offset)
        self.add_streaming_attribute("cache", current_input_cache)
        # Under CFG, the cache holds both pos and null branches; the caller
        # only cares about the positive branch's generated tokens.
        if self._cfg:
            out = out[:user_batch_size]
        return out, gate_weight

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """A step of the depformer.

        Under CFG, ``text_token`` is single-batch (already sampled from the
        interpolated main-LM logits) but ``transformer_out`` is double-batch
        (pos+null halves) because the depformer's own KV cache was allocated
        for the doubled batch. We repeat ``text_token`` to feed both halves
        through ``forward_depformer``, then interpolate each codebook's logits
        before sampling -- mirrors MoshiRAG's depformer_step at
        ``moshi-rag/moshi/moshi/models/lm.py:906``.
        """
        user_batch_size = text_token.shape[0]
        depformer_tokens: list[torch.Tensor] = []
        assert self.lm_model.depformer is not None

        with self.lm_model.depformer.streaming():
            next_token = text_token[:, None, None]

            for cb_index in range(self.lm_model.num_audio_codebooks_out):
                input_ = next_token
                if self._cfg:
                    input_ = input_.repeat(2, 1, 1)
                logits = self.lm_model.forward_depformer(
                    cb_index, input_, transformer_out
                )
                if self._cfg:
                    pos_logits, null_logits = logits.chunk(2, dim=0)
                    logits = null_logits + (pos_logits - null_logits) * self.cfg_coef
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (user_batch_size, 1, 1)
                depformer_tokens.append(next_token[:, 0, 0])
        out = torch.stack(depformer_tokens, dim=1)
        return out
