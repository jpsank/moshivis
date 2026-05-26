# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MultiHead Self-Attention module with optional KV Caching"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange
from kyuteye.modules.streaming_utils import StreamingModule
from kyuteye.modules.utils import RotaryEmbedding, multi_linear


@dataclass
class KVCache:
    """Efficient streaming KVCache to avoid allocating new memory too many times.

    :param batch_size: Batch size.
    :param num_heads: Number of heads in the attention.
    :param dim_per_head: Dimension per head.
    :param context: Context size for the attention, if None, will grow exponentially,
        otherwise will use a fixed allocation with a bit of overhead.
    :param growth: Growth factor for the exponential growth, fraction of overhead
        when context is not None.
    :param initial_size: Initial size of the cache, used only when context is None.
    :param device: Device on which to initialize the cache.
    :param dtype: dtype to use for the cache.
    :param cache: Initial cache, if provided.
    :param current_end: Current end of the cache, used only when cache is provided.
        Can be a Python int (single-stream) or a ``[batch_size]`` long tensor
        (multi-batch with per-slot end_offset, matching moshi 0.2.13's
        ``KVCache.end_offset`` semantics).
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        context: Optional[int] = None,
        growth: float = 1.2,
        initial_size: int = 100,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        cache: Optional[torch.Tensor] = None,
        current_end: int | torch.Tensor = 0,
    ) -> None:
        if cache is None:
            assert isinstance(current_end, int) and current_end == 0

        assert growth > 1
        self.growth = growth

        if context is not None:
            initial_size = 1 + int(growth * context)

        self.capacity = initial_size
        self.context = context
        self.batch_size = batch_size
        # Per-slot end-offset tensor. MoshiRAG (and upstream moshi 0.2.13)
        # tracks one end_offset per batch slot so idle slots can be skipped
        # via exec_mask without their cache pointer advancing. ``current_end``
        # remains as a ``@property`` for backwards-compat callers that read
        # the scalar max; writes go through the tensor.
        if isinstance(current_end, torch.Tensor):
            self._end_offset = current_end.to(device=device, dtype=torch.long)
        else:
            self._end_offset = torch.full(
                (batch_size,), current_end, device=device, dtype=torch.long
            )

        if cache is None:
            self._cache = torch.full(
                (2, batch_size, initial_size, num_heads, dim_per_head),
                float("NaN"),
                device=device,
                dtype=dtype,
            )
        else:
            self._cache = cache

    @property
    def current_end(self) -> int:
        """Backwards-compat scalar -- returns the max end_offset across slots."""
        return int(self._end_offset.max().item())

    @property
    def end_offset(self) -> torch.Tensor:
        """Per-slot end-offset tensor ``[batch_size]``."""
        return self._end_offset

    def clone(self) -> "KVCache":
        """Return a separate memory copy of the KV cache"""
        return KVCache(
            self._cache.shape[1],
            self._cache.shape[3],
            self._cache.shape[4],
            self.context,
            self.growth,
            self.capacity,
            self._cache.device,
            self._cache.dtype,
            self._cache.clone(),
            self._end_offset.clone(),
        )

    @property
    def current_start(self) -> int:
        """Current start of the KV cache (0 if no context size).

        Returns the floor of the per-slot starts; the per-slot active window
        for slot ``i`` is ``(end_offset[i] - context, end_offset[i])``.
        """
        if self.context is None:
            return 0
        max_end = self.current_end
        return max(max_end - self.context, 0)

    def reset(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        """Zero ``end_offset`` for the masked slots (or all slots if ``None``).

        Matches moshi 0.2.13 ``KVCache.reset`` semantics so per-slot reset
        from a parent ``MoshiVisGen.reset_streaming(reset_mask=...)`` call
        actually clears each slot's history. The cache buffer is left intact;
        future writes will overwrite the relevant positions.
        """
        if reset_mask is None:
            self._end_offset.zero_()
            return
        reset_mask = reset_mask.to(self._end_offset.device)
        self._end_offset[:] = torch.where(
            reset_mask,
            torch.zeros_like(self._end_offset),
            self._end_offset,
        )

    def __maybe_increase_capacity__(self, required_capacity: int) -> None:
        """If needed, increase capacity to the `required_capacity`
        using exponential growth strategy"""
        if required_capacity > self.capacity:
            if self.context is None:
                # We take an exponential growth approach.
                new_capacity = self.capacity
                while required_capacity > new_capacity:
                    new_capacity = int(math.ceil(new_capacity * self.growth))
                new_shape = list(self._cache.shape)
                new_shape[2] = new_capacity
                new_cache = torch.full(
                    tuple(new_shape),
                    float("NaN"),
                    device=self._cache.device,
                    dtype=self._cache.dtype,
                )
                # Copy valid prefix per slot. Slots advance at different rates
                # under exec_mask, so the safe copy covers the global max end.
                end = self.current_end
                new_cache[:, :, :end] = self._cache[:, :, :end]
                self._cache = new_cache
                self.capacity = new_capacity
            else:
                # With context, we roll the cache to the left.
                start = self.current_start
                assert start > 0
                self._cache[:] = self._cache.roll(-start, dims=2)
                self._end_offset.sub_(start)

    def complete(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        exec_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add keys/values to the cache at each slot's own ``end_offset``.

        Mirrors moshi 0.2.13's ``KVCache.complete`` (transformer.py:236) in
        spirit: writes use ``torch.scatter_`` with per-slot indices, idle
        slots (``exec_mask=False``) keep their existing values, and the
        per-slot ``end_offset`` advances only for active slots.

        Returns ``(keys, values)`` slices spanning ``[current_start,
        max(end_offset)]`` -- positions past a slot's own ``end_offset`` may
        contain stale values from prior writes. Callers that need exact
        per-slot masking should consult :attr:`end_offset` and build an
        attention mask. For MoshiVis streaming at T=1 the staleness only
        affects idle slots, whose outputs the server discards.

        :param exec_mask: Optional ``[batch_size]`` bool. ``True`` slots
            advance their ``end_offset`` and write new K/V; ``False`` slots
            keep their cache values at the write position and do not advance.
            ``None`` (default) advances all slots.
        """
        assert k.shape[1] == v.shape[1]
        B, T = k.shape[0], k.shape[1]
        if exec_mask is None:
            exec_mask = torch.ones(B, dtype=torch.bool, device=self._end_offset.device)

        # Grow capacity to the global max end.
        max_end = int((self._end_offset + T).max().item())
        self.__maybe_increase_capacity__(max_end)

        # Per-slot write positions ``[B, T]`` mod capacity (matches upstream's
        # ring-buffer semantics; for the exponential-growth path the modulo
        # is a no-op because positions never exceed capacity here).
        arange_t = torch.arange(T, device=self._end_offset.device, dtype=torch.long)
        write_positions = (self._end_offset.view(-1, 1) + arange_t.view(1, -1)) % self.capacity
        # Expand to ``[B, T, num_heads, dim_per_head]`` for scatter on dim 2.
        num_heads, dim_per_head = self._cache.shape[3], self._cache.shape[4]
        scatter_index = (
            write_positions.view(B, T, 1, 1)
            .expand(B, T, num_heads, dim_per_head)
        )
        # ``torch.where`` to skip idle slots: read old values at write
        # positions, gate on exec_mask, scatter back.
        keep = exec_mask.view(B, 1, 1, 1)
        old_k = self._cache[0].gather(1, scatter_index)
        old_v = self._cache[1].gather(1, scatter_index)
        new_k = torch.where(keep, k, old_k)
        new_v = torch.where(keep, v, old_v)
        self._cache[0].scatter_(1, scatter_index, new_k)
        self._cache[1].scatter_(1, scatter_index, new_v)

        # Advance end_offset per-slot.
        self._end_offset[:] = torch.where(
            exec_mask, self._end_offset + T, self._end_offset
        )

        # Return the slice up to the global max end_offset. Slots with
        # smaller end_offset have stale tail values; we accept that since
        # the attention output for those slots is discarded by the caller.
        end = int(self._end_offset.max().item())
        start = self.current_start
        valid = self._cache[:, :, start:end]
        return valid[0], valid[1]


class MultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for causal evaluation.

    Args:
        :param embed_dim: Dimension to project to.
        :param num_heads: Number of heads.
        :param causal: If true, applies causal mask automatically.
        :param context: Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        :param rope: Rope embedding to use. If None, no rope embedding is applied
        :param cross_attention: Should be true when used as a cross attention.
            Cannot be used with `causal` or `rope` (as it wouldn't make sens to
            interpret the time steps in the keys relative to those in the queries).
        :param use_kv_cache: If True, enables a KV cache with context size `context`.
        :param weights_per_step: use different weights per depformer step. If non zero,
            should correspond to the number of possible time steps.
        :param device: Device on which to initialize the module.
        :param dtype: dtype to use.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        cross_attention: bool = False,
        use_kv_cache: bool = False,
        weights_per_step: int = 0,
        xa_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.cross_attention = cross_attention
        self.num_heads = num_heads
        self.use_kv_cache = use_kv_cache
        self.weights_per_step = weights_per_step
        mult = max(1, weights_per_step)

        if cross_attention:
            assert not causal, "Cannot set causal mask when `cross attention` is True."
            assert (
                not context
            ), "Cannot set context size when `cross attention` is True."

        # if cross-attention source have != num_dims than the speech tokens,
        # we need to separate the KV and Q embeddings
        if cross_attention and xa_dim is not None and xa_dim != embed_dim:
            in_proj_q = torch.nn.Linear(
                embed_dim, mult * embed_dim, bias=False, **factory_kwargs
            )
            in_proj_kv = torch.nn.Linear(
                xa_dim, mult * 2 * embed_dim, bias=False, **factory_kwargs
            )
            self.in_proj_weight_q = in_proj_q.weight
            self.in_proj_bias_q = in_proj_q.bias
            self.in_proj_weight_kv = in_proj_kv.weight
            self.in_proj_bias_kv = in_proj_kv.bias
            self.in_proj_weight = None
            self.in_proj_bias = None
        else:
            in_proj = torch.nn.Linear(
                embed_dim, mult * 3 * embed_dim, bias=False, **factory_kwargs
            )
            self.in_proj_weight = in_proj.weight
            self.in_proj_bias = in_proj.bias
        self.out_proj = torch.nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def _complete_kv(
        self, k: torch.Tensor, v: torch.Tensor, initial_kv_cache_size: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add key/values to the KV cache.

        When an ``exec_mask`` has been set on this module's streaming state
        (via :meth:`StreamingModule.set_exec_mask`), slots marked ``False``
        keep their prior cache contents at the current write position --
        the cache is advanced per-slot via ``end_offset``, and idle slots'
        K/V are preserved. ``streaming_offset`` is also a per-slot tensor
        ``[batch_size]`` (mirrors upstream moshi 0.2.13) so RoPE positions
        and token-position counters stay in sync per-slot.
        """
        # With cross attention we assume all keys and values
        # are already available, and streaming is with respect
        # to the queries only.
        if self._is_streaming and not self.cross_attention:
            B = k.shape[0]
            if "kv_cache" not in self._streaming_state:
                self._streaming_state["kv_cache"] = KVCache(  # type: ignore
                    B,
                    k.shape[2],
                    k.shape[3],
                    self.context,
                    initial_size=self.weights_per_step or initial_kv_cache_size,
                    device=k.device,
                    dtype=k.dtype,
                )
                self.streaming_offset = torch.zeros(B, device=k.device, dtype=torch.long)  # type: ignore
            kv_cache: KVCache = self._streaming_state["kv_cache"]  # type: ignore
            exec_mask = self.get_streaming_attribute("exec_mask", None)
            # Per-slot streaming_offset advance, gated by exec_mask. Matches
            # the per-slot semantics of MoshiRAG's offsets tensor.
            offset = self._streaming_state.get("offset")
            if exec_mask is not None and isinstance(offset, torch.Tensor):
                self._streaming_state["offset"] = torch.where(
                    exec_mask.to(offset.device),
                    offset + k.shape[1],
                    offset,
                )
            elif isinstance(offset, torch.Tensor):
                self._streaming_state["offset"] = offset + k.shape[1]
            return kv_cache.complete(k, v, exec_mask=exec_mask)

        return k, v

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """If self.cross attention is False, we only expects the first input. Otherwise,
        when using cross attention, we need to explicitly give the source for the
        respective query/key/value embeddings"""
        # Get current streaming offset before it gets potentially modified by the KV cache update
        current_streaming_offset = self.streaming_offset

        if self.cross_attention:
            assert key is not None, "Missing inputs in cross attention"
            if isinstance(key, torch.Tensor):
                value = value or key
                assert value is not None, "Missing inputs in cross attention"
            # Case 1: Inputs x and ca_src have the same number of dimension
            # We have a single big weight for the QKV projections
            if self.in_proj_weight is not None:
                q = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight[: self.embed_dim]
                )
                if isinstance(key, torch.Tensor):
                    k = torch.nn.functional.linear(  # pylint: disable=not-callable
                        key, self.in_proj_weight[self.embed_dim : 2 * self.embed_dim]
                    )
                    v = torch.nn.functional.linear(  # pylint: disable=not-callable
                        value, self.in_proj_weight[2 * self.embed_dim :]  # type: ignore
                    )
                else:
                    k, v = key
            # Case 2: Inputs x and ca_src have different number of dimension
            # We have to separate the Q and KV proj
            else:
                q = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight_q[: self.embed_dim]
                )
                if isinstance(key, torch.Tensor):
                    k = torch.nn.functional.linear(  # pylint: disable=not-callable
                        key, self.in_proj_weight_kv[: self.embed_dim]
                    )
                    v = torch.nn.functional.linear(  # pylint: disable=not-callable
                        value, self.in_proj_weight_kv[self.embed_dim :]  # type: ignore
                    )
                else:
                    k, v = key
            q, k, v = [
                rearrange(x, "b t (h d) -> b t h d", h=self.num_heads)
                for x in [q, k, v]
            ]
        else:
            assert self.in_proj_weight is not None
            if self.weights_per_step > 0:
                projected = multi_linear(
                    self.weights_per_step,
                    self.in_proj_weight,
                    query,
                    offset=current_streaming_offset,
                )
            else:
                projected = torch.nn.functional.linear(  # pylint: disable=not-callable
                    query, self.in_proj_weight
                )
            packed = rearrange(
                projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads
            )
            q, k, v = torch.unbind(packed, dim=2)

        if self.rope:
            q, k = self.rope(q, k, offset=current_streaming_offset)

        k, v = self._complete_kv(k, v)

        # Attention
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v, is_causal=False, attn_mask=attention_mask
        )
        x = x.transpose(1, 2)

        # output projection
        x = rearrange(x, "b t h d -> b t (h d)")
        if self.weights_per_step > 0:
            x = multi_linear(
                self.weights_per_step,
                self.out_proj.weight,
                x,
                offset=current_streaming_offset,
            )
        else:
            x = self.out_proj(x)
        return x
