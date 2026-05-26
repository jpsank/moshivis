# pylint: disable=protected-access
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Common API for streaming modules during inference"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

import torch

State = Dict[str, float | int | torch.Tensor]


class StreamingModule(torch.nn.Module):
    """Common API for streaming components."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = {}
        self._is_streaming = False

    @property
    def empty_streaming_state(self) -> bool:
        """whether streaming state is empty"""
        return len(self._streaming_state) == 0

    def has_streaming_attribute(self, key: str) -> bool:
        """Whether `key` exists in the current streaming state"""
        return self._is_streaming and key in self._streaming_state

    def add_streaming_attribute(
        self, key: str, value: float | int | torch.Tensor
    ) -> None:
        """Add `value` into streaming state's `key`"""
        self._streaming_state[key] = value

    def get_streaming_attribute(self, key: str, default: Any = None) -> Any:
        """Add `value` into streaming state's `key`"""
        return self._streaming_state.get(key, default)

    @property
    def is_streaming(self) -> bool:
        """in streaming mode"""
        return self._is_streaming

    def get_streaming_info_as_int(self, attr_name: str, default: int = 0) -> int:
        """Tries to get attr_name as an integer"""
        if self._is_streaming and attr_name in self._streaming_state:
            if isinstance(self._streaming_state[attr_name], int):
                return self._streaming_state[attr_name]  # type: ignore
            if isinstance(self._streaming_state[attr_name], torch.Tensor):
                return int(self._streaming_state[attr_name].item())  # type: ignore
            raise ValueError(
                f"Unexpected type {type(self._streaming_state[attr_name])} in streaming state"
            )
        return default

    @property
    def streaming_offset(self) -> int:
        """Shortcut to get the current temporal offset in streaming mode"""
        return self.get_streaming_info_as_int("offset", default=0)

    @streaming_offset.setter
    def streaming_offset(self, value: int | torch.Tensor) -> None:
        if not self._is_streaming:
            raise NotImplementedError(
                "Updating streaming offset of a non-streaming module"
            )
        self._streaming_state["offset"] = value  # type: ignore

    def _apply_named_streaming(self, fn: Callable) -> None:
        for name, module in self.named_modules():
            if isinstance(module, StreamingModule):
                fn(name, module)

    def _set_streaming(self, streaming: bool) -> None:
        def _set_streaming(_: str, module: StreamingModule) -> None:
            module._is_streaming = streaming

        self._apply_named_streaming(_set_streaming)

    @contextmanager
    def streaming(self) -> Iterator:
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._set_streaming(True)
        try:
            yield
        finally:
            self._set_streaming(False)
            self.reset_streaming()

    def streaming_forever(self, batch_size: Optional[int] = None) -> None:
        """Set in permanent streaming state.

        When ``batch_size`` is provided, it is broadcast to all sub-modules via
        ``_streaming_state['batch_size']`` so per-slot logic (exec_mask,
        masked reset) can consult it. Modules that pre-allocate KV caches
        will still do so lazily on first forward, sized from the actual input
        tensor shape rather than this value.
        """
        self._set_streaming(True)
        if batch_size is not None:

            def _record(_: str, module: StreamingModule) -> None:
                module._streaming_state["batch_size"] = batch_size  # type: ignore[assignment]

            self._apply_named_streaming(_record)

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        """Set the per-slot execution mask, propagated to all sub-modules.

        ``exec_mask`` is a ``[batch_size]`` bool tensor: ``True`` for slots
        whose streaming state should advance this step, ``False`` for slots
        that should be skipped (idle). Matches the upstream
        ``moshi.modules.streaming.StreamingModule.set_exec_mask`` semantics
        and is the foundation for batched inference where users arrive and
        leave at independent times.

        Individual modules (attention KV cache, the LM gen wrapper) are
        responsible for honoring the mask -- this method just stores it.
        Storage is by key ``'exec_mask'`` in ``_streaming_state`` so any
        layer can read it with ``self.get_streaming_attribute('exec_mask')``.
        """

        def _set(_: str, module: StreamingModule) -> None:
            module._streaming_state["exec_mask"] = exec_mask  # type: ignore[assignment]

        self._apply_named_streaming(_set)

    @property
    def exec_mask(self) -> Optional[torch.Tensor]:
        """The current ``[batch_size]`` execution mask, or ``None`` if unset."""
        return self.get_streaming_attribute("exec_mask", None)

    def reset_streaming(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        """Reset the streaming state.

        :param reset_mask: Optional ``[batch_size]`` bool tensor selecting
            which slots to reset. ``None`` (default) resets all slots --
            matching the original single-batch behavior. When provided,
            modules are expected to scope their reset to the masked slots.
            The base implementation clears each sub-module's state dict
            entirely when no mask is given; when a mask is given, it removes
            only entries that aren't tensors (since per-slot tensor surgery
            is module-specific).
        """
        if reset_mask is None:

            def _reset(_: str, module: StreamingModule) -> None:
                module._streaming_state.clear()

            self._apply_named_streaming(_reset)
        else:

            def _reset_masked(_: str, module: StreamingModule) -> None:
                # Per-slot: subclasses with tensor-valued state are expected to
                # override their reset behavior. The default safe action is to
                # clear scalar state and leave tensor state alone (so the
                # subclass can apply its own per-slot logic).
                module._streaming_state["reset_mask"] = reset_mask  # type: ignore[assignment]
                for key in list(module._streaming_state.keys()):
                    val = module._streaming_state[key]
                    if not isinstance(val, torch.Tensor):
                        if key in ("reset_mask", "batch_size", "exec_mask"):
                            continue
                        # Non-tensor scalar state (e.g. ``offset`` int) is a
                        # session-level counter; reset it when *any* slot
                        # resets, since we can't have a different scalar per
                        # slot. Multi-batch deployments should store the
                        # offset as a tensor and override this method.
                        if reset_mask.any().item():
                            del module._streaming_state[key]

            self._apply_named_streaming(_reset_masked)

    def get_streaming_state(self) -> State:
        """Return the streaming state, including that of sub-modules."""
        state: State = {}

        def _add(name: str, module: StreamingModule) -> None:
            if name:
                name += "."
            for key, value in module._streaming_state.items():
                state[name + key] = value

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: State) -> None:
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule) -> None:
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                # complexity is not ideal here, but probably fine.
                if key.startswith(name):
                    local_key = key[len(name) :]
                    if "." not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        assert len(state) == 0, list(state.keys())

    def flush(self, x: Optional[torch.Tensor] = None) -> Optional["StreamingModule"]:
        """Flush any remaining outputs that were waiting for completion.
        Typically, for convolutions, this will add the final padding
        and process the last buffer.

        This should take an optional argument `x`, which will be provided
        if a module before this one in the streaming pipeline has already
        spitted out a flushed out buffer.
        """
        if x is None:
            return None
        return self(x)
