"""Conditioning core: ``ConditionAttributes``, ``ConditionProvider``,
``ConditionFuser``, ``BaseConditioner``.

Ported essentially verbatim from
kyutai-labs/moshi-rag (moshi/moshi/conditioners/base.py). Kept here so that
the merged MoshiVis architecture can host the same conditioner state-dict
layout as MoshiRAG. No modifications were made to the public API; the only
changes are import paths and dropped docstring references to audiocraft.

Architectural sketch:

* :class:`ConditionAttributes` carries a sample's raw conditioning inputs
  (text dict and tensor dict).
* :class:`ConditionProvider` owns the per-attribute :class:`BaseConditioner`
  modules. ``prepare`` does any CPU-side tokenization / collation; ``forward``
  produces a ``dict[name, ConditionType]`` -- one tensor + mask per attribute.
* :class:`ConditionFuser` routes those tensors into one of four pathways the
  LM consumes (``sum``, ``prepend``, ``cross``, ``streaming_sum``).
* :class:`BaseConditioner` is the abstract base for individual conditioners.
  Two concrete subclasses ship in this package: :class:`LUTConditioner`
  (text → small LUT, used for ``first_speaker``) and
  :class:`TensorConditioner` (passthrough, used for pre-encoded ARC outputs).
"""

from __future__ import annotations

import logging
import typing as tp
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain

import torch
from torch import nn

logger = logging.getLogger(__name__)
ConditionTensors = dict[str, "ConditionType"]


class ConditionType(tp.NamedTuple):
    """Return type for a conditioner: ``(condition, mask)`` pair."""

    condition: torch.Tensor
    mask: torch.Tensor


class TokenizedText(tp.NamedTuple):
    tokens: torch.Tensor  # long
    mask: torch.Tensor  # bool


class EmbeddedText(tp.NamedTuple):
    embeddings: torch.Tensor  # float
    mask: torch.Tensor


@dataclass(frozen=True)
class TensorCondition:
    """Input to a tensor conditioner. ``tensor`` is ``[B|1, T, D]``, ``mask`` is ``[B|1, T]``."""

    tensor: torch.Tensor
    mask: torch.Tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "TensorCondition":
        B, T, _ = tensor.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=tensor.device)
        return TensorCondition(tensor, mask)

    @staticmethod
    def cat(conditions: tp.Sequence["TensorCondition"]) -> "TensorCondition":
        assert conditions, "Cannot cat empty list."
        ref_tensor = conditions[0].tensor
        B, _, D = ref_tensor.shape
        assert B == 1
        B = len(conditions)
        T = max(c.tensor.shape[1] for c in conditions)
        mask = torch.zeros(B, T, dtype=torch.bool, device=ref_tensor.device)
        tensor = torch.zeros(B, T, D, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for b, c in enumerate(conditions):
            tensor[b, : c.tensor.shape[1], :] = c.tensor[0]
            mask[b, : c.mask.shape[1]] = c.mask[0]
        return TensorCondition(tensor, mask)


@dataclass
class ConditionAttributes:
    """A sample's conditioning inputs: text strings and pre-encoded tensors."""

    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    tensor: tp.Dict[str, TensorCondition] = field(default_factory=dict)

    @property
    def text_attributes(self) -> tp.Iterable[str]:
        return self.text.keys()

    @property
    def tensor_attributes(self) -> tp.Iterable[str]:
        # NB: kyutai-labs/moshi-rag has a dormant copy-paste bug here that
        # returns ``self.text.keys()`` (see
        # ``moshi-rag/moshi/moshi/conditioners/base.py:92``). Neither
        # MoshiRAG nor this port currently calls this property -- the
        # collation paths use ``ConditionProvider.tensor_conditions``
        # instead -- so fixing it here has no behaviour change. Kept as
        # a paper trail in case a future caller reaches for the property.
        return self.tensor.keys()

    @staticmethod
    def condition_types() -> tp.FrozenSet[str]:
        return frozenset(["text", "tensor"])

    def copy(self) -> "ConditionAttributes":
        return ConditionAttributes(dict(self.text), dict(self.tensor))


Prepared = tp.TypeVar("Prepared")


class BaseConditioner(nn.Module, tp.Generic[Prepared]):
    """Abstract base for conditioners. Subclasses implement ``prepare`` and ``_get_condition``."""

    def __init__(
        self,
        dim: int,
        output_dim: int,
        device: tp.Union[torch.device, str],
        force_linear: bool = True,
        pad_empty: bool = True,
        output_bias: bool = False,
        learn_padding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.pad_empty = pad_empty
        self.device = device
        self.output_proj: nn.Module
        if force_linear or dim != output_dim:
            self.output_proj = nn.Linear(dim, output_dim, bias=output_bias, device=device)
            assert not output_bias
        else:
            self.output_proj = nn.Identity()
        self.learnt_padding: tp.Optional[torch.Tensor]
        if learn_padding:
            self.learnt_padding = nn.Parameter(
                torch.randn(1, 1, output_dim, device=device), requires_grad=True
            )
            self.learnt_padding.data *= 0.2
        else:
            self.learnt_padding = None

    def prepare(self, *args, **kwargs) -> Prepared:
        raise NotImplementedError()

    def _get_condition(self, inputs: Prepared) -> ConditionType:
        raise NotImplementedError()

    def forward(self, inputs: Prepared) -> ConditionType:
        cond, mask = self._get_condition(inputs)
        B, T, C = cond.shape
        if T == 0 and self.pad_empty:
            cond = torch.zeros(B, T, C, device=cond.device, dtype=cond.dtype)
            mask = torch.zeros_like(cond[..., 0], dtype=torch.bool)
            return ConditionType(cond, mask)

        dtype = cond.dtype
        for weight in self.output_proj.parameters():
            dtype = weight.dtype
        cond = self.output_proj(cond.to(dtype))

        maskf = mask.float()[..., None]
        if self.learnt_padding is not None:
            cond = cond * maskf + self.learnt_padding * (1 - maskf)
        else:
            cond = cond * maskf

        return ConditionType(cond, mask)


class _BaseTextConditioner(BaseConditioner[Prepared]):
    pass


class _BaseTensorConditioner(BaseConditioner[Prepared]):
    pass


class ConditionProvider(nn.Module):
    """Holds the per-attribute conditioner modules and dispatches inputs."""

    def __init__(
        self,
        conditioners: tp.Dict[str, BaseConditioner],
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners).to(device)

    @property
    def text_conditions(self) -> list[str]:
        return [k for k, v in self.conditioners.items() if isinstance(v, _BaseTextConditioner)]

    @property
    def tensor_conditions(self) -> list[str]:
        return [k for k, v in self.conditioners.items() if isinstance(v, _BaseTensorConditioner)]

    def _collate_text(
        self, samples: tp.Sequence[ConditionAttributes]
    ) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        for sample in samples:
            for condition in self.text_conditions:
                if condition in sample.text:
                    out[condition].append(sample.text[condition])
        return out

    def _collate_tensors(
        self, samples: tp.Sequence[ConditionAttributes]
    ) -> tp.Dict[str, TensorCondition]:
        per_attribute = defaultdict(list)
        out: tp.Dict[str, TensorCondition] = {}
        for sample in samples:
            for attribute in self.tensor_conditions:
                per_attribute[attribute].append(sample.tensor[attribute])
        for attribute in self.tensor_conditions:
            out[attribute] = TensorCondition.cat(per_attribute[attribute])
        return out

    def prepare(
        self, inputs: tp.Sequence[ConditionAttributes]
    ) -> tp.Dict[str, tp.Any]:
        assert all(isinstance(x, ConditionAttributes) for x in inputs), (
            f"expected ConditionAttributes, got {set(type(x) for x in inputs)}"
        )
        output: tp.Dict[str, tp.Any] = {}
        text = self._collate_text(inputs)
        tensors = self._collate_tensors(inputs)
        provided = set(text.keys()) | set(tensors.keys())
        unknown = provided - set(self.conditioners.keys())
        assert not unknown, f"unknown condition keys: {unknown}"
        missing = set(self.conditioners.keys()) - provided
        if missing:
            raise RuntimeError(f"Some conditioners did not receive an input: {missing}")
        for attribute, batch in chain(text.items(), tensors.items()):
            output[attribute] = self.conditioners[attribute].prepare(batch)
        return output

    def forward(
        self, prepared: tp.Dict[str, tp.Any]
    ) -> tp.Dict[str, ConditionType]:
        output: tp.Dict[str, ConditionType] = {}
        for name, inputs in prepared.items():
            cond, mask = self.conditioners[name](inputs)
            output[name] = ConditionType(cond, mask)
        return output


class ConditionFuser(nn.Module):
    """Routes named conditions into the LM's four conditioning pathways."""

    FUSING_METHODS = ["sum", "prepend", "cross", "streaming_sum"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]]):
        super().__init__()
        unknown = set(fuse2cond.keys()) - set(self.FUSING_METHODS)
        assert not unknown, (
            f"invalid fuse method(s) {unknown}; allowed: {self.FUSING_METHODS}"
        )
        # Ensure every method key is present (callers index into them).
        self.fuse2cond: tp.Dict[str, tp.List[str]] = {m: [] for m in self.FUSING_METHODS}
        self.fuse2cond.update(fuse2cond)
        self.cond2fuse: tp.Dict[str, str] = {}
        for method, conditions in self.fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = method

    @property
    def has_conditions(self) -> bool:
        return bool(self.cond2fuse)

    @property
    def has_prepend(self) -> bool:
        return bool(self.fuse2cond["prepend"])

    def get_cross(self, conditions: ConditionTensors) -> torch.Tensor | None:
        cross = None
        for name in self.fuse2cond["cross"]:
            cond, mask = conditions[name]
            cond = cond * mask.unsqueeze(-1)
            cross = cond if cross is None else torch.cat([cross, cond], dim=1)
        return cross

    def get_sum(self, conditions: ConditionTensors) -> torch.Tensor | None:
        out = None
        for name in self.fuse2cond["sum"]:
            cond, mask = conditions[name]
            cond = cond * mask.unsqueeze(-1)
            assert cond.shape[1] == 1, cond.shape
            out = cond if out is None else out + cond
        return out

    def get_prepend(self, conditions: ConditionTensors) -> torch.Tensor | None:
        prepend = None
        for name in self.fuse2cond["prepend"]:
            cond, mask = conditions[name]
            cond = cond * mask.unsqueeze(-1)
            prepend = cond if prepend is None else torch.cat([cond, prepend], dim=1)
        if prepend is not None:
            extra_sum = self.get_sum(conditions)
            if extra_sum is not None:
                prepend = prepend + extra_sum
        return prepend

    def get_streaming_sum(self, conditions: ConditionTensors) -> torch.Tensor | None:
        out = None
        for name in self.fuse2cond["streaming_sum"]:
            cond, mask = conditions[name]
            cond = cond * mask.unsqueeze(-1)
            if out is None:
                out = cond
            else:
                max_len = max(out.shape[1], cond.shape[1])
                if out.shape[1] < max_len:
                    cond[:, : out.shape[1]] += out
                    out = cond
                else:
                    out[:, : cond.shape[1]] += cond
        return out
