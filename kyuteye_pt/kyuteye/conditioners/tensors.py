"""Passthrough tensor conditioner -- consumes pre-encoded tensors as-is.

Ported verbatim from kyutai-labs/moshi-rag
(moshi/moshi/conditioners/tensors.py). Used to plumb the remote ARC
encoder's output (the ``reference_with_time`` attribute) into the
:class:`ConditionFuser` so it can be routed to the ``streaming_sum`` path.
"""

from kyuteye.conditioners.base import (
    ConditionType,
    TensorCondition,
    _BaseTensorConditioner,
)


class TensorConditioner(_BaseTensorConditioner[TensorCondition]):
    """Does basically nothing -- passes tensors straight through to the fuser."""

    def prepare(self, tensor: TensorCondition) -> TensorCondition:
        device = next(iter(self.parameters())).device
        return TensorCondition(
            tensor.tensor.to(device=device), tensor.mask.to(device=device)
        )

    def _get_condition(self, inputs: TensorCondition) -> ConditionType:
        return ConditionType(inputs.tensor, inputs.mask)
