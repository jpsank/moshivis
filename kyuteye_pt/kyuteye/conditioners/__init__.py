"""Conditioner modules ported from kyutai-labs/moshi-rag.

The PyTorch backend of MoshiVis is inference-only; the conditioners here exist
so that the model's state-dict layout matches MoshiRAG's, enabling a future
combined fine-tune to plug in without architectural surgery. At inference time
the only path exercised by default is ``streaming_sum`` (used by
``kyuteye.omni`` for asynchronous RAG injection).

See ``kyuteye_pt/kyuteye/conditioners/base.py`` docstring for the architectural
overview. The ``T5Conditioner`` and local ``ArcEncoder`` from moshi-rag are
intentionally not ported: they pull in heavy dependencies that MoshiVis does
not need, and the remote ARC encoder service (see
``kyuteye.omni.arc_encoder_client``) covers the runtime path.
"""

from kyuteye.conditioners.base import (
    BaseConditioner,
    ConditionAttributes,
    ConditionFuser,
    ConditionProvider,
    ConditionTensors,
    ConditionType,
    TensorCondition,
    _BaseTensorConditioner,
    _BaseTextConditioner,
    dropout_all_conditions,
    dropout_condition_,
    dropout_tensor,
)
from kyuteye.conditioners.tensors import TensorConditioner
from kyuteye.conditioners.text import LUTConditioner, NoopTokenizer

__all__ = [
    "BaseConditioner",
    "ConditionAttributes",
    "ConditionFuser",
    "ConditionProvider",
    "ConditionTensors",
    "ConditionType",
    "LUTConditioner",
    "NoopTokenizer",
    "TensorCondition",
    "TensorConditioner",
    "_BaseTensorConditioner",
    "_BaseTextConditioner",
    "dropout_all_conditions",
    "dropout_condition_",
    "dropout_tensor",
]
