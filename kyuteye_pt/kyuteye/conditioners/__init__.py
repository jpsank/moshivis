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

# ARC encoder is in its own module because it pulls xformers (optional dep).
# Module import is fail-safe -- only constructing the conditioner triggers
# the xformers check (see arc_encoder.py:_require_xformers).
from kyuteye.conditioners.arc_encoder import (  # noqa: E402
    ArcEncoderConditioner,
    ArcEncoderTokenizer,
    ArcEncoderTransformer,
    EmbProjector,
    MultiArcEncoderConditioner,
)

__all__ = [
    "ArcEncoderConditioner",
    "ArcEncoderTokenizer",
    "ArcEncoderTransformer",
    "BaseConditioner",
    "ConditionAttributes",
    "ConditionFuser",
    "ConditionProvider",
    "ConditionTensors",
    "ConditionType",
    "EmbProjector",
    "LUTConditioner",
    "MultiArcEncoderConditioner",
    "NoopTokenizer",
    "TensorCondition",
    "TensorConditioner",
    "_BaseTensorConditioner",
    "_BaseTextConditioner",
    "dropout_all_conditions",
    "dropout_condition_",
    "dropout_tensor",
]
