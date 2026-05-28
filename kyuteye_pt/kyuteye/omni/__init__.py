"""Omni Assistant: MoshiVis + asynchronous RAG + pluggable tool calling.

Combines MoshiVis (vision-conditioned Moshi) with two background subsystems:

* Asynchronous knowledge retrieval, ported from kyutai-labs/moshi-rag, that
  triggers when the model emits a configurable retrieval pattern (``<ret>``)
  in its text stream. Retrieved context is encoded and injected back as an
  extra cross-attention source.
* Pluggable tool calling: text segments matching ``[TOOL: name(args)]``
  fire registered Python callables in the background.

Both retrievers and tools are configured as Python plugins -- there is no
YAML / config file involved. Register your own by importing this package
inside your plugin module and calling :func:`set_retriever` /
:func:`register_tool`, then pass ``--omni-plugin=your.module`` to the
server.
"""

from kyuteye.omni.context_injector import ContextInjector, encode_text_to_ca_kv
from kyuteye.omni.llm_client import LLMClient
from kyuteye.omni.rag_manager import OmniRAGManager
from kyuteye.omni.reference_generator import LLMReferenceGenerator
from kyuteye.omni.retrievers import (
    BaseRetriever,
    LLMRetriever,
    get_retriever,
    set_retriever,
)
from kyuteye.omni.text_monitor import OmniEvent, TextStreamMonitor
from kyuteye.omni.tools import (
    BaseTool,
    ToolCall,
    ToolRegistry,
    default_registry,
    register_tool,
    tool,
)

__all__ = [
    "BaseRetriever",
    "BaseTool",
    "ContextInjector",
    "LLMClient",
    "LLMReferenceGenerator",
    "LLMRetriever",
    "OmniEvent",
    "OmniRAGManager",
    "TextStreamMonitor",
    "ToolCall",
    "ToolRegistry",
    "default_registry",
    "encode_text_to_ca_kv",
    "get_retriever",
    "register_tool",
    "set_retriever",
    "tool",
]
