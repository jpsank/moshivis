"""Pluggable retriever interface for Omni Assistant.

A *retriever* takes the running conversation context (transcript of user
and model turns, newline-separated, prefixed with ``user:`` / ``moshi:``)
and returns a short factual reference string that is fed back into Moshi.

The default :class:`LLMRetriever` mirrors the kyutai-labs/moshi-rag design:
it forwards the formatted transcript to an OpenAI-compatible LLM. Users who
want a vector store, a domain knowledge base, or a hand-coded fact lookup
should subclass :class:`BaseRetriever` and register an instance via
:func:`set_retriever` from their plugin module.
"""

from __future__ import annotations

import abc
import logging
from typing import Optional

from kyuteye.omni.reference_generator import LLMReferenceGenerator, ReferenceHistory

logger = logging.getLogger(__name__)


class BaseRetriever(abc.ABC):
    """Async retrieval backend. Implement :meth:`retrieve` in a subclass."""

    @abc.abstractmethod
    async def retrieve(
        self,
        context: str,
        history: ReferenceHistory,
        *,
        timeout: float | None = 1.5,
        max_tokens: int = 512,
    ) -> tuple[str, int]:
        """Return ``(reference_text, num_turns_consumed)``.

        ``reference_text`` should be a single concise factual line (no
        newlines). Empty string signals "no reference produced" (the
        orchestrator will surface ``[RET_FAILED]`` to the UI in that case).

        ``num_turns_consumed`` is the number of turns in ``context`` that
        the retriever actually used; the orchestrator uses it to attribute
        the reference to a position in the running history.
        """

    def warmup(self) -> None:
        """Optional. Called once at server startup."""


class LLMRetriever(BaseRetriever):
    """Default retriever: forwards the transcript to an OpenAI-compatible LLM."""

    def __init__(self, generator: LLMReferenceGenerator | None = None) -> None:
        self.generator = generator or LLMReferenceGenerator()

    async def retrieve(
        self,
        context: str,
        history: ReferenceHistory,
        *,
        timeout: float | None = 1.5,
        max_tokens: int = 512,
    ) -> tuple[str, int]:
        return await self.generator.generate(
            context, history, llm_call_timeout=timeout, max_tokens=max_tokens
        )

    def warmup(self) -> None:
        self.generator.warmup()


_active_retriever: Optional[BaseRetriever] = None


def set_retriever(retriever: BaseRetriever) -> None:
    """Register the retriever the server should use. Call from a plugin module."""
    global _active_retriever
    _active_retriever = retriever
    logger.info("[Omni] retriever set to %s", type(retriever).__name__)


def get_retriever() -> BaseRetriever | None:
    """Return the currently-registered retriever, or ``None`` if unset."""
    return _active_retriever
