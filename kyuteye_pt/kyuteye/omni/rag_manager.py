"""Per-channel asynchronous RAG lifecycle.

Adapted from kyutai-labs/moshi-rag (moshi/moshi/inference_utils/rag_manager.py).
The structure is the same: one in-flight retrieval task at a time, cancellable
on shutdown, optional ``wait_steps`` to defer the LLM call until the model has
generated enough buffer to mask retrieval latency. The retriever itself is
plugged in via the :mod:`kyuteye.omni.retrievers` registry, so the manager has
no opinion on whether it is an LLM, a vector store, or a hand-coded lookup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Awaitable, Callable

from kyuteye.omni.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

ReferenceHistory = list[tuple[int, str]]


class OmniRAGManager:
    """Drives one background retrieval task per channel."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        rag_timeout: float = 1.5,
        max_tokens: int = 512,
    ) -> None:
        self.retriever = retriever
        self.rag_timeout = rag_timeout
        self.max_tokens = max_tokens
        self._history: ReferenceHistory = []
        self._wait_steps_remaining: int = 0
        self._wait_event: asyncio.Event | None = None
        self._pending_task: asyncio.Task | None = None
        self._stack: contextlib.AsyncExitStack | None = None

    async def __aenter__(self) -> "OmniRAGManager":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        self._stack.push_async_callback(self._cancel_and_await_pending)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._stack is not None
        try:
            return await self._stack.__aexit__(exc_type, exc, tb)
        finally:
            self._stack = None

    async def _cancel_and_await_pending(self) -> None:
        task = self._pending_task
        self._pending_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def trigger(
        self,
        *,
        wait_steps: int = 0,
        handle_reference_fn: Callable[[str], Awaitable[None]] | None = None,
        context_provider: Callable[[], str] | None = None,
    ) -> None:
        """Spawn a background retrieval task. Cancels any previous one in flight."""
        if self._stack is None:
            raise RuntimeError("OmniRAGManager.trigger called outside of `async with` scope")
        await self._cancel_and_await_pending()

        if wait_steps > 0:
            self._wait_steps_remaining = wait_steps
            self._wait_event = asyncio.Event()
        else:
            self._wait_event = None
            self._wait_steps_remaining = 0

        loop = asyncio.get_running_loop()
        self._pending_task = loop.create_task(
            self._background_task(handle_reference_fn, context_provider)
        )

    async def _background_task(
        self,
        handle_reference_fn: Callable[[str], Awaitable[None]] | None,
        context_provider: Callable[[], str] | None,
    ) -> None:
        try:
            if self._wait_event is not None:
                await self._wait_event.wait()
                self._wait_event = None

            context = context_provider() if context_provider is not None else ""
            logger.info(
                "[RAG] retrieving (ctx_len=%d, tail=%r)", len(context), context[-120:]
            )
            t0 = time.time()
            reference_text, num_turns = await self.retriever.retrieve(
                context,
                self._history,
                timeout=self.rag_timeout,
                max_tokens=self.max_tokens,
            )
            elapsed = time.time() - t0
            if num_turns > 0 and reference_text:
                self._history.append((num_turns, reference_text))
            logger.info(
                "[RAG] retrieved in %.3fs: %r", elapsed, reference_text[:120]
            )
            if handle_reference_fn is not None:
                await handle_reference_fn(reference_text)
        except asyncio.CancelledError:
            logger.info("[RAG] retrieval cancelled")
            raise
        except Exception as e:
            logger.error("[RAG] retrieval failed: %s", e)

    def step(self) -> None:
        """Tick the wait-steps counter. Call once per model step."""
        if self._wait_steps_remaining > 0:
            self._wait_steps_remaining -= 1
            if self._wait_steps_remaining == 0 and self._wait_event is not None:
                self._wait_event.set()

    def cancel_pending(self) -> None:
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    def reset(self) -> None:
        self.cancel_pending()
        self._wait_steps_remaining = 0
        self._wait_event = None
        self._history = []
