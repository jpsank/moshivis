"""OpenAI-compatible LLM client.

Ported from kyutai-labs/moshi-rag (moshi/moshi/llm/client.py) with two
adjustments: the ``openai`` import is lazy so the rest of the omni package
can be used without it, and the constructor accepts an explicit
``temperature`` value (MoshiRAG hard-codes ``1.0``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required for the default LLM-based "
                "retriever. Install with `pip install openai`, or register a "
                "custom retriever via `omni.set_retriever(...)`."
            ) from e

        self.model_name = (
            model_name if model_name is not None else os.environ.get("LLM_MODEL_NAME", "")
        )
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.temperature = temperature

        resolved_base = base_url or os.environ.get("LLM_BASE_URL")
        if not resolved_base:
            raise RuntimeError(
                "LLM_BASE_URL is not set and no base_url was passed to LLMClient."
            )
        resolved_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", None)
        self.client = OpenAI(base_url=resolved_base, api_key=resolved_key)

    def _build_messages(self, prompt_text: str, context: str = "") -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt_text + context}]}
        )
        return messages

    def warmup(self, prompt: str | None = None) -> None:
        """Smoke-test the endpoint with a short generation."""
        self.generate(prompt=prompt or self.prompt, context="", max_new_tokens=5)

    def generate(
        self,
        prompt: str,
        context: str,
        max_new_tokens: int = 512,
        stop_token: str | None = "\n",
        **kwargs: Any,
    ) -> str:
        """Blocking single-shot generation. Call from a worker thread."""
        messages = self._build_messages(prompt or self.prompt, context)
        t0 = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            stop=[stop_token] if stop_token is not None else None,
            **kwargs,
        )
        text_response = response.choices[0].message.content or ""
        text_response = text_response.strip().split("\n", 1)[0] if text_response else ""
        logger.info(
            "[LLM] %s -> %r (%.2fs)",
            self.model_name,
            text_response[:80],
            time.monotonic() - t0,
        )
        return text_response
