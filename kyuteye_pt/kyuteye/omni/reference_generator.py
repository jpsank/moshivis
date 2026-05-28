"""Generate short reference snippets from conversation context.

Simplified port of ``LLMReferenceGenerator`` from kyutai-labs/moshi-rag. The
multi-profile / summarization / Gradium machinery is removed -- we keep the
single-LLM path that drives the simplified prompt template.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from kyuteye.omni.llm_client import LLMClient

logger = logging.getLogger(__name__)

REFERENCE_PROMPT_PATH = Path(__file__).parent / "prompts" / "reference_prompt.txt"

# History entry: (number of turns at the time the reference was generated, reference text)
ReferenceHistory = list[tuple[int, str]]


def load_reference_prompt() -> str:
    with open(REFERENCE_PROMPT_PATH) as f:
        return f.read()


class LLMReferenceGenerator:
    """Build a short ``Reference: ...`` line from a running conversation."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient(
            system_prompt="You are a helpful assistant.",
            prompt=load_reference_prompt(),
        )

    @staticmethod
    def _format_context(context: str, history: ReferenceHistory) -> tuple[str, int]:
        """Normalize raw transcript into the format the prompt expects."""
        turns: list[tuple[str, str]] = []
        for turn in context.split("\n"):
            if turn.startswith("user:"):
                text = turn.split("user:", 1)[1].strip()
                turns.append(("Human", "".join(c for c in text if c.isprintable()).strip()))
            elif turn.startswith("moshi:"):
                text = turn.split("moshi:", 1)[1].strip()
                turns.append(("moshi", "".join(c for c in text if c.isprintable()).strip()))

        # Drop a trailing in-progress moshi turn (it may have triggered RAG mid-sentence).
        if turns and turns[-1][0] == "moshi":
            turns = turns[:-1]
        # Drop a leading orphan moshi turn.
        if turns and turns[0][0] == "moshi":
            turns = turns[1:]

        formatted = ""
        j = 0
        for i, (role, text) in enumerate(turns):
            formatted += f"{role}: {text}\n" if text else f"{role}:\n"
            if j < len(history) and i + 1 == history[j][0]:
                formatted += f"Reference: {history[j][1]}\n"
                j += 1
        formatted += "Reference:"
        return formatted, len(turns)

    async def generate(
        self,
        context: str,
        history: ReferenceHistory,
        *,
        llm_call_timeout: float | None = 1.5,
        max_tokens: int = 512,
    ) -> tuple[str, int]:
        """Return ``(reference_text, num_turns)``. Empty string on failure / timeout."""
        formatted, num_turns = self._format_context(context, history)
        if num_turns == 0:
            return "", 0

        loop = asyncio.get_event_loop()
        try:
            fut = loop.run_in_executor(
                None, self.llm.generate, self.llm.prompt, formatted, max_tokens
            )
            if llm_call_timeout is not None and llm_call_timeout > 0:
                text = await asyncio.wait_for(fut, timeout=llm_call_timeout)
            else:
                text = await fut
        except asyncio.TimeoutError:
            logger.warning("[Reference] LLM timed out after %ss", llm_call_timeout)
            return "", num_turns
        except Exception as e:
            logger.warning("[Reference] LLM call failed: %s", e)
            return "", num_turns

        ref = (text or "").strip()
        # The model sometimes echoes the "Reference:" prefix from the prompt.
        if ref.lower().startswith("reference:"):
            ref = ref[len("reference:") :].strip()
        return ref, num_turns

    def warmup(self) -> None:
        try:
            self.llm.warmup()
        except Exception as e:
            logger.warning("[Reference] warmup failed: %s", e)
