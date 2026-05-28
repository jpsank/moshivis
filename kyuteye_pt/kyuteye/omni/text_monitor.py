"""Detokenize the model's text stream and detect Omni trigger patterns.

Two kinds of trigger:

* RAG: a configurable substring (default ``<ret>``) appears in the recent
  detokenized text, or a configurable SentencePiece token id is emitted.
  MoshiVis is not trained to emit a dedicated retrieval token like
  MoshiRAG, so the substring path is the realistic one -- developers can
  prompt or finetune the model to say ``<ret>`` mid-turn when it doesn't
  know an answer.

* Tool: a ``[TOOL: name(args)]`` substring. The monitor waits until the
  closing ``]`` arrives before firing, so partial pieces don't trigger
  half-baked calls.

The monitor itself is a stateless-per-emit accumulator: feed it one text
piece (already de-tokenized, with the SentencePiece ``▁`` replaced by a
space) per model step, and it returns a list of :class:`OmniEvent`
records describing what was detected, plus the trimmed text safe to
forward to the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from kyuteye.omni.tools import ToolCall, parse_tool_call

logger = logging.getLogger(__name__)


@dataclass
class OmniEvent:
    kind: Literal["rag", "tool"]
    span: tuple[int, int]  # offsets into the running buffer
    raw: str
    tool: Optional[ToolCall] = None
    consumed_text: str = ""


@dataclass
class TextStreamMonitor:
    """Accumulates detokenized text and surfaces RAG / tool trigger events.

    Args:
        rag_trigger: Substring that fires a RAG retrieval (default ``<ret>``).
        rag_token_id: Optional sentencepiece token id that also fires RAG.
        tool_start: Opening marker that begins a tool call buffer.
        tool_end: Closing marker that completes a tool call.
        strip_triggers: When True, remove matched triggers from the text
            that gets forwarded to the UI / transcript.
    """

    rag_trigger: str = "<ret>"
    rag_token_id: Optional[int] = None
    tool_start: str = "[TOOL:"
    tool_end: str = "]"
    strip_triggers: bool = True

    _buffer: str = field(default="", init=False)
    _transcript: str = field(default="", init=False)

    def reset(self) -> None:
        self._buffer = ""
        self._transcript = ""

    @property
    def transcript(self) -> str:
        """Best-effort running transcript of model text emitted so far."""
        return self._transcript

    def consume(self, text_piece: str, token_id: int | None = None) -> tuple[str, list[OmniEvent]]:
        """Feed one detokenized text piece. Returns ``(text_to_emit, events)``.

        ``text_to_emit`` is the portion of the buffer that has been "flushed"
        and is safe to forward to the UI -- i.e. everything up to (but not
        including) any in-progress tool-call pattern.
        """
        events: list[OmniEvent] = []
        self._buffer += text_piece

        # 1. Token-id-level RAG trigger (fires immediately, no buffering needed).
        if (
            self.rag_token_id is not None
            and token_id is not None
            and token_id == self.rag_token_id
        ):
            events.append(
                OmniEvent(kind="rag", span=(0, 0), raw="<RAG_TOKEN>")
            )

        # 2. Substring RAG triggers (drain all occurrences in the buffer).
        while True:
            idx = self._buffer.find(self.rag_trigger)
            if idx == -1:
                break
            events.append(
                OmniEvent(
                    kind="rag",
                    span=(idx, idx + len(self.rag_trigger)),
                    raw=self.rag_trigger,
                )
            )
            if self.strip_triggers:
                self._buffer = (
                    self._buffer[:idx] + self._buffer[idx + len(self.rag_trigger):]
                )
            else:
                # Move past the trigger so we don't re-fire on the same instance.
                # We do this by replacing the first character with a marker the
                # subsequent find will skip; a simpler approach is to keep
                # strip_triggers=True. The else-branch here is mostly for tests.
                break

        # 3. Tool patterns: only fire once a complete `[TOOL: ... ]` has arrived.
        while True:
            start = self._buffer.find(self.tool_start)
            if start == -1:
                break
            end = self._buffer.find(self.tool_end, start + len(self.tool_start))
            if end == -1:
                break  # incomplete -- wait for more text
            raw = self._buffer[start : end + len(self.tool_end)]
            call = parse_tool_call(raw)
            event = OmniEvent(
                kind="tool",
                span=(start, end + len(self.tool_end)),
                raw=raw,
                tool=call,
            )
            events.append(event)
            if self.strip_triggers:
                self._buffer = self._buffer[:start] + self._buffer[end + len(self.tool_end):]
            else:
                # Drop only the matched span so we keep scanning.
                self._buffer = self._buffer[:start] + " " + self._buffer[end + len(self.tool_end):]

        # 4. Decide what is safe to emit: everything up to the first incomplete
        #    tool-start (so we don't leak half a "[TOOL:" prefix to the UI).
        pending_start = self._buffer.find(self.tool_start)
        if pending_start == -1:
            emit = self._buffer
            self._buffer = ""
        else:
            emit = self._buffer[:pending_start]
            self._buffer = self._buffer[pending_start:]

        if emit:
            self._transcript += emit
        for ev in events:
            if ev.kind == "tool":
                ev.consumed_text = ev.raw
        return emit, events
