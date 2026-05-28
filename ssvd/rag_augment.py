# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "openai>=1.0",
#     "rich",
#     "tqdm",
# ]
# ///
"""Synthetic RAG training-data generator for a combined MoshiVis + MoshiRAG fine-tune.

This is a *sketch* of the data pipeline, intentionally minimal: it shows the
shape of training examples that a combined fine-tune would learn from and
provides a working end-to-end path from a topic / SSVD dialogue to JSONL
output. Production use would want quality filters, deduplication,
human-in-the-loop curation, and a much larger seed set.

Three modes:

1. ``augment_visual`` -- take an existing SSVD-generated visual dialogue
   (or any conversation indexed by image path) and use an LLM to insert
   ``<ret>`` markers before factual claims, then synthesize the reference
   document for each marker. The image is preserved so the resulting
   example trains both the vision pathway and the RAG pathway.

2. ``generate_text`` -- given a seed topic, ask an LLM to (a) generate a
   conversation about it, (b) identify the factual turns, (c) write a
   reference document for each. No image. Useful for getting text-only RAG
   capability without diluting the visual training signal.

3. ``generate_tools`` -- given a tool-spec file (name, description, arg
   schema, example output) and a topic list, ask an LLM to write
   conversations where moshi emits ``[TOOL: name(arg=value)]`` calls and
   the simulated tool output is provided as a ``tool`` turn. Trains the
   model to (a) emit tool calls in the right places and (b) consume tool
   results via the same conditioner pathway as RAG references. No image
   by default; combine outputs with the visual modes for multi-modal
   tool use.

Output is JSONL, one example per line. Each example has:

    {
      "image_path": "path/to/image.png" | null,
      "turns": [
        {"role": "user", "text": "..."},
        {"role": "moshi", "text": "I will check ...", "rag_trigger": true},
        {"role": "reference", "text": "Reference: ..."},
        {"role": "moshi", "text": "Looking up. [TOOL: get_weather(location=\\"NYC\\")]"},
        {"role": "tool", "text": "get_weather: 72F, sunny"},
        {"role": "moshi", "text": "It's 72F and sunny in NYC."},
        ...
      ]
    }

The ``rag_trigger`` boolean marks the moshi turn that emitted ``<ret>``;
the immediately-following ``reference`` turn is what the ARC encoder
would have produced from the retrieval LLM. ``tool`` turns work the
same way: their text is encoded by the ARC encoder and pushed into the
LM's ``streaming_sum`` conditioning queue (both at training time, where
the collator routes ``tool`` turns into ``reference_with_time``, and at
inference time, where the omni layer POSTs the tool result to the ARC
encoder service after a ``[TOOL: ...]`` completes -- exactly mirroring
the ``<ret>`` retrieval path). Tool turns format their text as
``<tool_name>: <result>`` so the encoder sees both what was queried
and what came back. The training loop is responsible for converting
these into the actual token sequences and conditioner inputs.

Usage (requires an OpenAI-compatible endpoint -- vLLM, llama.cpp server,
hosted API, etc.):

.. code-block:: bash

    export LLM_BASE_URL=http://localhost:8000/v1
    export LLM_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    export LLM_API_KEY=anything

    # Visual + RAG augmentation from an SSVD JSONL export
    uv run rag_augment.py augment_visual \\
        --input ssvd_dialogues.jsonl --output augmented.jsonl --num 100

    # Pure text RAG from a list of seed topics
    uv run rag_augment.py generate_text \\
        --topics topics.txt --output text_rag.jsonl --per_topic 3

    # Tool-calling dialogues from a tools spec + topics list
    uv run rag_augment.py generate_tools \\
        --tools example_tools.json --topics tool_topics.txt \\
        --output tools_train.jsonl --per_topic 2
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fire
from openai import OpenAI

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------------


@dataclass
class Llm:
    """Thin wrapper around an OpenAI-compatible chat endpoint."""

    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_NAME", ""))
    api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", "x") or "x"
    )
    temperature: float = 0.7
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise RuntimeError("Set LLM_BASE_URL or pass base_url=...")
        if not self.model:
            raise RuntimeError("Set LLM_MODEL_NAME or pass model=...")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        assert self._client is not None
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


_AUGMENT_PROMPT = """You are augmenting a transcript of a conversation between a user and a chatbot named "moshi", so it can train a retrieval-augmented generation (RAG) model. The conversation may be about a shared image.

Your task: for each turn where moshi states a specific fact that an external knowledge source would be required for (a name, a date, a measurement, a definition, an attributed quote, an obscure detail) -- mark that turn by prepending ``<ret>`` to it, and immediately after that turn, insert a Reference turn with the form ``Reference: <one-line factual statement that would let moshi answer>``. Keep moshi's existing wording.

Do NOT mark turns that are pure perceptual descriptions of the image (those should come from vision, not retrieval). Do NOT mark greetings, opinions, or sentence fragments. Be conservative -- at most one or two retrieval markers per dialogue.

Output the augmented transcript in the same alternating format as the input. Use exactly these labels: ``USER:``, ``MOSHI:``, ``REFERENCE:``. No commentary, no markdown, no surrounding prose.

Input:
{transcript}

Augmented transcript:"""


_GENERATE_PROMPT = """You are writing a short synthetic conversation between a user and a chatbot named "moshi" for training a retrieval-augmented generation (RAG) model.

Topic: {topic}

Write a 4-to-6-turn conversation alternating USER and MOSHI. At least one MOSHI turn must contain a specific fact that would require an external knowledge source. Mark that turn by prepending ``<ret>`` to it, and immediately after that turn insert a Reference turn (``REFERENCE: <one-line factual statement>``) that grounds moshi's answer. Keep the dialogue natural and concise; no commentary, no markdown.

Output the conversation using exactly these labels: ``USER:``, ``MOSHI:``, ``REFERENCE:``. Start with ``USER:``."""


_GENERATE_TOOLS_PROMPT = """You are writing a short synthetic conversation between a user and a chatbot named "moshi" for training a tool-calling model.

The model "moshi" can invoke the following tools by emitting the substring ``[TOOL: name(arg=value, ...)]`` inside its reply. Immediately after the moshi turn that invokes a tool, the tool result appears in a ``TOOL:`` turn. The next moshi turn must reference the tool result conversationally (do not just repeat it verbatim).

Available tools:
{tools_doc}

Topic / scenario: {topic}

Write a 4-to-8-turn conversation. At least one MOSHI turn must contain a tool call. Rules:

* Tool arguments use kwarg syntax with quoted string values: ``[TOOL: get_weather(location="New York")]``. Only use tools listed above; only use argument names listed for that tool.
* After every moshi turn that contains a ``[TOOL: ...]`` call, the very next turn must be a ``TOOL:`` turn formatted exactly as ``TOOL: <tool_name>: <result>`` (the tool name appears in the result for grounding). One short factual line, no markdown, no quotes.
* The moshi turn after a tool result must use the result naturally -- not echo it verbatim.
* Keep the dialogue natural and concise. No commentary, no markdown, no system messages.

Use exactly these labels: ``USER:``, ``MOSHI:``, ``TOOL:``. Start with ``USER:``.

Example of a correctly-formatted tool turn:
``TOOL: get_weather: 68F, partly cloudy with light wind from the east.``"""


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def parse_augmented(text: str) -> list[dict]:
    """Parse an LLM-emitted ``USER:`` / ``MOSHI:`` / ``REFERENCE:`` / ``TOOL:`` transcript.

    Returns a list of ``{"role": ..., "text": ..., "rag_trigger": bool}`` dicts.
    Lines that don't start with a known label are appended to the previous
    turn's text (lets the LLM write multi-line moshi turns). Skips empty input.
    """
    turns: list[dict] = []
    label_map = {
        "USER": "user",
        "MOSHI": "moshi",
        "REFERENCE": "reference",
        "TOOL": "tool",
    }
    current: Optional[dict] = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        # ``<ret>`` may appear either before the label (``<ret>MOSHI: ...``)
        # or inside the body (``MOSHI: <ret>...``). Strip both before parsing.
        line_for_label = line.lstrip()
        rag_prefix = False
        if line_for_label.startswith("<ret>"):
            rag_prefix = True
            line_for_label = line_for_label[len("<ret>") :].lstrip()
        head, _, body = line_for_label.partition(":")
        if head.strip().upper() in label_map and body:
            role = label_map[head.strip().upper()]
            text_part = body.strip()
            rag = rag_prefix
            if role == "moshi" and "<ret>" in text_part:
                rag = True
                text_part = text_part.replace("<ret>", "").strip()
            current = {"role": role, "text": text_part}
            if role == "moshi" and rag:
                current["rag_trigger"] = True
            turns.append(current)
        elif current is not None:
            current["text"] = current["text"] + " " + line.strip()

    return turns


# ----------------------------------------------------------------------------
# Mode 1: visual augmentation
# ----------------------------------------------------------------------------


def _ssvd_to_transcript(turns: list[dict]) -> str:
    """Render a list of role+text turns back into the labeled text format the LLM prompt expects."""
    out: list[str] = []
    label_map = {
        "user": "USER",
        "moshi": "MOSHI",
        "reference": "REFERENCE",
        "tool": "TOOL",
    }
    for t in turns:
        label = label_map.get(t["role"], t["role"].upper())
        out.append(f"{label}: {t['text']}")
    return "\n".join(out)


def augment_visual(
    input: str,
    output: str,
    num: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Augment SSVD visual dialogues with RAG triggers + references.

    :param input: JSONL file with one ``{"image_path": ..., "turns": [...]}``
        per line. The ``turns`` field uses ``{"role": "user"/"moshi", "text": ...}``
        entries -- this matches what an SSVD export produces after the
        existing ``ssvd/generate.py`` pipeline.
    :param output: JSONL destination. Each line is one augmented example.
    :param num: Optional cap on number of dialogues to process.
    :param base_url, model: Override env-derived LLM config.
    """
    llm = Llm(**{k: v for k, v in dict(base_url=base_url, model=model).items() if v})
    n_done = 0
    n_marked = 0

    with open(input) as fin, open(output, "w") as fout:
        for line in fin:
            if num is not None and n_done >= num:
                break
            sample = json.loads(line)
            transcript = _ssvd_to_transcript(sample.get("turns", []))
            try:
                augmented_text = llm.chat(
                    system="You augment dialogue transcripts for RAG training.",
                    user=_AUGMENT_PROMPT.format(transcript=transcript),
                )
            except Exception as e:
                logger.warning("LLM call failed for example %d: %s", n_done, e)
                continue
            new_turns = parse_augmented(augmented_text)
            if not new_turns:
                continue
            out_sample = {
                "image_path": sample.get("image_path"),
                "turns": new_turns,
            }
            fout.write(json.dumps(out_sample) + "\n")
            n_done += 1
            if any(t.get("rag_trigger") for t in new_turns):
                n_marked += 1

    print(
        f"Wrote {n_done} augmented dialogues to {output}; "
        f"{n_marked} ({100 * n_marked / max(n_done, 1):.0f}%) include a RAG trigger."
    )


# ----------------------------------------------------------------------------
# Mode 2: text-only RAG generation
# ----------------------------------------------------------------------------


def generate_text(
    topics: str,
    output: str,
    per_topic: int = 1,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Generate synthetic text-only RAG conversations from a list of seed topics.

    :param topics: Path to a text file with one topic per line.
    :param output: JSONL destination.
    :param per_topic: How many distinct conversations to generate per topic.
    :param base_url, model: Override env-derived LLM config.
    """
    llm = Llm(**{k: v for k, v in dict(base_url=base_url, model=model).items() if v})
    with open(topics) as f:
        seeds = [line.strip() for line in f if line.strip()]
    n_done = 0
    n_marked = 0

    with open(output, "w") as fout:
        for topic in seeds:
            for _ in range(per_topic):
                try:
                    text = llm.chat(
                        system="You write synthetic dialogue for RAG training.",
                        user=_GENERATE_PROMPT.format(topic=topic),
                    )
                except Exception as e:
                    logger.warning("LLM call failed for topic %r: %s", topic, e)
                    continue
                turns = parse_augmented(text)
                if not turns:
                    continue
                fout.write(
                    json.dumps({"image_path": None, "turns": turns}) + "\n"
                )
                n_done += 1
                if any(t.get("rag_trigger") for t in turns):
                    n_marked += 1

    print(
        f"Wrote {n_done} text dialogues to {output}; "
        f"{n_marked} ({100 * n_marked / max(n_done, 1):.0f}%) include a RAG trigger."
    )


# ----------------------------------------------------------------------------
# Mode 3: tool-calling dialogue generation
# ----------------------------------------------------------------------------


def _load_tools_spec(path: str) -> list[dict]:
    """Load and validate a tools spec JSON file.

    Expected format::

        {
          "tools": [
            {
              "name": "get_weather",
              "description": "Get current weather for a location.",
              "args": {"location": "string"},
              "example_output": "72F, sunny, light wind"
            },
            ...
          ]
        }

    Validates that each entry has a ``name`` and ``description``. Other
    fields are passed through to the prompt unchecked.
    """
    with open(path) as f:
        spec = json.load(f)
    tools = spec.get("tools") if isinstance(spec, dict) else spec
    if not isinstance(tools, list) or not tools:
        raise ValueError(
            f"{path}: expected a non-empty list at root or under key 'tools'"
        )
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or "name" not in t or "description" not in t:
            raise ValueError(
                f"{path}: tool entry {i} missing required 'name' or 'description'"
            )
    return tools


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """Render a tools spec into a compact text block for the LLM prompt."""
    lines: list[str] = []
    for t in tools:
        args = t.get("args") or {}
        arg_sig = ", ".join(f"{k}: {v}" for k, v in args.items())
        lines.append(f"- {t['name']}({arg_sig}) -- {t['description']}")
        example = t.get("example_output")
        if example:
            lines.append(f"  Example tool result: {example}")
    return "\n".join(lines)


def generate_tools(
    tools: str,
    topics: str,
    output: str,
    per_topic: int = 1,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Generate synthetic tool-calling dialogues from a tools spec + topics.

    For each topic, the LLM is asked to pick one or more tools from the
    spec and write a multi-turn conversation in which moshi emits
    ``[TOOL: name(arg=value)]`` and the simulated tool output appears as
    a ``TOOL:`` turn. The follow-up moshi turn references the tool result
    so the model learns to consume tool output.

    :param tools: Path to a JSON tools spec (see :func:`_load_tools_spec`
        for the expected shape). A starter ``ssvd/example_tools.json``
        is shipped with the repo.
    :param topics: Path to a text file with one topic per line. Topics
        give the LLM a scenario hook -- e.g. ``"user wants the weather
        before a hike"``.
    :param output: JSONL destination.
    :param per_topic: How many distinct conversations to generate per topic.
    :param base_url, model: Override env-derived LLM config.
    """
    llm = Llm(**{k: v for k, v in dict(base_url=base_url, model=model).items() if v})
    tool_specs = _load_tools_spec(tools)
    tools_doc = _format_tools_for_prompt(tool_specs)
    with open(topics) as f:
        seeds = [line.strip() for line in f if line.strip()]
    n_done = 0
    n_with_tool = 0

    with open(output, "w") as fout:
        for topic in seeds:
            for _ in range(per_topic):
                try:
                    text = llm.chat(
                        system="You write synthetic dialogue for tool-calling training.",
                        user=_GENERATE_TOOLS_PROMPT.format(
                            tools_doc=tools_doc, topic=topic
                        ),
                    )
                except Exception as e:
                    logger.warning("LLM call failed for topic %r: %s", topic, e)
                    continue
                turns = parse_augmented(text)
                if not turns:
                    continue
                # A dialogue without any tool-call moshi turn is useless
                # for tool training -- drop it rather than dilute the
                # signal with examples that never exercise the pathway.
                has_tool_call = any(
                    t["role"] == "moshi" and "[TOOL:" in t["text"] for t in turns
                )
                if not has_tool_call:
                    logger.warning(
                        "LLM emitted no tool call for topic %r; dropping", topic
                    )
                    continue
                fout.write(
                    json.dumps({"image_path": None, "turns": turns}) + "\n"
                )
                n_done += 1
                n_with_tool += 1

    print(
        f"Wrote {n_done} tool-calling dialogues to {output}; "
        f"{n_with_tool} ({100 * n_with_tool / max(n_done, 1):.0f}%) "
        f"include a tool call."
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fire.Fire(
        {
            "augment_visual": augment_visual,
            "generate_text": generate_text,
            "generate_tools": generate_tools,
        }
    )


if __name__ == "__main__":
    main()
