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

Two modes:

1. ``augment_visual`` -- take an existing SSVD-generated visual dialogue
   (or any conversation indexed by image path) and use an LLM to insert
   ``<ret>`` markers before factual claims, then synthesize the reference
   document for each marker. The image is preserved so the resulting
   example trains both the vision pathway and the RAG pathway.

2. ``generate_text`` -- given a seed topic, ask an LLM to (a) generate a
   conversation about it, (b) identify the factual turns, (c) write a
   reference document for each. No image. Useful for getting text-only RAG
   capability without diluting the visual training signal.

Output is JSONL, one example per line. Each example has:

    {
      "image_path": "path/to/image.png" | null,
      "turns": [
        {"role": "user", "text": "..."},
        {"role": "moshi", "text": "I will check ...", "rag_trigger": true},
        {"role": "reference", "text": "Reference: ..."},
        {"role": "moshi", "text": "..."},
        ...
      ]
    }

The ``rag_trigger`` boolean marks the moshi turn that emitted ``<ret>``;
the immediately-following ``reference`` turn is what the ARC encoder would
have produced from the retrieval LLM. The training loop is responsible for
converting these into the actual token sequences and conditioner inputs.

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


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def parse_augmented(text: str) -> list[dict]:
    """Parse an LLM-emitted ``USER:`` / ``MOSHI:`` / ``REFERENCE:`` transcript.

    Returns a list of ``{"role": ..., "text": ..., "rag_trigger": bool}`` dicts.
    Lines that don't start with a known label are appended to the previous
    turn's text (lets the LLM write multi-line moshi turns). Skips empty input.
    """
    turns: list[dict] = []
    label_map = {"USER": "user", "MOSHI": "moshi", "REFERENCE": "reference"}
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
    label_map = {"user": "USER", "moshi": "MOSHI", "reference": "REFERENCE"}
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
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fire.Fire({"augment_visual": augment_visual, "generate_text": generate_text})


if __name__ == "__main__":
    main()
