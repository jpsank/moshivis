"""Dataset for combined MoshiVis+RAG training.

Consumes the JSONL produced by ``ssvd/rag_augment.py`` and yields
training examples a future trainer can collate into batches. The
on-disk schema is one JSON object per line:

.. code-block:: json

   {
     "image_path": "path/to/image.png" | null,
     "turns": [
       {"role": "user", "text": "..."},
       {"role": "moshi", "text": "...", "rag_trigger": true},
       {"role": "reference", "text": "Reference: ..."},
       {"role": "moshi", "text": "..."}
     ]
   }

The dataset itself doesn't load audio (Mimi codec invocation lives in
the trainer for batch efficiency) and doesn't compute losses; it just
exposes the structured records. A trainer using these examples would:

1. Tokenize the moshi turns with SentencePiece -> text codebook targets.
2. Synthesize or load audio for each turn -> Mimi-encoded audio codes.
3. Embed the image via PaliGemma2 -> ``cross_attention_src`` per sample.
4. Tokenize the reference turns with the ARC encoder tokenizer ->
   ``ConditionAttributes`` with ``reference_with_time`` populated and
   ``first_speaker`` set per turn.
5. Apply ``dropout_all_conditions`` randomly for CFG training.
6. Forward through ``MoshiVis.forward_text`` and compute next-token CE
   loss with ``kyuteye.training.loss.next_token_ce_loss``.

This module only handles (the structural part of) step (1)-(4)'s input;
the heavy lifting (audio synthesis, batching) is the trainer's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is a hard dep elsewhere
    Dataset = object  # type: ignore[misc,assignment]


@dataclass
class RagTurn:
    """One turn of a dialogue.

    Roles:

    * ``user`` -- the human side. Text appears in the model's context;
      the model is not trained to predict it (``loss_mask=False``).
    * ``moshi`` -- the model's side. Trained as next-token CE target.
    * ``reference`` -- factual reference for the immediately-preceding
      ``<ret>`` moshi turn. Skipped from the inline text stream and fed
      to the ARC encoder via the ``ConditionAttributes``.
    * ``tool`` -- simulated tool result, emitted right after a moshi
      turn that contains a ``[TOOL: name(args)]`` call. Treated like a
      user turn by the collator: included in the model's context with
      ``loss_mask=False`` so the model learns to consume tool output
      without being trained to generate the tool result text itself.
    """

    role: str
    text: str
    rag_trigger: bool = False  # True only on the moshi turn that emitted <ret>


@dataclass
class RagExample:
    """One training example.

    :param image_path: Path to the conditioning image, or ``None`` for
        text-only RAG dialogues.
    :param turns: Ordered list of turns. The trainer is responsible for
        flattening into the next-token prediction targets.
    :param index: Stable dataset-global index, populated by
        :class:`RagJsonlDataset` when the file is loaded. Used by the
        collator to look up per-example pre-encoded audio codes by
        deterministic name (``{index}.pt``) instead of the in-batch
        position (which is wrong after shuffling -- the original bug
        this field fixes).
    """

    image_path: Optional[str]
    turns: list[RagTurn]
    index: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "RagExample":
        return cls(
            image_path=raw.get("image_path"),
            turns=[
                RagTurn(
                    role=str(t["role"]),
                    text=str(t.get("text", "")),
                    rag_trigger=bool(t.get("rag_trigger", False)),
                )
                for t in raw.get("turns", [])
            ],
            index=raw.get("index"),
        )

    def moshi_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "moshi")

    def has_rag(self) -> bool:
        return any(t.rag_trigger for t in self.turns)

    def has_tool(self) -> bool:
        """True iff any moshi turn contains a ``[TOOL: ...]`` call.

        A tool call is detected by substring -- no explicit flag is set
        by the data generator since the inline marker is unambiguous.
        Used by :meth:`RagJsonlDataset.tool_fraction` for the trainer
        startup log.
        """
        return any(
            t.role == "moshi" and "[TOOL:" in t.text for t in self.turns
        )

    def has_image(self) -> bool:
        return self.image_path is not None


class RagJsonlDataset(Dataset):  # type: ignore[misc]
    """``torch.utils.data.Dataset`` over a JSONL training-data file.

    Holds the full dataset in memory (the SSVD outputs are small enough --
    on the order of MB for tens of thousands of dialogues). For larger
    corpora swap in an indexed-on-disk variant; the public ``__getitem__``
    contract stays the same.

    :param path: Path to the JSONL file (or list of paths -- concatenated).
    :param filter_no_rag: When ``True`` (default ``False``), skip
        examples that don't include a ``<ret>`` trigger -- useful for
        ablations that train only on the RAG distribution.
    :param require_image: When ``True``, skip examples without an image.
        Use for a vision-only fine-tune phase.
    """

    def __init__(
        self,
        path: str | Path | Iterable[str | Path],
        *,
        filter_no_rag: bool = False,
        require_image: bool = False,
    ) -> None:
        paths: list[Path]
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        examples: list[RagExample] = []
        # Global counter across all files so audio-code lookup
        # (``{index}.pt``) remains stable. We assign indices to ALL
        # records, even filtered ones, so callers running with and
        # without ``filter_no_rag`` see the same index for the same
        # JSONL line -- audio pre-encoding done once can be reused.
        global_idx = 0
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = RagExample.from_dict(json.loads(line))
                    # Honour any explicit "index" field in the JSONL (rare;
                    # mostly for resumed preprocessing). Default: the
                    # global running counter.
                    if ex.index is None:
                        ex.index = global_idx
                    global_idx += 1
                    if filter_no_rag and not ex.has_rag():
                        continue
                    if require_image and not ex.has_image():
                        continue
                    examples.append(ex)
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RagExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[RagExample]:
        return iter(self.examples)

    # Convenience aggregates -- useful for the trainer's startup log.

    def rag_fraction(self) -> float:
        if not self.examples:
            return 0.0
        return sum(1 for e in self.examples if e.has_rag()) / len(self.examples)

    def tool_fraction(self) -> float:
        if not self.examples:
            return 0.0
        return sum(1 for e in self.examples if e.has_tool()) / len(self.examples)

    def visual_fraction(self) -> float:
        if not self.examples:
            return 0.0
        return sum(1 for e in self.examples if e.has_image()) / len(self.examples)
