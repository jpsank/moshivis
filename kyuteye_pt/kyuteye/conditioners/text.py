"""Lightweight text conditioners.

Ported and trimmed from kyutai-labs/moshi-rag
(moshi/moshi/conditioners/text.py). The MoshiRAG file ships three
conditioners: :class:`LUTConditioner` (small hash + LUT), :class:`T5Conditioner`,
and :class:`MultiT5Conditioner`. Only ``LUTConditioner`` is kept here -- it is
the conditioner MoshiRAG uses for the ``first_speaker`` attribute and it has
no heavy dependencies. The T5 path is intentionally omitted: MoshiVis does
not need it at inference, and pulling in T5 / spacy adds non-trivial
dependencies to the PyTorch backend.

If a future MoshiVis+RAG fine-tune needs the T5 conditioner, restore it by
copying ``T5Conditioner`` from moshi-rag's source.
"""

from __future__ import annotations

import hashlib
import typing as tp

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from kyuteye.conditioners.base import (
    ConditionType,
    TokenizedText,
    _BaseTextConditioner,
)


def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    """Turn a 1-d length tensor into a [B, max_len] bool mask."""
    assert lengths.dim() == 1
    final_length = int(lengths.max().item()) if not max_len else max_len
    final_length = max(final_length, 1)
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    """Hash a word into an integer slot < ``vocab_size``."""
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % vocab_size


class TextConditioner(_BaseTextConditioner[TokenizedText]):
    """Marker subclass; kept so isinstance checks match moshi-rag."""


class Tokenizer:
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        raise NotImplementedError()


class NoopTokenizer(Tokenizer):
    """One-token-per-string tokenizer for fixed-vocabulary attributes (e.g. ``first_speaker``)."""

    def __init__(self, n_bins: int, possible_values: list[str] | None = None):
        self.n_bins = n_bins
        self.pad_idx = n_bins
        if possible_values is None:
            self.possible_values: tp.Optional[tp.Dict[str, int]] = None
        else:
            self.possible_values = {v: i for i, v in enumerate(possible_values)}
            assert n_bins >= len(possible_values)

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        output: list[int] = []
        lengths: list[int] = []
        for text in texts:
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                if self.possible_values is None:
                    output.append(hash_trick(text, self.n_bins))
                else:
                    if text not in self.possible_values:
                        raise ValueError(
                            f"'{text}' is not in possible_values {self.possible_values}"
                        )
                    output.append(self.possible_values[text])
                lengths.append(1)
        tokens = torch.tensor(output).int()[:, None]
        mask = length_to_mask(torch.tensor(lengths))
        return TokenizedText(tokens, mask)


class WhiteSpaceTokenizer(Tokenizer):
    """Whitespace-split + per-word hash trick. Kept for state-dict compatibility."""

    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int):
        self.n_bins = n_bins
        self.pad_idx = n_bins

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        output: list[torch.Tensor] = []
        lengths: list[int] = []
        for text in texts:
            if text is None:
                output.append(torch.tensor([self.pad_idx]))
                lengths.append(0)
                continue
            words = [w for w in text.split() if w not in self.PUNCTUATION]
            lengths.append(len(words))
            output.append(torch.tensor([hash_trick(w, self.n_bins) for w in words]))
        mask = length_to_mask(torch.tensor(lengths))
        padded = pad_sequence(output, padding_value=self.pad_idx, batch_first=True).int()
        return TokenizedText(padded, mask)


class LUTConditioner(TextConditioner):
    """Hash-then-embedding conditioner.

    Args:
        n_bins: Vocabulary size (hash modulo this).
        tokenizer: Either ``"noop"`` (one token per string) or ``"whitespace"``.
        possible_values: For ``noop`` mode, list of expected strings (used to
            pin specific tokens to specific slots).
        init_scale: Scale the embedding init by this factor.
    """

    def __init__(
        self,
        n_bins: int,
        tokenizer: str = "noop",
        possible_values: list[str] | None = None,
        init_scale: float = 1.0,
        **kwargs: tp.Any,
    ):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(n_bins + 1, self.dim)
        self.embed.weight.data *= init_scale
        if tokenizer == "noop":
            self.tokenizer: Tokenizer = NoopTokenizer(n_bins, possible_values)
        elif tokenizer == "whitespace":
            self.tokenizer = WhiteSpaceTokenizer(n_bins)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        return TokenizedText(tokens.to(device), mask.to(device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        return ConditionType(self.embed(tokens), mask)
