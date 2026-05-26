"""Collator: turn a batch of :class:`RagExample` into model input tensors.

This is the bridge between the structured JSONL records (text turns,
optional image path, optional reference markers) and the dense tensors
:meth:`MoshiVis.forward_text` expects: ``input_ids`` of shape
``[B, num_codebooks, T]`` plus ``cross_attention_src``, plus the
:class:`ConditionAttributes` list for the conditioner provider.

Audio handling -- read carefully
================================

Training a speech model needs **Mimi-encoded audio codes** for every
turn. The SSVD-augmented JSONL emitted by ``ssvd/rag_augment.py`` is
text-only -- it does not include audio. There are two ways to plug
audio in here:

1. **Pre-process** the dataset once: synthesize TTS audio for each
   turn, Mimi-encode it, save the codes alongside the JSONL as
   companion ``.pt`` files keyed by example index. This collator
   reads those files via the ``audio_codes_dir`` argument. Recommended
   for any non-toy training run.
2. **Smoke-test mode** (``audio_codes_dir=None``): the collator fills
   audio slots with zeros. The trainer runs end-to-end, gradients
   flow, but the model isn't learning meaningful audio behavior --
   only the text codebook gets a meaningful signal. Useful exclusively
   for sanity-checking that the training pipeline executes.

A complete audio preprocessing script is out of scope for this commit
(it requires a TTS choice -- Coqui XTTS, Bark, internal Kyutai TTS,
etc. -- which is deployment-specific). See ``training/README.md`` for
the recommended preprocessing recipe.

Image handling
==============

Images are encoded via ``ImageProjection`` once at collation time. For
larger datasets, pre-encode images and load the cross-attention KV
tensors directly to avoid re-running PaliGemma per epoch. Both paths
are supported -- pass ``precomputed_image_kv_dir`` to use cached KVs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from kyuteye.conditioners import ConditionAttributes, TensorCondition
from kyuteye.training.dataset import RagExample

if TYPE_CHECKING:
    import sentencepiece


# Callable signature: ``(image_path: str) -> torch.Tensor`` returning a
# ``[T_img, dim]`` cross-attention KV (or ``[B=1, T_img, dim]`` -- the
# collator strips the leading batch dim if present). Used for live
# image encoding when the user opts out of pre-computing.
ImageEncodeFn = "Optional[Any]"  # callable[[str], torch.Tensor]

logger = logging.getLogger(__name__)


def make_live_image_encoder(
    moshi_vis_gen: Any,
    image_proj: Any,
    *,
    image_size: int = 448,
) -> "Any":
    """Build a callable suitable for ``RagDataCollator(image_encode_fn=...)``.

    Closes over the image-encoder model + the model's
    ``precompte_ca_kv`` projection so each call goes from an image file
    on disk to a ``[T_img, dim]`` cross-attention KV tensor ready to
    drop into ``input_ids`` cross-attention.

    Important: the returned callable runs the encoder inline -- on
    whichever device ``image_proj`` is on. If the image encoder lives
    on GPU, the call from a DataLoader worker process will fail
    (workers are CPU-only by default). For the live-encoding path,
    instantiate the ``ImageProjection`` on CPU before training (and
    accept the slowdown), OR use ``num_workers=0`` so the encoder
    call happens in the main process where the GPU is available.

    Pre-computing to ``precomputed_image_kv_dir`` is the recommended
    path for any non-toy dataset; this helper exists for completeness
    and quick prototyping.

    :param moshi_vis_gen: A :class:`MoshiVisGen` (has the
        ``precompte_ca_kv`` cross-attention KV projection).
    :param image_proj: A :class:`ImageProjection` -- the image
        backbone + projection from the model loader.
    :param image_size: Resolution the image is resized to before
        encoding. Matches ``image_size`` in the YAML config.
    """
    # Lazy imports to keep this module's load cheap.
    from torchvision.io import ImageReadMode, decode_image  # type: ignore[import-untyped]

    from kyuteye.modules.image_transforms import get_minimal_transforms

    transforms = get_minimal_transforms(image_size)

    def encode(path: str) -> torch.Tensor:
        with open(path, "rb") as f:
            raw = f.read()
        img = decode_image(
            torch.frombuffer(raw, dtype=torch.uint8), mode=ImageReadMode.RGB
        )
        img = transforms(img)
        img = image_proj.to_tensor_and_normalize(img)
        # Match ``server.extract_image``: pixtral encoder expects a list,
        # others expect ``[B=1, C, H, W]``.
        from kyuteye.config.enums import ImageEncoder

        if image_proj.encoder_type == ImageEncoder.PIXTRAL:
            img = [img]
        else:
            img = img[None, ...]
        with torch.no_grad():
            ca_src = image_proj(img)["cross_attention_src"]
            k, v = moshi_vis_gen.precompte_ca_kv(ca_src)
        # Return as (K, V) tuple stacked into a single tensor for the
        # collator's storage. The trainer/model later splits it back.
        # ``[2, T_img, dim]`` packs both halves; the collator's image
        # stacking handles either shape transparently.
        return torch.stack([k.squeeze(0), v.squeeze(0)], dim=0)

    return encode


@dataclass
class CollatedBatch:  # noqa: D101
    """One training batch ready for ``MoshiVis.forward_text``.

    :param input_ids: ``[B, num_codebooks, T]`` long tensor -- text codebook
        in channel 0, audio output codebooks in 1..dep_q, user audio
        input codebooks in dep_q+1..num_codebooks-1.
    :param target_text: ``[B, T]`` long tensor of text codebook targets
        (== ``input_ids[:, 0]`` shifted by 1 for next-token prediction).
    :param loss_mask: ``[B, T]`` bool tensor. Loss is computed only at
        positions where ``loss_mask`` is True -- the trainer masks out
        user turns, padding, reference turns.
    :param cross_attention_src: Optional ``[B, T_img, dim]`` image
        cross-attention input, or ``None`` for text-only examples.
    :param condition_attributes: ``[ConditionAttributes]`` of length B,
        carrying ``first_speaker`` (LUT) and ``reference_with_time``
        (TensorCondition for ARC encoder).
    """

    input_ids: torch.Tensor
    target_text: torch.Tensor
    loss_mask: torch.Tensor
    cross_attention_src: Optional[torch.Tensor]
    condition_attributes: list[ConditionAttributes] = field(default_factory=list)

    def to(self, device: torch.device) -> "CollatedBatch":
        """Move tensor fields to ``device``. ``condition_attributes`` are
        left CPU-side since the conditioner provider handles its own
        device-placement at ``prepare``-time.
        """
        return CollatedBatch(
            input_ids=self.input_ids.to(device),
            target_text=self.target_text.to(device),
            loss_mask=self.loss_mask.to(device),
            cross_attention_src=(
                self.cross_attention_src.to(device)
                if self.cross_attention_src is not None
                else None
            ),
            condition_attributes=self.condition_attributes,
        )


class RagDataCollator:
    """Stateful collator that consumes :class:`RagExample` batches.

    The collator is initialized once per training process with a tokenizer
    and the model's static config (num_codebooks, audio_offset, etc.) and
    is called per batch by the DataLoader.

    :param tokenizer: SentencePiece processor (the same one the model
        was trained with).
    :param num_codebooks: Total codebooks in the model input
        (text + audio in + audio out). Matches ``MoshiVis.num_codebooks``.
    :param audio_offset: Index where audio codebooks start in
        ``input_ids``. Typically ``1`` (text in channel 0).
    :param max_seq_len: Truncate sequences longer than this. ``None``
        disables truncation (rarely a good idea for training).
    :param audio_codes_dir: Path to per-example pre-encoded audio
        codes (``.pt`` files indexed ``{example_idx}.pt``). When
        ``None``, audio slots are filled with zeros (smoke-test mode).
    :param image_dir: Root directory the JSONL ``image_path`` entries
        are relative to. ``None`` means treat ``image_path`` as
        absolute paths.
    :param precomputed_image_kv_dir: When set, look up
        ``{image_path_stem}.pt`` here for pre-computed cross-attention
        KV instead of running the image encoder. Fast path.
    :param image_encode_fn: Optional callable ``(image_path: str) ->
        Tensor`` that returns the cross-attention KV for that image
        live. Slower than the precomputed path but doesn't require a
        preprocessing pass. Use :func:`make_live_image_encoder` to
        build one from a ``MoshiVisGen`` and ``ImageProjection``. The
        callable runs inside the DataLoader worker (CPU side); if you
        need GPU encoding, pre-compute instead.
    :param ungenerated_token_id: Fill value for positions outside the
        sequence (matches ``MoshiVis.ungenerated_token_id`` so the
        model recognises it as "not yet predicted"). Defaults to -2.
    :param text_padding_token_id: SentencePiece pad id. Defaults to 3
        which matches MoshiVis's default.
    :param first_speaker_default: Default speaker label written to the
        ``first_speaker`` condition. ``"SPEAKER_MAIN"`` matches MoshiRAG.

    Image config rule: if any example in the dataset has ``image_path``
    set, **exactly one** of ``precomputed_image_kv_dir`` or
    ``image_encode_fn`` must also be set. Otherwise the collator raises
    a ``RuntimeError`` the first time it sees an image-bearing example
    -- a deliberate fail-loud to catch silent-vision-loss bugs.
    """

    def __init__(
        self,
        tokenizer: "sentencepiece.SentencePieceProcessor",
        *,
        num_codebooks: int,
        audio_offset: int = 1,
        max_seq_len: Optional[int] = None,
        audio_codes_dir: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        precomputed_image_kv_dir: Optional[str | Path] = None,
        image_encode_fn: ImageEncodeFn = None,
        ungenerated_token_id: int = -2,
        text_padding_token_id: int = 3,
        first_speaker_default: str = "SPEAKER_MAIN",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.audio_offset = audio_offset
        self.max_seq_len = max_seq_len
        self.audio_codes_dir = Path(audio_codes_dir) if audio_codes_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.precomputed_image_kv_dir = (
            Path(precomputed_image_kv_dir) if precomputed_image_kv_dir else None
        )
        self.image_encode_fn = image_encode_fn
        self.ungenerated_token_id = ungenerated_token_id
        self.text_padding_token_id = text_padding_token_id
        self.first_speaker_default = first_speaker_default

        if self.audio_codes_dir is None:
            logger.warning(
                "[collator] audio_codes_dir not set -- audio inputs will be "
                "zero-filled. This is smoke-test mode; gradient flows through "
                "the model but no meaningful audio behaviour is learned. See "
                "training/README.md for the audio preprocessing recipe."
            )

    # ------------------------------------------------------------------ helpers

    def _tokenize_turn(
        self, text: str, *, is_moshi: bool
    ) -> tuple[list[int], list[bool]]:
        """Tokenize one turn. Returns ``(token_ids, loss_mask)``.

        Only moshi turns contribute to the next-token CE loss; user and
        reference turns get ``loss_mask=False`` for every position.
        """
        ids = list(self.tokenizer.encode(text))  # type: ignore[no-untyped-call]
        mask = [is_moshi] * len(ids)
        return ids, mask

    def _load_audio_codes(
        self, example_idx: int, seq_len: int
    ) -> torch.Tensor:
        """Load (or zero-fill) audio codes for one example.

        Returns ``[num_audio_codebooks, seq_len]``. ``num_audio_codebooks``
        is ``self.num_codebooks - self.audio_offset``.
        """
        n_audio = self.num_codebooks - self.audio_offset
        if self.audio_codes_dir is None:
            return torch.zeros(n_audio, seq_len, dtype=torch.long)
        path = self.audio_codes_dir / f"{example_idx}.pt"
        if not path.exists():
            logger.warning(
                "[collator] audio codes missing for example %d (%s); "
                "zero-filling",
                example_idx,
                path,
            )
            return torch.zeros(n_audio, seq_len, dtype=torch.long)
        codes = torch.load(path, map_location="cpu", weights_only=True)
        # Coerce dtype: Mimi (depending on version) sometimes emits
        # ``torch.int32`` or ``torch.uint16`` codes. ``input_ids``
        # downstream is ``torch.long``; cast here so the cat downstream
        # doesn't silently widen + slow + produce shape surprises.
        if codes.dtype != torch.long:
            codes = codes.long()
        assert codes.dim() == 2 and codes.shape[0] == n_audio, (
            f"expected audio codes of shape [{n_audio}, T] (n_audio derived "
            f"from num_codebooks={self.num_codebooks} - audio_offset={self.audio_offset}); "
            f"got {tuple(codes.shape)} from {path}. Most likely cause: the "
            f"Mimi codec used at preprocessing emits a different number of "
            f"codebooks than the LM expects -- check the preprocessing "
            f"recipe vs the model config."
        )
        if codes.shape[1] < seq_len:
            pad = torch.zeros(n_audio, seq_len - codes.shape[1], dtype=torch.long)
            codes = torch.cat([codes, pad], dim=1)
        elif codes.shape[1] > seq_len:
            codes = codes[:, :seq_len]
        return codes

    def _build_condition_attrs(
        self, example: RagExample
    ) -> ConditionAttributes:
        """Build the per-example ConditionAttributes for the fuser.

        * ``first_speaker``: LUT conditioner -- one of ``SPEAKER_MAIN`` /
          ``SPEAKER_OTHER``. We always pass ``SPEAKER_MAIN`` here; the
          trainer can override per-example if it has speaker labels.
        * ``reference_with_time``: TensorCondition placeholder. At
          training time we don't pre-encode references here; the
          conditioner's own ``prepare`` is called by the
          :class:`ConditionProvider` on the raw text in
          ``ConditionAttributes.text``. We surface the reference text
          via the ``text`` dict (key ``reference_with_time``) so a
          text-based conditioner sees it; for ARC encoder use, this
          collator is paired with a tensor-typed conditioner that
          accepts raw strings via its own prepare hook.

        For training a CFG model: the trainer optionally calls
        ``dropout_all_conditions`` on the returned list to build the
        null branch. That's a per-step random decision, not per-example,
        so we don't do it here.
        """
        # Concatenate all reference turns into one string (matches the
        # MoshiRAG convention of "Reference: <ref>" appearing once per
        # retrieval event in the conversation).
        references = " ".join(
            t.text for t in example.turns if t.role == "reference"
        )
        text_conditions: dict[str, Optional[str]] = {
            "first_speaker": self.first_speaker_default,
            "reference_with_time": references or "",
        }
        return ConditionAttributes(text=text_conditions, tensor={})

    def _encode_image(
        self, example: RagExample
    ) -> Optional[torch.Tensor]:
        """Return the image cross-attention KV for ``example``.

        Three cases, ordered by precedence:

        1. No image: ``image_path is None`` -> return ``None``.
        2. Precomputed: ``precomputed_image_kv_dir/{stem}.pt`` exists
           -> load and return. The fast, per-epoch-cached path.
        3. Live encode: ``image_encode_fn`` is set -> call it. Slower
           but doesn't require a preprocessing pass.

        If the example has an image but neither path is available,
        raise ``RuntimeError`` -- a deliberate fail-loud so vision
        conditioning doesn't get silently dropped.
        """
        if not example.has_image():
            return None
        img_path = example.image_path  # type: ignore[assignment]

        # 1. Precomputed KV: fast path.
        if self.precomputed_image_kv_dir is not None:
            stem = Path(img_path).stem
            kv_path = self.precomputed_image_kv_dir / f"{stem}.pt"
            if kv_path.exists():
                tensor = torch.load(kv_path, map_location="cpu", weights_only=True)
                # If saved as ``[1, T_img, dim]``, drop the batch dim --
                # the caller stacks to a fresh batch.
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                return tensor
            logger.warning(
                "[collator] precomputed image KV missing for %s (looked at %s); "
                "will try image_encode_fn fallback if configured",
                img_path,
                kv_path,
            )

        # 2. Live encode.
        if self.image_encode_fn is not None:
            full_path = img_path
            if self.image_dir is not None:
                full_path = str(Path(self.image_dir) / img_path)
            try:
                tensor = self.image_encode_fn(full_path)
            except Exception as e:
                logger.warning(
                    "[collator] live image_encode_fn failed for %s: %s -- "
                    "vision dropped for this example",
                    img_path,
                    e,
                )
                return None
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            return tensor

        # 3. Neither path configured but the example has an image.
        # Fail loud rather than silently lose vision conditioning.
        raise RuntimeError(
            f"RagDataCollator received an example with image_path={img_path!r} "
            f"but neither ``precomputed_image_kv_dir`` nor ``image_encode_fn`` "
            f"is configured. Either pre-encode images (see "
            f"``training/README.md``) or pass an ``image_encode_fn`` (see "
            f"``make_live_image_encoder``). Silently dropping vision was the "
            f"pre-fix behaviour and produced misleading 'training' that "
            f"didn't actually exercise the vision pathway."
        )

    # ------------------------------------------------------------------ entry

    def __call__(self, batch: list[RagExample]) -> CollatedBatch:
        """Build a :class:`CollatedBatch` from a list of examples."""
        B = len(batch)

        # 1. Tokenize and concatenate moshi/user turns into a single text
        #    stream per example. Reference turns are excluded from the
        #    inline stream -- they live in the ConditionAttributes.
        text_ids_per_example: list[list[int]] = []
        loss_mask_per_example: list[list[bool]] = []
        for ex in batch:
            ids: list[int] = []
            mask: list[bool] = []
            for turn in ex.turns:
                if turn.role == "reference":
                    continue
                turn_ids, turn_mask = self._tokenize_turn(
                    turn.text, is_moshi=(turn.role == "moshi")
                )
                ids.extend(turn_ids)
                mask.extend(turn_mask)
            text_ids_per_example.append(ids)
            loss_mask_per_example.append(mask)

        seq_lens = [len(ids) for ids in text_ids_per_example]
        max_len = max(seq_lens) if seq_lens else 0
        if self.max_seq_len is not None and max_len > self.max_seq_len:
            max_len = self.max_seq_len
            text_ids_per_example = [ids[:max_len] for ids in text_ids_per_example]
            loss_mask_per_example = [m[:max_len] for m in loss_mask_per_example]

        # 2. Pad text to max_len and build the text codebook + loss mask.
        text_codebook = torch.full(
            (B, max_len), self.text_padding_token_id, dtype=torch.long
        )
        loss_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for b, (ids, mask) in enumerate(
            zip(text_ids_per_example, loss_mask_per_example)
        ):
            text_codebook[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            loss_mask[b, : len(mask)] = torch.tensor(mask, dtype=torch.bool)

        # 3. Audio codes (one row per audio codebook, per example).
        # Use each example's stable dataset-global ``index`` so audio
        # lookup survives DataLoader shuffling. ``RagJsonlDataset``
        # populates this when loading; ``in_batch_position`` is a
        # last-resort fallback that surfaces the bug loudly via a
        # warning rather than silently zero-filling.
        n_audio = self.num_codebooks - self.audio_offset
        audio_codebooks = torch.zeros(B, n_audio, max_len, dtype=torch.long)
        for b, ex in enumerate(batch):
            if ex.index is None:
                if self.audio_codes_dir is not None:
                    logger.warning(
                        "[collator] example has no stable index but "
                        "audio_codes_dir is set; falling back to in-batch "
                        "position %d -- this will load WRONG audio after "
                        "DataLoader shuffling. Set ex.index via "
                        "RagJsonlDataset, which assigns indices on load.",
                        b,
                    )
                example_idx = b
            else:
                example_idx = ex.index
            codes = self._load_audio_codes(example_idx, max_len)
            audio_codebooks[b] = codes

        # 4. Stack into [B, num_codebooks, T] with text in channel 0.
        input_ids = torch.cat(
            [text_codebook.unsqueeze(1), audio_codebooks], dim=1
        )
        assert input_ids.shape == (B, self.num_codebooks, max_len), input_ids.shape

        # 5. Targets are the next-token shift of the text codebook. The
        #    trainer's loss compares logits[t] against target[t+1]. For
        #    simplicity here, we return targets unshifted; the loss
        #    function handles the shift via slicing.
        target_text = text_codebook.clone()

        # 6. Image cross-attention source (or None for text-only).
        image_kvs = [self._encode_image(ex) for ex in batch]
        if any(kv is not None for kv in image_kvs):
            # Stack only the present ones; pad the rest with zeros to a
            # common shape (assumes equal image resolution -- standard).
            present = [kv for kv in image_kvs if kv is not None]
            ref_shape = present[0].shape  # [T_img, dim]
            ca_src = torch.zeros(B, *ref_shape, dtype=present[0].dtype)
            for b, kv in enumerate(image_kvs):
                if kv is not None:
                    ca_src[b] = kv
            cross_attention_src: Optional[torch.Tensor] = ca_src
        else:
            cross_attention_src = None

        # 7. ConditionAttributes per example for the fuser.
        condition_attrs = [self._build_condition_attrs(ex) for ex in batch]

        return CollatedBatch(
            input_ids=input_ids,
            target_text=target_text,
            loss_mask=loss_mask,
            cross_attention_src=cross_attention_src,
            condition_attributes=condition_attrs,
        )
