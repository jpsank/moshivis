"""Audio preprocessing for combined MoshiVis + MoshiRAG training.

Bridges the gap between text-only JSONL (what ``ssvd/rag_augment.py``
emits) and the per-example Mimi-encoded audio codes the trainer's
collator expects on disk.

Pipeline per example:

1. Walk the dialogue turns in order.
2. For each turn, synthesize speech via a pluggable :class:`BaseTTS` --
   use a different speaker embedding for ``user`` vs ``moshi`` turns so
   the model sees two distinct voices.
3. Concatenate the synthesized waveforms with short silence gaps so the
   turn boundaries are audible. Track the per-channel layout:
   ``moshi`` audio goes into the "model output" stream; ``user`` audio
   into the "other speaker" stream. (Reference turns produce no audio.)
4. Encode each stream through Mimi.
5. Stack into ``[n_audio_codebooks, T]`` and save as ``{idx}.pt`` in
   ``output_dir``.

The TTS choice is intentionally pluggable. We ship :class:`SilenceTTS`
(returns zero audio of the right shape -- useful for end-to-end
pipeline validation without a real TTS dep) and :class:`CoquiXTTS`
(driver for Coqui's ``TTS`` package -- runs if you ``pip install TTS``).
Most production deployments will write their own ``BaseTTS`` subclass
wrapping whatever TTS they prefer (Bark, internal Kyutai, StyleTTS2,
your in-house model).

Memory note: the preprocessing job runs Mimi (and the TTS) on a GPU
but the model itself is not loaded -- this fits comfortably on a
single A100 40GB or even a smaller GPU.
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from kyuteye.training.dataset import RagExample, RagJsonlDataset

if TYPE_CHECKING:
    from moshi.models import MimiModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# TTS interface
# ----------------------------------------------------------------------------


class BaseTTS(abc.ABC):
    """Abstract TTS backend.

    Subclasses synthesize a single utterance into a 1-D float32 PCM
    tensor at the codec's sample rate (the preprocessor resamples if
    needed). The two speaker identities are passed as opaque strings;
    a concrete TTS can map them to specific speaker embeddings or
    voices.
    """

    sample_rate: int  # Hz, set by subclass

    @abc.abstractmethod
    def synthesize(self, text: str, speaker: str) -> torch.Tensor:
        """Synthesize ``text`` in the named voice. Returns ``[T_samples]`` float32 PCM."""


class SilenceTTS(BaseTTS):
    """Placeholder TTS that emits zero audio proportional to text length.

    Useful only for validating the preprocessing -> collator -> trainer
    pipeline end-to-end without committing to a TTS dependency. The
    resulting audio codes are essentially the Mimi "silence" token at
    every timestep; the trainer runs and gradients flow but the model
    learns nothing meaningful about audio. Swap in a real TTS for
    production.

    Roughly mirrors typical speech rate: ~150 wpm -> ~80 ms per word.
    """

    def __init__(self, sample_rate: int = 24000, ms_per_word: int = 80) -> None:
        self.sample_rate = sample_rate
        self.ms_per_word = ms_per_word

    def synthesize(self, text: str, speaker: str) -> torch.Tensor:
        del speaker
        n_words = max(1, len(text.split()))
        n_samples = int(self.sample_rate * (self.ms_per_word / 1000.0) * n_words)
        return torch.zeros(n_samples, dtype=torch.float32)


class CoquiXTTS(BaseTTS):
    """Driver for Coqui's open-source ``TTS`` package (XTTS v2).

    Lazy-imports ``TTS`` so the rest of this module loads without it.
    Each speaker name is mapped to a reference audio clip via the
    ``speaker_refs`` dict you provide at construction:

    .. code-block:: python

        tts = CoquiXTTS(speaker_refs={
            "user": "/path/to/user_reference.wav",
            "moshi": "/path/to/moshi_reference.wav",
        })

    Install with ``pip install TTS``. Note: TTS pulls heavy deps
    (transformers, librosa, etc.); a separate venv for the
    preprocessing job is recommended.
    """

    def __init__(
        self,
        speaker_refs: dict[str, str],
        *,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
        device: str = "cuda",
    ) -> None:
        try:
            from TTS.api import TTS as _TTS  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "CoquiXTTS requires the Coqui ``TTS`` package. "
                "Install with ``pip install TTS``."
            ) from e
        self._tts = _TTS(model_name=model_name).to(device)
        self.sample_rate = int(self._tts.synthesizer.output_sample_rate)
        self.speaker_refs = speaker_refs
        self.language = language

    def synthesize(self, text: str, speaker: str) -> torch.Tensor:
        ref = self.speaker_refs.get(speaker)
        if ref is None:
            raise KeyError(
                f"No speaker reference configured for {speaker!r}; "
                f"known speakers: {list(self.speaker_refs)}"
            )
        wav = self._tts.tts(text=text, speaker_wav=ref, language=self.language)
        return torch.tensor(np.asarray(wav, dtype=np.float32))


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------


@dataclass
class AudioPreprocessorConfig:
    """Per-example processing parameters."""

    output_dir: str | Path
    mimi_sample_rate: int = 24000
    silence_ms_between_turns: int = 200
    user_speaker_id: str = "user"
    moshi_speaker_id: str = "moshi"
    # Cap individual turns to this many seconds so a runaway TTS or a
    # very long moshi turn doesn't blow up the on-disk file.
    max_turn_seconds: float = 30.0


class AudioPreprocessor:
    """Runs the per-example pipeline. One instance per process."""

    def __init__(
        self,
        tts: BaseTTS,
        mimi: "MimiModel",
        config: AudioPreprocessorConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.tts = tts
        self.mimi = mimi
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resample_if_needed(self, pcm: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.config.mimi_sample_rate:
            return pcm
        # Lazy torchaudio import; the resampling op is fairly common and
        # not worth a hard dep on the rest of the codebase.
        try:
            import torchaudio.functional as taf  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Audio resampling requires torchaudio. "
                "Install with ``pip install torchaudio``."
            ) from e
        return taf.resample(pcm, src_sr, self.config.mimi_sample_rate)

    def _silence(self, ms: int) -> torch.Tensor:
        n = int(self.config.mimi_sample_rate * ms / 1000)
        return torch.zeros(n, dtype=torch.float32)

    def process_example(self, example: RagExample, idx: int) -> Optional[Path]:
        """Synthesize, encode, and save audio for one example. Returns the
        output path (or ``None`` if the example produced no audio)."""
        # Two parallel streams: moshi output, user input. Both at the same
        # timeline; positions not occupied by that speaker are silence.
        moshi_pcm: list[torch.Tensor] = []
        user_pcm: list[torch.Tensor] = []

        for turn in example.turns:
            if turn.role == "reference":
                # No audio for reference turns. Skip entirely (don't even
                # add silence -- they aren't part of the speech timeline).
                continue
            if turn.role == "tool":
                # Tool result is text-only: the model consumes it via
                # the text stream (collator includes it with loss_mask=
                # False). Both audio streams stay silent for the tool
                # turn's notional duration so the audio timeline stays
                # roughly aligned with the text timeline. Sizing the
                # silence proportional to text length matches the
                # SilenceTTS heuristic (~80 ms per word).
                n_words = max(1, len(turn.text.split()))
                n_samples = int(self.config.mimi_sample_rate * 0.08 * n_words)
                silence = torch.zeros(n_samples, dtype=torch.float32)
                gap = self._silence(self.config.silence_ms_between_turns)
                moshi_pcm.append(silence)
                moshi_pcm.append(gap)
                user_pcm.append(silence)
                user_pcm.append(gap)
                continue
            speaker_id = (
                self.config.moshi_speaker_id
                if turn.role == "moshi"
                else self.config.user_speaker_id
            )
            try:
                wav = self.tts.synthesize(turn.text, speaker_id)
            except Exception as e:
                logger.warning(
                    "[preprocess] example %d turn %r TTS failed: %s -- "
                    "filling with silence",
                    idx,
                    turn.role,
                    e,
                )
                wav = torch.zeros(
                    int(self.tts.sample_rate * 0.5), dtype=torch.float32
                )
            # Guard against TTS returning empty or 1-sample tensors --
            # both produce a degenerate stream alignment (the two
            # speakers' timelines need equal sample counts). Substitute
            # a minimal silence so the rest of the pipeline doesn't
            # crash on a zero-length cat.
            min_samples = max(1, int(self.tts.sample_rate * 0.1))
            if wav.numel() < min_samples:
                logger.warning(
                    "[preprocess] example %d turn %r TTS returned %d samples "
                    "(< %d); padding with silence",
                    idx,
                    turn.role,
                    wav.numel(),
                    min_samples,
                )
                wav = torch.zeros(min_samples, dtype=torch.float32)
            wav = self._resample_if_needed(wav, self.tts.sample_rate)
            max_samples = int(
                self.config.max_turn_seconds * self.config.mimi_sample_rate
            )
            if wav.numel() > max_samples:
                wav = wav[:max_samples]

            turn_len = wav.numel()
            gap = self._silence(self.config.silence_ms_between_turns)

            if turn.role == "moshi":
                moshi_pcm.append(wav)
                moshi_pcm.append(gap)
                # User stays silent during a moshi turn.
                user_pcm.append(torch.zeros(turn_len + gap.numel(), dtype=torch.float32))
            else:  # user
                user_pcm.append(wav)
                user_pcm.append(gap)
                moshi_pcm.append(torch.zeros(turn_len + gap.numel(), dtype=torch.float32))

        if not moshi_pcm and not user_pcm:
            return None

        moshi_track = torch.cat(moshi_pcm)
        user_track = torch.cat(user_pcm)
        assert moshi_track.numel() == user_track.numel(), (
            f"stream length mismatch: moshi={moshi_track.numel()} "
            f"user={user_track.numel()}"
        )

        # Encode through Mimi. Mimi takes [B, channels, T_samples] and
        # returns [B, K_codebooks, T_frames].
        moshi_codes = self._encode(moshi_track)
        user_codes = self._encode(user_track)

        # Stack into [n_codebooks, T] for the trainer collator. Mimi's
        # output is per-stream; the trainer's collator expects
        # ``[num_audio_codebooks, T]`` where the first half are moshi
        # output codebooks and the second half are user input codebooks.
        # The number of codebooks per stream is ``mimi.num_codebooks``.
        # Concatenating along the codebook dim gives the layout
        # MoshiVis expects in ``input_ids`` channels ``1..num_codebooks``.
        codes = torch.cat([moshi_codes, user_codes], dim=0)
        out_path = self.output_dir / f"{idx}.pt"
        torch.save(codes, out_path)
        return out_path

    def _encode(self, pcm: torch.Tensor) -> torch.Tensor:
        """Mimi-encode a 1-D PCM track. Returns ``[K, T_frames]`` long."""
        pcm = pcm.to(self.device).view(1, 1, -1)
        with torch.no_grad():
            codes = self.mimi.encode(pcm)
        # Drop batch dim: [1, K, T] -> [K, T]
        return codes[0].cpu()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_default_tts(name: str, **kwargs) -> BaseTTS:
    if name == "silence":
        return SilenceTTS()
    if name == "coqui_xtts":
        return CoquiXTTS(**kwargs)
    raise ValueError(
        f"Unknown TTS backend {name!r}. Choose one of: silence, coqui_xtts. "
        f"For others, instantiate the BaseTTS subclass directly and call "
        f"AudioPreprocessor.process_example."
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone preprocessing entry point.

    .. code-block:: bash

        python -m kyuteye.training.audio_preprocess \\
            --input data/augmented.jsonl \\
            --output-dir data/audio_codes \\
            --mimi-weight $MIMI_WEIGHTS \\
            --tts silence

    For real TTS:

    .. code-block:: bash

        python -m kyuteye.training.audio_preprocess \\
            --input data/augmented.jsonl \\
            --output-dir data/audio_codes \\
            --mimi-weight $MIMI_WEIGHTS \\
            --tts coqui_xtts \\
            --tts-user-ref refs/user.wav \\
            --tts-moshi-ref refs/moshi.wav

    See ``slurm/preprocess_audio.sbatch`` for the cluster-submission
    pattern (one preprocessing job per dataset, run once).
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--input", required=True, help="Path to JSONL input")
    parser.add_argument("--output-dir", required=True, help="Where to write per-example .pt files")
    parser.add_argument("--mimi-weight", required=True, help="Path to Mimi weights (.safetensors)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tts", default="silence", help="silence|coqui_xtts")
    parser.add_argument("--tts-user-ref", help="Reference audio for user voice (coqui_xtts)")
    parser.add_argument("--tts-moshi-ref", help="Reference audio for moshi voice (coqui_xtts)")
    parser.add_argument("--start-idx", type=int, default=0, help="Process examples [start_idx, end_idx)")
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip examples whose output already exists")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build TTS.
    if args.tts == "coqui_xtts":
        if not args.tts_user_ref or not args.tts_moshi_ref:
            parser.error("coqui_xtts needs --tts-user-ref and --tts-moshi-ref")
        tts = _build_default_tts(
            args.tts,
            speaker_refs={"user": args.tts_user_ref, "moshi": args.tts_moshi_ref},
            device=args.device,
        )
    else:
        tts = _build_default_tts(args.tts)

    # Build Mimi (lazy-import since loaders pulls torch + safetensors).
    from moshi.models.loaders import get_mimi

    mimi = get_mimi(args.mimi_weight, device=args.device)
    mimi.eval()

    config = AudioPreprocessorConfig(
        output_dir=args.output_dir,
        mimi_sample_rate=int(mimi.sample_rate),
    )
    preprocessor = AudioPreprocessor(tts, mimi, config, device=args.device)
    dataset = RagJsonlDataset(args.input)

    start = args.start_idx
    end = args.end_idx if args.end_idx >= 0 else len(dataset)
    end = min(end, len(dataset))
    logger.info(
        "[preprocess] processing examples [%d, %d) of %d total",
        start, end, len(dataset),
    )
    output_dir = Path(args.output_dir)
    n_done = 0
    n_skipped = 0
    for idx in range(start, end):
        target = output_dir / f"{idx}.pt"
        if args.skip_existing and target.exists():
            n_skipped += 1
            continue
        try:
            preprocessor.process_example(dataset[idx], idx)
        except Exception as e:
            logger.error("[preprocess] example %d failed: %s", idx, e)
            continue
        n_done += 1
        if n_done % 100 == 0:
            logger.info("[preprocess] %d done (%d skipped) ...", n_done, n_skipped)
    logger.info(
        "[preprocess] complete: %d processed, %d skipped, output in %s",
        n_done, n_skipped, output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
