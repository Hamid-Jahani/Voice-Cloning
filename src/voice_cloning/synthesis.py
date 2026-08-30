"""Speech generation from a cloned voice prompt.

Three paths, matching the notebook:

``synthesize``
    One call to Bark's high-level API.
``synthesize_controlled``
    The staged pipeline (semantic -> coarse -> fine -> decode), exposing
    temperature and nucleus-sampling knobs per stage.
``synthesize_long_form``
    Sentence-by-sentence generation stitched into continuous narration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .audio import make_silence, stitch
from .config import ControlledSynthesisConfig, LongFormConfig, SynthesisConfig
from .text import split_sentences

__all__ = ["synthesize", "synthesize_controlled", "synthesize_long_form"]

ProgressCallback = Callable[[int, int, str], None]


def _prompt_path(voice_prompt: str | Path) -> str:
    source = Path(voice_prompt)
    if not source.exists():
        raise FileNotFoundError(f"voice prompt not found: {source}")
    return str(source)


def synthesize(
    text: str,
    voice_prompt: str | Path,
    config: SynthesisConfig | None = None,
) -> np.ndarray:
    """Generate speech for ``text`` in the voice at ``voice_prompt``."""
    from bark.api import generate_audio  # noqa: PLC0415

    config = config or SynthesisConfig()
    return generate_audio(
        text,
        history_prompt=_prompt_path(voice_prompt),
        text_temp=config.text_temp,
        waveform_temp=config.waveform_temp,
    )


def synthesize_controlled(
    text: str,
    voice_prompt: str | Path,
    config: ControlledSynthesisConfig | None = None,
) -> np.ndarray:
    """Generate speech through Bark's staged pipeline for finer control.

    Equivalent to :func:`synthesize` but exposes each stage, which matters when
    tuning away artefacts: the semantic stage governs prosody and phrasing, the
    fine stage governs timbre detail.
    """
    from bark.generation import (  # noqa: PLC0415
        codec_decode,
        generate_coarse,
        generate_fine,
        generate_text_semantic,
    )

    config = config or ControlledSynthesisConfig()
    prompt = _prompt_path(voice_prompt)

    semantic = generate_text_semantic(
        text,
        history_prompt=prompt,
        temp=config.semantic_temp,
        top_k=config.semantic_top_k,
        top_p=config.semantic_top_p,
    )
    coarse = generate_coarse(
        semantic,
        history_prompt=prompt,
        temp=config.coarse_temp,
        top_k=config.coarse_top_k,
        top_p=config.coarse_top_p,
    )
    fine = generate_fine(coarse, history_prompt=prompt, temp=config.fine_temp)
    return codec_decode(fine)


def synthesize_long_form(
    script: str,
    voice_prompt: str | Path,
    config: LongFormConfig | None = None,
    *,
    on_progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Generate narration for a multi-sentence ``script``.

    Bark degrades past roughly one sentence, so each sentence is generated
    independently and the clips are stitched with a short gap.

    ``on_progress`` is called as ``(index, total, sentence)`` before each
    sentence, letting a caller report progress on long scripts.

    Raises ``ValueError`` when ``script`` contains no sentences, rather than
    silently returning an empty waveform.
    """
    from bark.api import semantic_to_waveform  # noqa: PLC0415
    from bark.generation import generate_text_semantic  # noqa: PLC0415

    config = config or LongFormConfig()
    prompt = _prompt_path(voice_prompt)

    sentences = split_sentences(script)
    if not sentences:
        raise ValueError("script contains no sentences to synthesise")

    silence = make_silence(config.silence_seconds, config.sample_rate)

    pieces: list[np.ndarray] = []
    for index, sentence in enumerate(sentences):
        if on_progress is not None:
            on_progress(index, len(sentences), sentence)

        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=prompt,
            temp=config.gen_temp,
            min_eos_p=config.min_eos_p,
        )
        pieces.append(semantic_to_waveform(semantic_tokens, history_prompt=prompt))

    return stitch(pieces, silence)
