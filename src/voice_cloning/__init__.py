"""Zero-shot voice cloning and long-form text-to-speech built on Suno Bark.

Typical use::

    from voice_cloning import clone_voice, synthesize, write_wav, sample_rate

    prompt = clone_voice("reference.wav", "voice.npz")
    audio = synthesize("Hello there.", prompt)
    write_wav("out.wav", audio, sample_rate())

Submodules import Bark and torch lazily, so ``import voice_cloning`` succeeds
on a machine without model weights installed.
"""

from __future__ import annotations

from .audio import make_silence, stitch, write_flac, write_wav
from .config import (
    ControlledSynthesisConfig,
    LongFormConfig,
    ModelConfig,
    SynthesisConfig,
    sample_rate,
)
from .text import normalise_script, split_sentences
from .voice_prompt import VoicePrompt, load_voice_prompt, save_voice_prompt

__version__ = "0.1.0"

__all__ = [
    "ControlledSynthesisConfig",
    "LongFormConfig",
    "ModelConfig",
    "SynthesisConfig",
    "VoicePrompt",
    "__version__",
    "clone_voice",
    "load_voice_prompt",
    "make_silence",
    "normalise_script",
    "sample_rate",
    "save_voice_prompt",
    "split_sentences",
    "stitch",
    "synthesize",
    "synthesize_controlled",
    "synthesize_long_form",
    "write_flac",
    "write_wav",
]


def __getattr__(name: str):
    """Expose Bark-dependent entry points without importing Bark at import time.

    ``clone_voice`` and the ``synthesize*`` functions live in modules that pull
    in torch and Bark. Resolving them lazily keeps ``import voice_cloning`` free
    of heavy dependencies while still offering them as top-level names.
    """
    if name == "clone_voice":
        from .cloning import clone_voice

        return clone_voice
    if name in {"synthesize", "synthesize_controlled", "synthesize_long_form"}:
        from . import synthesis

        return getattr(synthesis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
