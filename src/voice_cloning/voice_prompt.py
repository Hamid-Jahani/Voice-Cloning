"""Reading and writing Bark voice prompts (``.npz`` history prompts).

A Bark history prompt is an ``.npz`` holding three arrays:

``semantic_prompt``
    Discrete semantic tokens from the HuBERT quantizer.
``coarse_prompt``
    The first two EnCodec codebooks.
``fine_prompt``
    All EnCodec codebooks.

Serialisation lives here, apart from the model code that produces the arrays,
so the on-disk format can be tested without Bark or a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["VoicePrompt", "REQUIRED_KEYS", "save_voice_prompt", "load_voice_prompt"]

#: Keys Bark expects in a history prompt.
REQUIRED_KEYS = ("semantic_prompt", "coarse_prompt", "fine_prompt")

#: Number of EnCodec codebooks Bark's coarse stage consumes.
COARSE_CODEBOOKS = 2


class VoicePrompt(NamedTuple):
    """The three arrays that make up a Bark history prompt."""

    semantic_prompt: np.ndarray
    coarse_prompt: np.ndarray
    fine_prompt: np.ndarray


def save_voice_prompt(
    path: str | Path,
    *,
    codes: np.ndarray,
    semantic_tokens: np.ndarray,
) -> Path:
    """Write a Bark history prompt built from EnCodec ``codes``.

    ``codes`` is the full ``(n_codebooks, n_frames)`` EnCodec matrix. The
    coarse prompt is its first :data:`COARSE_CODEBOOKS` rows and the fine
    prompt is the whole matrix, as Bark expects.

    Raises ``ValueError`` if ``codes`` is not 2-D or has too few codebooks,
    which otherwise surfaces much later as an opaque failure inside Bark.
    """
    codes = np.asarray(codes)
    semantic_tokens = np.asarray(semantic_tokens)

    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D (n_codebooks, n_frames), got shape {codes.shape}")
    if codes.shape[0] < COARSE_CODEBOOKS:
        raise ValueError(
            f"codes needs at least {COARSE_CODEBOOKS} codebooks, got {codes.shape[0]}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        semantic_prompt=semantic_tokens,
        coarse_prompt=codes[:COARSE_CODEBOOKS, :],
        fine_prompt=codes,
    )
    # np.savez appends .npz when the name lacks it; report the real path.
    if destination.suffix != ".npz":
        destination = destination.with_suffix(destination.suffix + ".npz")
    return destination


def load_voice_prompt(path: str | Path) -> VoicePrompt:
    """Load a history prompt, verifying all three required arrays are present."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"voice prompt not found: {source}")

    with np.load(source) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"{source} is not a Bark voice prompt; missing array(s): {', '.join(missing)}"
            )
        return VoicePrompt(
            semantic_prompt=data["semantic_prompt"],
            coarse_prompt=data["coarse_prompt"],
            fine_prompt=data["fine_prompt"],
        )
