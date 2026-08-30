"""Waveform assembly and file output.

Pure array and file handling, deliberately free of Bark and torch so that the
stitching logic can be tested without a GPU or model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = ["make_silence", "stitch", "write_wav", "write_flac"]


def make_silence(seconds: float, sample_rate: int) -> np.ndarray:
    """Return ``seconds`` of digital silence at ``sample_rate``.

    Length is truncated toward zero, matching the notebook's
    ``np.zeros(int(0.15 * SAMPLE_RATE))``.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds!r}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def stitch(pieces: Sequence[np.ndarray] | Iterable[np.ndarray], silence: np.ndarray) -> np.ndarray:
    """Concatenate clips, inserting ``silence`` between consecutive clips.

    The notebook appended silence after *every* clip, leaving a trailing gap on
    the final segment. Here silence is placed strictly between clips, so the
    result does not end in dead air.

    Returns an empty float32 array when there are no clips.
    """
    clips = [np.asarray(p) for p in pieces]
    if not clips:
        return np.zeros(0, dtype=np.float32)
    if len(clips) == 1:
        return clips[0]

    out: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        if index:
            out.append(silence)
        out.append(clip)
    return np.concatenate(out)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write ``audio`` as a WAV file, creating parent directories as needed."""
    from scipy.io.wavfile import write  # noqa: PLC0415

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(str(destination), sample_rate, audio)
    return destination


def write_flac(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> Path:
    """Write ``audio`` as FLAC. Requires the optional ``soundfile`` extra."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "FLAC output needs the 'soundfile' package. "
            "Install it with: pip install 'voice-cloning[flac]'"
        ) from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio, sample_rate, format="flac", subtype=subtype)
    return destination
