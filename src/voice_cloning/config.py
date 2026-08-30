"""Configuration objects.

Every default here is the value that was hard-coded in the original Colab
notebook, so behaviour is unchanged by the refactor. They are gathered in one
place so a caller can override them without editing generation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Bark's output sample rate. Re-declared rather than imported so this module
#: stays importable without Bark installed; :func:`sample_rate` reconciles the
#: two at runtime.
DEFAULT_SAMPLE_RATE = 24_000


def sample_rate() -> int:
    """Return Bark's sample rate, falling back to the documented default.

    Importing Bark pulls in torch and triggers model-path resolution, which is
    unwanted in tests and in the CLI's ``--help`` path.
    """
    try:
        from bark.generation import SAMPLE_RATE  # noqa: PLC0415

        return int(SAMPLE_RATE)
    except Exception:
        return DEFAULT_SAMPLE_RATE


@dataclass(frozen=True)
class ModelConfig:
    """Where the HuBERT checkpoints live and what device to run on."""

    hubert_checkpoint: Path = Path("data/models/hubert/hubert.pt")
    tokenizer_checkpoint: Path = Path("data/models/hubert/tokenizer.pth")
    device: str | None = None
    """``None`` selects cuda when available, else cpu."""

    use_small_models: bool = False
    force_reload: bool = False

    def resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@dataclass(frozen=True)
class SynthesisConfig:
    """Sampling parameters for one-shot generation.

    Mirrors ``generate_audio(text_temp=0.7, waveform_temp=0.7)`` from the
    notebook's Inference section.
    """

    text_temp: float = 0.7
    waveform_temp: float = 0.7

    def __post_init__(self) -> None:
        for name in ("text_temp", "waveform_temp"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")


@dataclass(frozen=True)
class ControlledSynthesisConfig:
    """Per-stage parameters for the staged Bark pipeline.

    semantic -> coarse -> fine -> codec_decode. The fine stage uses a lower
    temperature than the earlier two, as in the notebook.
    """

    semantic_temp: float = 0.7
    semantic_top_k: int = 50
    semantic_top_p: float = 0.95
    coarse_temp: float = 0.7
    coarse_top_k: int = 50
    coarse_top_p: float = 0.95
    fine_temp: float = 0.5


@dataclass(frozen=True)
class LongFormConfig:
    """Parameters for sentence-by-sentence narration.

    ``min_eos_p`` controls how eagerly Bark ends a segment; ``silence_seconds``
    is the gap inserted between sentences when the clips are stitched.
    """

    gen_temp: float = 0.6
    min_eos_p: float = 0.05
    silence_seconds: float = 0.15
    sample_rate: int = field(default_factory=sample_rate)

    def __post_init__(self) -> None:
        if self.silence_seconds < 0:
            raise ValueError(f"silence_seconds must be >= 0, got {self.silence_seconds!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate!r}")
