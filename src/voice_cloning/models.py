"""Loading the Bark stack and the HuBERT semantic quantizer.

Every heavy import is deferred into function bodies. Importing this module is
therefore cheap and safe without Bark, torch, or GPU weights present - which
keeps ``--help``, tests, and static analysis fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig

__all__ = ["ModelBundle", "load_models"]


@dataclass
class ModelBundle:
    """The three models needed to clone a voice and synthesise speech."""

    codec: Any
    """Bark's EnCodec model, used to encode a reference clip."""

    hubert: Any
    """CustomHubert, producing semantic vectors from a waveform."""

    tokenizer: Any
    """CustomTokenizer, mapping semantic vectors to discrete tokens."""

    device: str


def _require(module: str, install_hint: str) -> Any:
    """Import ``module`` or raise with an actionable installation hint."""
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for this operation but is not installed.\n{install_hint}"
        ) from exc


_BARK_HINT = (
    "Install Bark and the HuBERT quantizer:\n"
    "  pip install git+https://github.com/suno-ai/bark.git\n"
    "  pip install git+https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer\n"
    "See requirements.txt for the full set."
)


def load_models(config: ModelConfig | None = None, *, download_checkpoints: bool = True) -> ModelBundle:
    """Load the codec, HuBERT, and tokenizer, preloading Bark's weights.

    ``download_checkpoints`` mirrors the notebook's ``HuBERTManager`` calls,
    which fetch ``hubert.pt`` and ``tokenizer.pth`` into ``data/models/hubert/``
    when absent. Set it False when the checkpoints are already in place.

    Raises ``ImportError`` with install instructions when Bark is missing, and
    ``FileNotFoundError`` when a checkpoint is absent and downloading is off.
    """
    config = config or ModelConfig()
    device = config.resolve_device()
    use_gpu = device == "cuda"

    generation = _require("bark.generation", _BARK_HINT)
    hubert_manager_mod = _require(
        "bark_hubert_quantizer.hubert_manager", _BARK_HINT
    )
    tokenizer_mod = _require("bark_hubert_quantizer.customtokenizer", _BARK_HINT)
    hubert_mod = _require("bark_hubert_quantizer.pre_kmeans_hubert", _BARK_HINT)

    codec = generation.load_codec_model(use_gpu=use_gpu)
    generation.preload_models(
        text_use_gpu=use_gpu,
        text_use_small=config.use_small_models,
        coarse_use_gpu=use_gpu,
        coarse_use_small=config.use_small_models,
        fine_use_gpu=use_gpu,
        fine_use_small=config.use_small_models,
        codec_use_gpu=use_gpu,
        force_reload=config.force_reload,
    )

    if download_checkpoints:
        manager = hubert_manager_mod.HuBERTManager()
        manager.make_sure_hubert_installed()
        manager.make_sure_tokenizer_installed()

    for checkpoint in (config.hubert_checkpoint, config.tokenizer_checkpoint):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint}. "
                "Pass download_checkpoints=True to fetch it automatically."
            )

    hubert = hubert_mod.CustomHubert(checkpoint_path=str(config.hubert_checkpoint)).to(device)
    tokenizer = tokenizer_mod.CustomTokenizer.load_from_checkpoint(
        str(config.tokenizer_checkpoint), map_location=device
    ).to(device)

    return ModelBundle(codec=codec, hubert=hubert, tokenizer=tokenizer, device=device)
