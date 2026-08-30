"""Turning a reference recording into a reusable Bark voice prompt.

No training or fine-tuning happens here. A single clip is encoded twice - into
semantic tokens via HuBERT, and into acoustic codes via EnCodec - and the two
representations are stored as the ``.npz`` history prompt Bark conditions on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import ModelBundle, load_models
from .voice_prompt import save_voice_prompt

__all__ = ["encode_reference", "clone_voice"]


def encode_reference(models: ModelBundle, audio_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Encode a reference clip into ``(codes, semantic_tokens)``.

    The waveform is resampled to the codec's rate and channel count before
    encoding, since EnCodec assumes its own sample rate.
    """
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from encodec.utils import convert_audio  # noqa: PLC0415

    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"reference audio not found: {source}")

    wav, sample_rate = torchaudio.load(str(source))
    wav = convert_audio(wav, sample_rate, models.codec.sample_rate, models.codec.channels)
    wav = wav.to(models.device)

    semantic_vectors = models.hubert.forward(wav, input_sample_hz=models.codec.sample_rate)
    semantic_tokens = models.tokenizer.get_token(semantic_vectors)

    with torch.no_grad():
        encoded_frames = models.codec.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    return codes.cpu().numpy(), semantic_tokens.cpu().numpy()


def clone_voice(
    audio_path: str | Path,
    output_path: str | Path = "output.npz",
    models: ModelBundle | None = None,
) -> Path:
    """Clone the speaker in ``audio_path`` into a voice prompt at ``output_path``.

    Pass ``models`` to reuse an already-loaded bundle; the notebook reloaded
    every model on each call, which dominated runtime when cloning more than
    one speaker.

    Returns the path to the written ``.npz``.
    """
    models = models or load_models()
    codes, semantic_tokens = encode_reference(models, audio_path)
    return save_voice_prompt(output_path, codes=codes, semantic_tokens=semantic_tokens)
