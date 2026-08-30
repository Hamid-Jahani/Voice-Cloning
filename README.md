# Voice-Cloning

Zero-shot voice cloning and long-form text-to-speech, built on [Suno Bark](https://github.com/suno-ai/bark) and the [bark-voice-cloning HuBERT quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer), with an optional NISQA audio-quality check.

**Tech stack:** Python · PyTorch · Suno Bark · EnCodec · HuBERT · torchaudio · NLTK · NISQA

## Overview

This project clones a target speaker's voice from a single reference recording and then generates new speech in that voice from arbitrary text. It does this **without training or fine-tuning** any model: a short audio clip is turned into a reusable Bark "voice prompt" (an `.npz` file), which conditions Bark's generative TTS stack so subsequent generations sound like the cloned speaker.

It covers four things:

- **Voice cloning** — convert a reference `.wav` into a Bark history prompt.
- **Speech generation** — synthesize new audio from text in the cloned voice, with both a one-call API and a lower-level, fully controllable pipeline.
- **Long-form synthesis** — split long scripts into sentences and stitch the generated clips together for paragraph-length narration.
- **Audio quality check (optional)** — score generated audio with [NISQA](https://github.com/gabrielmittag/NISQA), a no-reference speech-quality model.

## What's inside

```
Voice-Cloning/
├── src/voice_cloning/
│   ├── config.py         # Dataclass configs; defaults match the original notebook
│   ├── text.py           # Sentence splitting for long-form narration
│   ├── audio.py          # Silence, clip stitching, WAV/FLAC output
│   ├── voice_prompt.py   # Read/write Bark .npz history prompts
│   ├── models.py         # Bark + HuBERT + tokenizer loading
│   ├── cloning.py        # Reference audio -> voice prompt
│   ├── synthesis.py      # Simple, controlled, and long-form generation
│   ├── quality.py        # NISQA subprocess wrapper
│   └── cli.py            # clone / speak / narrate / score
├── tests/                # 63 tests over the model-independent logic
├── notebooks/
│   └── Voice_generation.ipynb   # Original Colab exploration notebook
├── pyproject.toml
└── requirements.txt
```

The package is split so that everything not requiring Bark — sentence splitting, waveform assembly, the `.npz` format, configuration — is importable and testable without model weights or a GPU. `import voice_cloning` pulls in no heavy dependencies; Bark and torch are imported lazily inside the functions that need them.

## Installation

```bash
git clone https://github.com/sheperd007/Voice-Cloning.git
cd Voice-Cloning
pip install -r requirements.txt
pip install -e .
```

Bark and the HuBERT quantizer install from git, so they are listed in `requirements.txt` rather than in `pyproject.toml`. A CUDA GPU is strongly recommended — on CPU, long-form narration is impractical.

The HuBERT and tokenizer checkpoints are fetched automatically into `data/models/hubert/` on first run.

## Usage

### Command line

```bash
# 1. Clone a voice from a reference recording
voice-cloning clone --reference recording.wav --output voice.npz

# 2. Speak a single line in that voice
voice-cloning speak --voice voice.npz --text "Hello, this is a cloned voice." --output out.wav

# 3. Narrate a long script, sentence by sentence
voice-cloning narrate --voice voice.npz --script script.txt --output narration.wav

# 4. Score the result (needs a NISQA checkout)
voice-cloning score --audio out.wav
```

Use `--controlled` with `speak` to run Bark's staged pipeline instead of the single-call API, exposing per-stage temperature and nucleus sampling.

### Python

```python
from voice_cloning import clone_voice, synthesize, write_wav, sample_rate

prompt = clone_voice("recording.wav", "voice.npz")
audio = synthesize("Hello, this is a cloned voice.", prompt)
write_wav("out.wav", audio, sample_rate())
```

Long-form narration, reusing one loaded model bundle across several clones:

```python
from voice_cloning import LongFormConfig, synthesize_long_form, write_wav, sample_rate
from voice_cloning.models import load_models
from voice_cloning.cloning import clone_voice

models = load_models()                       # load once, reuse
prompt = clone_voice("recording.wav", "voice.npz", models=models)

audio = synthesize_long_form(
    open("script.txt", encoding="utf-8").read(),
    prompt,
    LongFormConfig(gen_temp=0.6, silence_seconds=0.15),
)
write_wav("narration.wav", audio, sample_rate())
```

## Methods

**1. Building the voice prompt (cloning).**
A reference recording is loaded with `torchaudio` and resampled to EnCodec's rate via `encodec.utils.convert_audio`. Two representations are extracted: **semantic tokens** (`CustomHubert` produces semantic vectors, which `CustomTokenizer` maps to discrete tokens) and **acoustic codes** (Bark's EnCodec codec encodes the waveform into discrete codes). These are packed into an `.npz` with `semantic_prompt`, `coarse_prompt`, and `fine_prompt` arrays — the history prompt Bark conditions on.

**2. Generating speech.**
Two paths: `synthesize()` wraps Bark's `generate_audio`, while `synthesize_controlled()` runs the staged pipeline — `generate_text_semantic` → `generate_coarse` → `generate_fine` → `codec_decode` — exposing `temp`, `top_k`, and `top_p` at each stage. The semantic stage governs prosody and phrasing; the fine stage governs timbre detail.

**3. Long-form synthesis.**
Bark degrades past roughly one sentence, so scripts are split with NLTK, generated per sentence, and stitched with a short silence gap.

**4. Quality scoring.**
`quality.score_file()` drives NISQA's `run_predict.py` as a subprocess against its own checkout and parses the resulting CSV.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the model-independent logic — sentence splitting and its regex fallback, silence generation, clip stitching, the `.npz` prompt format and its validation, config defaults and bounds, and CLI argument parsing. It requires no GPU, no model weights, and no Bark install.

The Bark-dependent paths (`models.py`, `cloning.py`, `synthesis.py`) are **not** covered by automated tests, since exercising them needs GPU weights and multi-gigabyte downloads. They are a direct port of the notebook's working code.

## Notes

Two behaviours differ deliberately from the original notebook:

- **Trailing silence.** The notebook appended a silence gap after *every* sentence, so narration ended in dead air. `audio.stitch()` places silence strictly *between* clips.
- **Model reloading.** The notebook's `voice_cloner()` reloaded every model on each call. `clone_voice()` accepts a preloaded `ModelBundle`, which dominates runtime when cloning more than one speaker.

## Credits

- [Suno Bark](https://github.com/suno-ai/bark) — generative text-to-audio model.
- [bark-voice-cloning HuBERT quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer) by gitmylo — semantic tokenization for Bark voice cloning.
- [NISQA](https://github.com/gabrielmittag/NISQA) by Gabriel Mittag — no-reference speech quality assessment.

## License

Released under the [MIT License](LICENSE).
