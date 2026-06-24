# Voice-Cloning

Zero-shot voice cloning and long-form text-to-speech in a single Colab notebook, built on [Suno Bark](https://github.com/suno-ai/bark) and the [bark-voice-cloning HuBERT quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer), with an optional NISQA audio-quality check.

## Overview

This project clones a target speaker's voice from a single reference recording and then generates new speech in that voice from arbitrary text. It does this **without training or fine-tuning** any model: a short audio clip is turned into a reusable Bark "voice prompt" (an `.npz` file), which conditions Bark's generative TTS stack so subsequent generations sound like the cloned speaker.

The workflow runs end to end in Google Colab on a GPU runtime and covers three things:

- **Voice cloning** — convert a reference `.wav` into a Bark history prompt.
- **Speech generation** — synthesize new audio from text in the cloned voice, with both a simple one-call API and a lower-level, fully controllable pipeline.
- **Long-form synthesis** — split long scripts into sentences and stitch the generated clips together for paragraph-length narration.
- **Audio quality check (optional)** — score generated audio with [NISQA](https://github.com/gabrielmittag/NISQA), a no-reference speech-quality model.

## What's inside

| File | Description |
| --- | --- |
| `Voice_generation.ipynb` | The complete Colab notebook: installation, voice cloning, inference, long-form generation, and the NISQA quality-check section. |
| `README.md` | This file. |

The notebook is organized into labeled sections — **Installation**, **Inference**, **Long form text generation**, and **Audio Quality Check** — and includes reusable helpers `voice_cloner(audio_file, text_prompt)` and `voice_generator(npz_file, text_prompt, Gen_temp)`.

## Methods / Approach

**1. Building the voice prompt (cloning).**
A reference recording is loaded with `torchaudio` and resampled to the EnCodec model's rate via `encodec.utils.convert_audio`. Two representations are then extracted:

- **Semantic tokens** — `CustomHubert` produces semantic vectors from the audio, which a `CustomTokenizer` maps to discrete semantic tokens.
- **Acoustic codes** — Bark's EnCodec codec encodes the waveform into discrete acoustic codes.

These are packed into an `.npz` file with `semantic_prompt`, `coarse_prompt`, and `fine_prompt` arrays — the history prompt Bark uses to condition generation on the cloned speaker.

**2. Generating speech.**
Two paths are provided:

- **Simple:** `generate_audio(text_prompt, history_prompt=voice.npz, text_temp=..., waveform_temp=...)`.
- **Controllable:** the staged Bark pipeline — `generate_text_semantic` → `generate_coarse` → `generate_fine` → `codec_decode` — exposing `temp`, `top_k`, and `top_p` at each stage.

**3. Long-form synthesis.**
Long scripts are tokenized into sentences with `nltk`, each sentence is generated independently, and the clips are concatenated with short silence gaps to form continuous narration. Output is written as WAV (`scipy.io.wavfile`) or 24-bit FLAC (`soundfile`).

**4. Quality scoring.**
The notebook wires up NISQA (`run_predict.py --mode predict_file`) to score generated audio with a pretrained no-reference quality model.

## How to run

The notebook targets **Google Colab with a GPU runtime**. Open `Voice_generation.ipynb` in Colab, set the runtime to GPU, and run the cells top to bottom.

The installation cell pulls in the required dependencies:

```bash
pip install git+https://github.com/suno-ai/bark.git
git clone https://github.com/suno-ai/bark.git
git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
pip install -r ./bark-voice-cloning-HuBERT-quantizer/requirements.txt
mv bark-voice-cloning-HuBERT-quantizer bark_voice_cloning_HuBERT_quantizer
```

Then, in the **Inference** section:

1. Upload a reference recording (e.g. `/content/Recording.wav`).
2. Set `text_prompt` to the words you want spoken.
3. Run the cloning cell to build `output.npz`, then generate audio with that prompt.

The HuBERT and tokenizer checkpoints are fetched into `data/models/hubert/` via the quantizer's `HuBERTManager`.

For the optional quality check:

```bash
git clone https://github.com/gabrielmittag/NISQA.git
python NISQA/run_predict.py --mode predict_file \
  --pretrained_model weights/nisqa.tar \
  --deg /content/Recording.wav \
  --output_dir /content/results
```

## Tech stack

Python · PyTorch · Suno Bark · EnCodec · HuBERT (bark-voice-cloning quantizer) · torchaudio · NLTK · NISQA · Google Colab (GPU)

## Notes

This notebook is a research/experimentation harness rather than a packaged library. The NISQA quality-check section is included as a scaffold; depending on the Colab environment it may require dependency pinning (e.g. NumPy) to run cleanly.

## Credits

- [Suno Bark](https://github.com/suno-ai/bark) — generative text-to-audio model.
- [bark-voice-cloning HuBERT quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer) by gitmylo — semantic tokenization for Bark voice cloning.
- [NISQA](https://github.com/gabrielmittag/NISQA) by Gabriel Mittag — no-reference speech quality assessment.
