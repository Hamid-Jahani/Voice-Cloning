"""Command-line interface.

    voice-cloning clone    --reference ref.wav --output voice.npz
    voice-cloning speak    --voice voice.npz --text "Hello." --output out.wav
    voice-cloning narrate  --voice voice.npz --script script.txt --output story.wav
    voice-cloning score    --audio out.wav

Bark is imported only inside the subcommand that needs it, so ``--help`` and
argument validation work on a machine with no models installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    ControlledSynthesisConfig,
    LongFormConfig,
    ModelConfig,
    SynthesisConfig,
    sample_rate,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-cloning",
        description="Zero-shot voice cloning and long-form TTS built on Suno Bark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    clone = sub.add_parser("clone", help="build a voice prompt from a reference recording")
    clone.add_argument("--reference", required=True, type=Path, help="reference .wav of the target speaker")
    clone.add_argument("--output", type=Path, default=Path("output.npz"), help="where to write the .npz prompt")
    clone.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    clone.add_argument("--no-download", action="store_true", help="do not fetch HuBERT checkpoints")

    speak = sub.add_parser("speak", help="generate a single utterance")
    speak.add_argument("--voice", required=True, type=Path, help="voice prompt .npz")
    speak.add_argument("--text", required=True, help="text to speak")
    speak.add_argument("--output", type=Path, default=Path("out.wav"))
    speak.add_argument("--text-temp", type=float, default=SynthesisConfig().text_temp)
    speak.add_argument("--waveform-temp", type=float, default=SynthesisConfig().waveform_temp)
    speak.add_argument(
        "--controlled",
        action="store_true",
        help="use the staged pipeline (semantic/coarse/fine) instead of the single-call API",
    )

    narrate = sub.add_parser("narrate", help="generate long-form narration, sentence by sentence")
    narrate.add_argument("--voice", required=True, type=Path)
    narrate.add_argument("--script", required=True, type=Path, help="text file holding the script")
    narrate.add_argument("--output", type=Path, default=Path("narration.wav"))
    narrate.add_argument("--gen-temp", type=float, default=LongFormConfig().gen_temp)
    narrate.add_argument("--silence", type=float, default=LongFormConfig().silence_seconds,
                         help="seconds of silence between sentences")

    score = sub.add_parser("score", help="score audio quality with NISQA")
    score.add_argument("--audio", required=True, type=Path)
    score.add_argument("--nisqa-dir", type=Path, default=Path("NISQA"))
    score.add_argument("--output-dir", type=Path, default=Path("results"))

    return parser


def _cmd_clone(args: argparse.Namespace) -> int:
    from .cloning import clone_voice
    from .models import load_models

    models = load_models(ModelConfig(device=args.device), download_checkpoints=not args.no_download)
    written = clone_voice(args.reference, args.output, models=models)
    print(f"voice prompt written: {written}")
    return 0


def _cmd_speak(args: argparse.Namespace) -> int:
    from .audio import write_wav
    from .synthesis import synthesize, synthesize_controlled

    if args.controlled:
        audio = synthesize_controlled(args.text, args.voice, ControlledSynthesisConfig())
    else:
        audio = synthesize(
            args.text,
            args.voice,
            SynthesisConfig(text_temp=args.text_temp, waveform_temp=args.waveform_temp),
        )

    written = write_wav(args.output, audio, sample_rate())
    print(f"audio written: {written}")
    return 0


def _cmd_narrate(args: argparse.Namespace) -> int:
    from .audio import write_wav
    from .synthesis import synthesize_long_form

    script = args.script.read_text(encoding="utf-8")

    def report(index: int, total: int, sentence: str) -> None:
        preview = sentence if len(sentence) <= 60 else sentence[:57] + "..."
        print(f"  [{index + 1}/{total}] {preview}", file=sys.stderr)

    audio = synthesize_long_form(
        script,
        args.voice,
        LongFormConfig(gen_temp=args.gen_temp, silence_seconds=args.silence),
        on_progress=report,
    )
    written = write_wav(args.output, audio, sample_rate())
    print(f"narration written: {written}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .quality import score_file

    metrics = score_file(args.audio, nisqa_dir=args.nisqa_dir, output_dir=args.output_dir)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


_COMMANDS = {
    "clone": _cmd_clone,
    "speak": _cmd_speak,
    "narrate": _cmd_narrate,
    "score": _cmd_score,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
