"""Optional no-reference audio quality scoring via NISQA.

NISQA ships as a repository with its own script and conda environment rather
than as a PyPI package, so it is driven as a subprocess. The notebook's version
pinned NumPy globally to satisfy it, which breaks the rest of the stack; here
NISQA is isolated to its own checkout and the caller keeps their environment.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

__all__ = ["NisqaNotAvailable", "score_file"]


class NisqaNotAvailable(RuntimeError):
    """Raised when the NISQA checkout or its pretrained weights are missing."""


def score_file(
    audio_path: str | Path,
    *,
    nisqa_dir: str | Path = "NISQA",
    pretrained_model: str | Path = "weights/nisqa.tar",
    output_dir: str | Path = "results",
    timeout: float = 600.0,
) -> dict[str, str]:
    """Score one audio file and return NISQA's predicted metrics.

    Expects a NISQA checkout at ``nisqa_dir``::

        git clone https://github.com/gabrielmittag/NISQA.git

    ``pretrained_model`` is resolved relative to ``nisqa_dir`` when relative.
    Returns the first row of NISQA's prediction CSV as a dict (typically
    ``mos_pred``, ``noi_pred``, ``dis_pred``, ``col_pred``, ``loud_pred``).
    """
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")

    root = Path(nisqa_dir)
    script = root / "run_predict.py"
    if not script.is_file():
        raise NisqaNotAvailable(
            f"NISQA not found at {root}. Clone it first:\n"
            "  git clone https://github.com/gabrielmittag/NISQA.git"
        )

    weights = Path(pretrained_model)
    if not weights.is_absolute():
        weights = root / weights
    if not weights.is_file():
        raise NisqaNotAvailable(f"pretrained model not found: {weights}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script),
            "--mode", "predict_file",
            "--pretrained_model", str(weights),
            "--deg", str(audio.resolve()),
            "--output_dir", str(destination.resolve()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"NISQA failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )

    predictions = sorted(destination.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not predictions:
        raise RuntimeError(f"NISQA produced no CSV in {destination}")

    with predictions[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"NISQA CSV {predictions[0]} is empty")

    return rows[0]
