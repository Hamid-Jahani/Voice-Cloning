"""Text preparation for long-form synthesis.

Bark degrades on inputs longer than roughly a sentence, so narration is
produced one sentence at a time and stitched back together. This module owns
the splitting, kept free of Bark and torch so it can be tested directly.
"""

from __future__ import annotations

import re

__all__ = ["normalise_script", "split_sentences"]

# Fallback splitter: break after . ! or ? when followed by whitespace and a
# character that plausibly starts a new sentence. Not as good as Punkt, but it
# keeps the package usable when the nltk model has not been downloaded.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Z0-9])")


def normalise_script(text: str) -> str:
    """Collapse newlines and surrounding whitespace into single spaces.

    The notebook applied ``.replace("\\n", " ").strip()`` before splitting;
    this additionally collapses runs of whitespace so that sentence detection
    is not confused by indentation in triple-quoted scripts.
    """
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def split_sentences(text: str, *, use_nltk: bool = True) -> list[str]:
    """Split ``text`` into sentences, preferring nltk's Punkt tokenizer.

    Falls back to a regex when nltk is missing or its ``punkt`` data has not
    been downloaded, so a missing model degrades output quality rather than
    raising at generation time.

    Returns an empty list for blank input.
    """
    normalised = normalise_script(text)
    if not normalised:
        return []

    if use_nltk:
        try:
            import nltk  # noqa: PLC0415

            return [s for s in nltk.sent_tokenize(normalised) if s.strip()]
        except Exception:
            pass  # fall through to the regex splitter

    return [s.strip() for s in _SENTENCE_BOUNDARY.split(normalised) if s.strip()]
