"""Evidence-matching heuristics: checks whether an evidence quote can be
traced back to the input note. Labeled heuristic — not definitive proof.
"""
from __future__ import annotations

from src.normalization import normalize, word_tokens


def quote_found_in_note(quote: str, note: str) -> bool:
    """Heuristic: returns True when the quote (or its main content words)
    appears in the normalized input note."""
    if not quote or not note:
        return False
    q = normalize(quote)
    n = normalize(note)
    if not q:
        return False
    if q in n:
        return True
    words = word_tokens(quote)
    if not words:
        return False
    return all(w in n for w in words)
