"""Arabic-aware text normalization and lightweight sentence splitting.

Heuristics only: these functions are deterministic, dependency-free tools used
by the baseline and evaluation engine. They are not a substitute for linguistic
analysis.
"""
from __future__ import annotations

import re
from typing import List

# Arabic diacritics (tashkeel)
_HARAKAT = re.compile(r"[\u064B-\u065F]")
_TATWEEL = re.compile(r"ـ+")
_MULTISPACE = re.compile(r"\s+")

# Arabic punctuation mapped to ASCII equivalents for simpler matching
_ARABIC_PUNCT_MAP = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٫": ".",
    "：": ":",
}

# Alef variants -> plain alef
_ALEF_MAP = str.maketrans({"إ": "ا", "أ": "ا", "آ": "ا"})
_YE_MAP = str.maketrans({"ى": "ي", "ؤ": "و", "ئ": "ي"})
_TEH_MARBUTA_MAP = str.maketrans({"ة": "ه"})


def normalize_arabic(text: str) -> str:
    """Normalize Arabic script for matching (not for display)."""
    if not text:
        return text
    text = _HARAKAT.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_ALEF_MAP)
    text = text.translate(_YE_MAP)
    text = text.translate(_TEH_MARBUTA_MAP)
    for ar, en in _ARABIC_PUNCT_MAP.items():
        text = text.replace(ar, en)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def normalize(text: str, arabic: bool = True) -> str:
    """Lowercase; optionally apply Arabic-script normalization."""
    if not text:
        return text
    if arabic:
        text = normalize_arabic(text)
    return text.lower()


def split_sentences(text: str) -> List[str]:
    """Crude sentence splitter handling Arabic and English sentence boundaries."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?،؛؟])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def word_tokens(text: str, arabic: bool = True) -> List[str]:
    return normalize(text, arabic=arabic).split()


def content_words(text: str, arabic: bool = True) -> List[str]:
    """Words longer than 2 chars, useful for fuzzy fact matching."""
    return [w for w in word_tokens(text, arabic=arabic) if len(w) > 2]
