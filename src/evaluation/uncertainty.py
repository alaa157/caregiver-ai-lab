"""Uncertainty-marker detection for preservation metrics."""
from __future__ import annotations

from src.normalization import normalize

UNCERTAINTY_MARKERS_AR = [
    "مش متأكد", "غير متأكد", "لا ندري", "غير واضحة", "لا نعلم", "ربما",
    "ممكن", "غالباً", "أعتقد", "لا أعرف",
]
UNCERTAINTY_MARKERS_EN = [
    "unsure", "uncertain", "unknown", "maybe", "possibly", "probably",
    "not sure", "unclear", "i think",
]

_ALL_MARKERS = [normalize(m) for m in UNCERTAINTY_MARKERS_AR + UNCERTAINTY_MARKERS_EN]


def has_uncertainty_marker(text: str) -> bool:
    if not text:
        return False
    norm = normalize(text)
    return any(m in norm for m in _ALL_MARKERS)
