"""Rule-based extractive baseline.

No generation, no API key, no weights. It scores sentences by care-domain
keywords and returns the highest-scoring ones as a structured summary. This is
the reference point that proves whether a generative model adds real value.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import BASELINE_MODEL_ID
from src.models.base import ModelInterface, ModelResult
from src.normalization import split_sentences, word_tokens, normalize
from src.schemas import Observation, UnclearOrConflicting, CareSummary

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "appetite": ["أكل", "طعام", "وجبة", "شهية", "ate", "eat", "food", "lunch", "breakfast", "dinner"],
    "medication": ["دواء", "دواء", "دواء", "dawa", "medication", "medicine", "drug", "pill"],
    "sleep": ["نوم", "سهر", "نمت", "نوم", "sleep", "slept", "awake"],
    "mobility": ["تمشي", "مشي", "كرسي", "مساعدة", "walk", "walks", "wheelchair", "mobility"],
    "pain": ["ألم", "وجع", "مؤلم", "pain", "hurt", "hurts"],
    "mood": ["مزاج", "حزينة", "متفائلة", "مزاج", "mood", "sad", "optimistic", "withdrawn"],
    "toileting": ["حمام", "الحمام", "دخول الحمام", "bathroom", "toilet"],
    "hydration": ["ماء", "شربت", "ماء كاف", "water", "drank", "thirsty"],
}

UNCERTAINTY_MARKERS = [
    "مش متأكد", "غير متأكد", "لا ندري", "غير واضحة", "لا نعلم", "ربما", "ممكن",
    "unsure", "uncertain", "unknown", "maybe", "possibly", "not sure", "unclear",
]

# Opposing normalized sentence starts used to detect contradictions
OPPOSITE_STARTS = [
    ("أكلت", "لم تأكل"),
    ("نمت", "لم تنم"),
    ("تمشي", "تحتاج مساعدة"),
    ("لا تشكو", "تشكو"),
    ("شربت", "لم تشرب"),
]


def _category_score(sentence: str) -> tuple[str, int]:
    norm = normalize(sentence)
    best_cat, best_score = "general", 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if normalize(kw) in norm)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat, best_score


def _has_uncertainty(sentence: str) -> bool:
    norm = normalize(sentence)
    return any(normalize(m) in norm for m in UNCERTAINTY_MARKERS)


def _detect_contradictions(note: str) -> List[str]:
    norm = normalize(note)
    flags: List[str] = []
    for a, b in OPPOSITE_STARTS:
        na = normalize(a)
        nb = normalize(b)
        if na in norm and nb in norm:
            flags.append(f"contradiction between '{a}' and '{b}'")
    return flags


def _fallback_structured() -> Dict:
    return CareSummary(
        summary="Unable to extract observations from the note.",
        observations=[],
        unclear_or_conflicting=[],
        unsupported_claims=[],
        valid_json=True,
    ).model_dump()


class RuleBasedBaseline(ModelInterface):
    def __init__(self, model_id: str = BASELINE_MODEL_ID):
        self.model_id = model_id

    def name(self) -> str:
        return self.model_id

    def available(self) -> bool:
        return True

    def predict(self, note: str, **kwargs) -> ModelResult:
        start = __import__("time").time()
        config: Dict[str, Any] = {"approach": "extractive-rule-based", "top_k": 6}
        try:
            sentences = [s for s in split_sentences(note) if s.strip()]
            if not sentences:
                structured = CareSummary(
                    summary="Empty note.",
                    observations=[],
                    unclear_or_conflicting=[],
                    unsupported_claims=[],
                    model_id=self.model_id,
                    valid_json=True,
                ).model_dump()
                latency = __import__("time").time() - start
                return ModelResult(
                    model_id=self.model_id,
                    input_note=note,
                    generation_config=config,
                    summary="Empty note.",
                    structured=structured,
                    evidence_quotes=[],
                    latency_s=round(latency, 4),
                    ok=True,
                    valid_json=True,
                )
            scored = []
            for s in sentences:
                cat, score = _category_score(s)
                scored.append((score, s, cat))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score = scored[0][0]
            if best_score <= 0:
                candidates = scored
            else:
                candidates = [item for item in scored if item[0] > 0]
            top = candidates[: kwargs.get("top_k", 6)]

            observations: List[Observation] = []
            for score, sentence, cat in top:
                observations.append(
                    Observation(
                        category=cat,
                        statement=sentence,
                        evidence_quote=sentence,
                        uncertain=_has_uncertainty(sentence),
                    )
                )

            contradictions = _detect_contradictions(note)
            unclear = [
                UnclearOrConflicting(statement=c, evidence_quote=sentence)
                for c, sentence in _contradiction_sentences(note, contradictions)
            ]

            summary_parts = [o.statement for o in observations] or [note.strip() or "No content provided."]
            summary = " ".join(summary_parts)

            structured = CareSummary(
                summary=summary,
                observations=observations,
                unclear_or_conflicting=unclear,
                unsupported_claims=[],
                model_id=self.model_id,
                valid_json=True,
            ).model_dump()

            latency = __import__("time").time() - start
            return ModelResult(
                model_id=self.model_id,
                input_note=note,
                generation_config=config,
                summary=summary,
                structured=structured,
                evidence_quotes=[o.evidence_quote for o in observations],
                latency_s=round(latency, 4),
                ok=True,
                valid_json=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency = __import__("time").time() - start
            return ModelResult(
                model_id=self.model_id,
                input_note=note,
                generation_config=config,
                summary="",
                structured=_fallback_structured(),
                evidence_quotes=[],
                latency_s=round(latency, 4),
                ok=False,
                error=str(exc),
                valid_json=True,
            )


def _contradiction_sentences(note: str, contradictions: List[str]) -> List[tuple[str, str]]:
    """Return (contradiction text, evidence sentence) pairs for unclear list."""
    out: List[tuple[str, str]] = []
    sentences = split_sentences(note)
    norm_sentences = [normalize(s) for s in sentences]
    for c in contradictions:
        for s, ns in zip(sentences, norm_sentences):
            keys = c.replace("contradiction between '", "").split("' and '")
            if any(normalize(k) in ns for k in keys):
                out.append((c, s))
                break
    return out
