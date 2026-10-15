"""Synthetic Egyptian-Arabic care-note dataset.

All examples are fictional and generated for engineering practice only.
They do NOT represent real caregivers, patients, or clinical populations.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import DATASET_PATH, SYNTHETIC_DATASET_VERSION, DATASET_LICENSE_NOTE

SEED = 42
LANGUAGE_TYPES = ("ar", "mix", "en")

REQUIRED_FIELDS = [
    "example_id",
    "language_type",
    "input_note",
    "expected_facts",
    "important_details",
    "uncertainties",
    "known_contradictions",
    "expected_summary",
    "expected_evidence_quotes",
]

# Each clause: (arabic text, english text, normalized fact, category, uncertain)
CLAUSES: Dict[str, List[Tuple[str, str, str, str, bool]]] = {
    "appetite": [
        ("أكلت وجبة الغداء جيداً", "ate lunch well", "ate lunch well", "appetite", False),
        ("أكلت كمية قليلة جداً", "ate a very small amount", "ate small amount", "appetite", False),
        ("لم تأكل شيئًا منذ الصباح", "has not eaten anything since morning", "not eaten since morning", "appetite", False),
        ("غير متأكدين إذا كانت أكلت أم لا", "unsure whether she ate or not", "uncertain whether ate", "appetite", True),
    ],
    "medication": [
        ("أخدت الدواء في الموعد", "took the medication on time", "medication taken on time", "medication", False),
        ("لم تتناول الدواء بعد", "has not taken the medication yet", "medication not taken", "medication", False),
        ("مش متأكدين إذا أخذت الدواء ولا لأ", "unsure if she took the medication or not", "uncertain medication intake", "medication", True),
        ("المedicine ... skip".replace("medicine ... skip", "الدواء موجود في الحقيبة"), "the medication is in the bag", "medication present in bag", "medication", False),
    ],
    "sleep": [
        ("نمت جيداً طوال الليل", "slept well all night", "slept well all night", "sleep", False),
        ("النوم كان متقطعاً", "sleep was interrupted", "sleep interrupted", "sleep", False),
        ("لم تنم جيداً أمس", "did not sleep well yesterday", "did not sleep well", "sleep", False),
        ("غير متأكدين إذا نمت أو سهرت", "unsure whether she slept or stayed awake", "uncertain sleep", "sleep", True),
    ],
    "mobility": [
        ("تمشي بمفردها داخل البيت", "walks alone inside the house", "walks alone inside house", "mobility", False),
        ("تحتاج مساعدة للمشي", "needs assistance to walk", "needs assistance walking", "mobility", False),
        ("تستخدم كرسي متحرك", "uses a wheelchair", "uses wheelchair", "mobility", False),
        ("مش متأكدين إذا تقدر تمشي أم لأ", "unsure whether she can walk or not", "uncertain mobility", "mobility", True),
    ],
    "pain": [
        ("تشكو من ألم في الركبة", "complains of knee pain", "knee pain", "pain", False),
        ("لا تشكو من أي ألم", "does not complain of any pain", "no pain", "pain", False),
        ("ألم خفيف في البطن يأتي ويذهب", "mild stomach pain that comes and goes", "mild intermittent stomach pain", "pain", False),
        ("غير متأكدين إذا كانت تعاني من ألم", "unsure whether she is in pain", "uncertain pain", "pain", True),
    ],
    "mood": [
        ("مزاجها جيد ومتفائلة", "mood is good and she is optimistic", "good mood", "mood", False),
        ("تبدو حزينة ومقفلة", "appears sad and withdrawn", "sad and withdrawn", "mood", False),
        ("متقلبة المزاج اليوم", "mood is changeable today", "changeable mood", "mood", False),
    ],
    "toileting": [
        ("استخدمت الحمام بشكل طبيعي", "used the bathroom normally", "used bathroom normally", "toileting", False),
        ("تحتاج مساعدة لدخول الحمام", "needs assistance to use the bathroom", "needs bathroom assistance", "toileting", False),
        ("لم تذهب للحمام منذ الصباح", "has not gone to the bathroom since morning", "no bathroom visit since morning", "toileting", False),
    ],
    "hydration": [
        ("شربت ماءً كافٍ اليوم", "drank enough water today", "drank enough water", "hydration", False),
        ("شربت كمية قليلة من الماء", "drank a small amount of water", "drank little water", "hydration", False),
        ("غير متأكدين إذا شربت الماء أم لا", "unsure whether she drank water or not", "uncertain water intake", "hydration", True),
    ],
}

IRRELEVANT_CLAUSES = [
    ("الطقس معتدل اليوم", "the weather is mild today", "weather mild", "irrelevant", False),
    ("البيت نظيف ومرتب", "the house is clean and tidy", "house clean", "irrelevant", False),
    ("شاهدت التلفزيون مساء أمس", "watched television last night", "watched tv", "irrelevant", False),
]

NAMES_AR = ["أم خالد", "سارة", "جدّي", "ماما", "الأخت منى", "عم يوسف", "التيتو", "خالتي", "خالد", "نورا"]
NAMES_EN = ["Um Khalid", "Sara", "Grandpa", "Mama", "Aunt Mona", "Uncle Youssef", "Teta", "Khalti", "Khalid", "Nora"]

SUMMARY_TEMPLATES_AR = [
    "{name}: {facts}.",
    "ملخص حالة {name}: {facts}.",
    "{name} case summary: {facts}.",
]
SUMMARY_TEMPLATES_EN = [
    "{name}: {facts}.",
    "Care summary for {name}: {facts}.",
    "{name} note summary: {facts}.",
]

INTRO_AR = "ملاحظة رعاية يومية خيالية لـ {name}."
INTRO_EN = "Fictional daily care note for {name}."


def _rng_for(example_id: str) -> random.Random:
    return random.Random(hash(example_id) & 0xFFFFFFFF)


def _pick(clauses_pool: List[Tuple], rng: random.Random, k: int) -> List[Tuple]:
    return rng.sample(clauses_pool, min(k, len(clauses_pool)))


def _join(texts: List[str], conj_ar: str = " و ", conj_en: str = " and ") -> Tuple[str, str]:
    ar = conj_ar.join(t[0] for t in texts)
    en = conj_en.join(t[1] for t in texts)
    return ar, en


def _contradiction_pair(category: str) -> Optional[Tuple[Tuple, Tuple]]:
    opposites = {
        "appetite": ("أكلت جيدًا", "لم تأكل شيئًا"),
        "sleep": ("نمت جيدًا", "لم تنم جيدًا"),
        "mobility": ("تمشي بمفردها", "تحتاج مساعدة للمشي"),
        "pain": ("لا تشكو من أي ألم", "تشكو من ألم"),
        "hydration": ("شربت ماءً كافٍ", "لم تشرب ماءً"),
    }
    if category not in opposites:
        return None
    a_text, b_text = opposites[category]
    ar_a = next(c[0] for c in CLAUSES[category] if c[0].startswith(a_text.split(" ")[0]))
    ar_b = next(c[0] for c in CLAUSES[category] if not c[0].startswith(a_text.split(" ")[0]))
    return (ar_a, ar_b)


def generate_example(idx: int) -> Dict[str, Any]:
    rng = _rng_for(f"EG-SYN-{idx:04d}")
    lang = LANGUAGE_TYPES[idx % 3]
    name_ar = rng.choice(NAMES_AR)
    name_en = rng.choice(NAMES_EN)

    categories = rng.sample(list(CLAUSES.keys()), k=rng.randint(2, 4))
    chosen: List[Tuple] = []
    for cat in categories:
        chosen.append(rng.choice(CLAUSES[cat]))

    contradictions: List[Tuple[str, str]] = []
    if rng.random() < 0.25:
        pair = _contradiction_pair(rng.choice(list(CLAUSES.keys())))
        if pair:
            chosen.extend(list(pair))
            contradictions.append((pair[0][0], pair[1][0]))

    uncertain_clauses: List[str] = []
    if rng.random() < 0.35:
        for c in chosen[:]:
            if c[4]:
                uncertain_clauses.append(c[0])

    irrelevant: List[Tuple] = []
    if rng.random() < 0.4:
        irrelevant = _pick(IRRELEVANT_CLAUSES, rng, rng.randint(1, 2))

    care_texts_ar = [c[0] for c in chosen]
    care_texts_en = [c[1] for c in chosen]
    facts = [c[2] for c in chosen]

    note_ar = f"{INTRO_AR.format(name=name_ar)} " + " ".join(care_texts_ar)
    note_en = f"{INTRO_EN.format(name=name_en)} " + " ".join(care_texts_en)
    if irrelevant:
        irr = irrelevant[0]
        note_ar += " " + irr[0]
        note_en += " " + irr[1]

    if lang == "ar":
        input_note = note_ar
    elif lang == "en":
        input_note = note_en
    else:
        half = max(1, len(care_texts_ar) // 2)
        input_note = " ".join(care_texts_ar[:half]) + " " + " ".join(care_texts_en[half:]) + "."

    important_details = [f for f in facts if not f.startswith(("weather", "house", "watched"))]
    uncertainties = uncertain_clauses[:]
    expected_evidence_quotes = [c[0] for c in chosen]
    if irrelevant:
        expected_evidence_quotes.extend([c[0] for c in irrelevant])

    fact_list_ar = "، ".join(care_texts_ar) if care_texts_ar else "لا توجد ملاحظات واضحة"
    fact_list_en = "; ".join(care_texts_en) if care_texts_en else "no clear observations"
    if lang == "ar":
        template = rng.choice(SUMMARY_TEMPLATES_AR)
        expected_summary = template.format(name=name_ar, facts=fact_list_ar)
    elif lang == "en":
        template = rng.choice(SUMMARY_TEMPLATES_EN)
        expected_summary = template.format(name=name_en, facts=fact_list_en)
    else:
        expected_summary = f"{name_en} / {name_ar}: {fact_list_en}."

    # Hallucination trap: ~10% of examples contain no medication information at all.
    has_medication = any(c[3] == "medication" for c in chosen)

    return {
        "example_id": f"EG-SYN-{idx:04d}",
        "language_type": lang,
        "input_note": input_note,
        "expected_facts": facts,
        "important_details": important_details,
        "uncertainties": uncertainties,
        "known_contradictions": contradictions,
        "expected_summary": expected_summary,
        "expected_evidence_quotes": expected_evidence_quotes,
        "_synthetic": True,
        "_dataset_version": SYNTHETIC_DATASET_VERSION,
        "_has_medication_info": has_medication,
    }


def generate_dataset(n: int = 150) -> List[Dict[str, Any]]:
    return [generate_example(i) for i in range(1, n + 1)]


def write_dataset(examples: List[Dict[str, Any]], path: Path = DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_dataset(path: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def validate_dataset(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    seen_ids: set = set()
    malformed = 0
    for idx, ex in enumerate(examples):
        label = ex.get("example_id", f"row-{idx}")
        if not isinstance(ex, dict):
            malformed += 1
            errors.append(f"{label}: entry is not a JSON object")
            continue
        for field in REQUIRED_FIELDS:
            value = ex.get(field)
            if value is None or value == "":
                errors.append(f"{label}: missing or empty field '{field}'")
            if field == "expected_facts" and not isinstance(value, list):
                errors.append(f"{label}: 'expected_facts' must be a list")
            if field == "language_type" and value not in LANGUAGE_TYPES:
                errors.append(f"{label}: invalid language_type '{value}'")
        eid = ex.get("example_id")
        if eid:
            if eid in seen_ids:
                errors.append(f"duplicate example_id: {eid}")
            seen_ids.add(eid)
        if not ex.get("input_note"):
            malformed += 1
    return {
        "valid": len(errors) == 0 and malformed == 0,
        "count": len(examples),
        "unique_ids": len(seen_ids),
        "duplicate_ids": len(seen_ids) != len(examples),
        "malformed": malformed,
        "errors": errors[:50],
    }
