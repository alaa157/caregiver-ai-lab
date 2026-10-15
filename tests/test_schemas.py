"""Tests for Pydantic structured-output schema."""
from src.schemas import CareSummary, Observation, UnsupportedClaim


def test_valid_summary_parses():
    data = {
        "summary": "المريض تناول الدواء.",
        "observations": [
            {"category": "medication", "statement": "تناول الدواء", "evidence_quote": "أخدت الدواء", "uncertain": False}
        ],
        "unclear_or_conflicting": [],
        "unsupported_claims": [],
    }
    obj = CareSummary(**data)
    assert obj.summary == "المريض تناول الدواء."
    assert obj.valid_json is True


def test_missing_summary_raises():
    try:
        CareSummary(**{"observations": []})
    except Exception:
        pass
    else:
        raise AssertionError("expected ValidationError")


def test_to_flat_text_nonempty():
    obj = CareSummary(summary="أم خالد أكلت قليلاً", observations=[])
    flat = obj.to_flat_text()
    assert "أم خالد" in flat


def test_unsupported_claim_default_reason():
    claim = UnsupportedClaim(statement="أخذت دواء X")
    assert claim.reason == "no supporting evidence in the input note"
