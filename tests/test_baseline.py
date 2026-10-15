"""Tests for the rule-based baseline (no model, no network)."""
from src.models.baseline import RuleBasedBaseline


def test_baseline_always_available():
    m = RuleBasedBaseline()
    assert m.available() is True


def test_baseline_predict_returns_result():
    m = RuleBasedBaseline()
    res = m.predict("أم خالد أكلت جيداً ونمت جيداً.")
    assert res.ok is True
    assert res.valid_json is True
    assert res.error is None
    assert res.summary != ""
    assert res.latency_s >= 0


def test_baseline_empty_note_no_crash():
    m = RuleBasedBaseline()
    res = m.predict("")
    assert res.ok is True


def test_baseline_uncertainty_marked():
    m = RuleBasedBaseline()
    res = m.predict("غير متأكدين إذا كانت أكلت أم لا.")
    obs = res.structured.get("observations", [])
    uncertain = [o for o in obs if o.get("uncertain")]
    assert len(uncertain) >= 1


def test_baseline_contradiction_detected():
    m = RuleBasedBaseline()
    note = "أكلت جيدًا. لم تأكل شيئًا."
    res = m.predict(note)
    unclear = res.structured.get("unclear_or_conflicting", [])
    assert len(unclear) >= 1


def test_baseline_deterministic():
    m = RuleBasedBaseline()
    note = "المريض تناول الدواء في الموعد ونم جيداً."
    r1 = m.predict(note)
    r2 = m.predict(note)
    assert r1.summary == r2.summary
