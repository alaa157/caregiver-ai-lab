"""Tests for evaluation metrics with deterministic inputs."""
from src.evaluation.metrics import (
    json_valid_rate,
    schema_compliance_rate,
    fact_recall_rate,
    unsupported_claim_rate,
    uncertainty_preservation_rate,
    evaluate_all,
    failure_rate,
)


def _result(ok=True, valid_json=True, example_id="ex-1", structured=None, latency=0.1, error=None):
    return {
        "example_id": example_id,
        "model_id": "test-model",
        "ok": ok,
        "valid_json": valid_json,
        "error": error,
        "latency_s": latency,
        "structured": structured or {},
    }


def test_json_valid_rate():
    results = [
        _result(valid_json=True),
        _result(valid_json=True),
        _result(valid_json=False),
    ]
    assert json_valid_rate(results) == 2 / 3


def test_schema_compliance_rate():
    good = _result(structured={"summary": "x", "observations": []})
    bad = _result(structured={"not-a-summary": 1})
    assert schema_compliance_rate([good, bad]) == 0.5


def test_fact_recall_known():
    results = [
        _result(
            structured={
                "summary": "the patient ate well and slept all night",
                "observations": [
                    {"statement": "ate well", "evidence_quote": "ate well"},
                    {"statement": "slept well", "evidence_quote": "slept well"},
                ],
            }
        )
    ]
    examples = {"ex-1": {"expected_facts": ["ate well", "slept well"], "uncertainties": []}}
    m = fact_recall_rate(results, examples)
    assert m["mean_recall"] == 1.0


def test_fact_recall_omits():
    results = [_result(structured={"summary": "nothing relevant", "observations": []})]
    examples = {"ex-1": {"expected_facts": ["took medication"], "uncertainties": []}}
    m = fact_recall_rate(results, examples)
    assert m["mean_recall"] == 0.0
    assert "ex-1" in m["omitted_fact_examples"]


def test_unsupported_claim_rate():
    results = [
        _result(
            example_id="ex-1",
            structured={
                "summary": "the patient walked alone",
                "observations": [
                    {"statement": "walked alone", "evidence_quote": "walks alone inside the house"}
                ],
            },
        )
    ]
    examples = {"ex-1": {"input_note": "the patient walks alone inside the house", "uncertainties": []}}
    m = unsupported_claim_rate(results, examples)
    assert m["rate"] == 0.5


def test_unsupported_claim_rate_missing_evidence():
    results = [
        _result(
            example_id="ex-1",
            structured={
                "summary": "the patient took medicine",
                "observations": [
                    {"statement": "took medicine", "evidence_quote": "took medicine"}
                ],
            },
        )
    ]
    examples = {"ex-1": {"input_note": "the patient slept well", "uncertainties": []}}
    m = unsupported_claim_rate(results, examples)
    assert m["rate"] == 1.0


def test_unsupported_claim_rate_missing_evidence():
    results = [
        _result(
            example_id="ex-1",
            structured={
                "summary": "the patient took medicine",
                "observations": [
                    {"statement": "took medicine", "evidence_quote": ""}
                ],
            },
        )
    ]
    examples = {"ex-1": {"input_note": "the patient slept well", "uncertainties": []}}
    m = unsupported_claim_rate(results, examples)
    assert m["rate"] == 1.0


def test_uncertainty_preservation():
    results = [
        _result(structured={"summary": "unsure", "observations": [{"statement": "x", "uncertain": True}]}),
        _result(structured={"summary": "all good", "observations": []}),
    ]
    examples = {
        "ex-1": {"uncertainties": ["not sure if ate"]},
        "ex-2": {"uncertainties": ["not sure if ate"]},
    }
    m = uncertainty_preservation_rate(results, examples)
    assert m["rate"] == 0.5
    assert m["examples_with_uncertainty"] == 2


def test_failure_rate():
    results = [_result(ok=True), _result(ok=False, error="boom")]
    m = failure_rate(results)
    assert m["rate"] == 0.5
    assert m["failures"] == 1


def test_evaluate_all_keys():
    results = [_result(structured={"summary": "ate well", "observations": [{"statement": "ate well", "evidence_quote": "ate well"}]}), ]
    examples = {"ex-1": {"expected_facts": ["ate well"], "uncertainties": [], "input_note": "ate well"}}
    out = evaluate_all(results, examples)
    for key in ["json_valid_rate", "schema_compliance_rate", "fact_recall", "unsupported_claims", "uncertainty_preservation", "contradiction_preservation", "latency", "failure_rate"]:
        assert key in out
