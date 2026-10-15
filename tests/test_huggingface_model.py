"""Tests for the Hugging Face model adapter — fully mocked, no network."""
from unittest import mock

import pytest

from src.models.huggingface_model import HfGenerativeModel, _parse_structured, _extract_json_block


def test_extract_json_block():
    assert _extract_json_block("summary: {\"summary\": \"x\"}") == '{"summary": "x"}'
    assert _extract_json_block("no json here") is None


def test_parse_structured_valid():
    structured, valid, err = _parse_structured('{"summary": "ate well", "observations": []}')
    assert valid is True
    assert err is None
    assert structured["summary"] == "ate well"


def test_parse_structured_invalid():
    structured, valid, err = _parse_structured("not json at all")
    assert valid is False
    assert err is not None


@pytest.fixture
def fake_pipeline():
    def fake(text, max_new_tokens=256, truncation=True):
        return [{"generated_text": '{"summary": "mocked summary", "observations": [{"category": "appetite", "statement": "ate well", "evidence_quote": "ate well", "uncertain": false}], "unclear_or_conflicting": [], "unsupported_claims": []}'}]
    return fake


def test_predict_mocked_success(fake_pipeline):
    m = HfGenerativeModel(model_id="fake/model-id")
    m._pipe = fake_pipeline
    m._tried_load = True
    res = m.predict("أم خالد أكلت جيداً.")
    assert res.ok is True
    assert res.valid_json is True
    assert res.summary == "mocked summary"
    assert res.structured.get("summary") == "mocked summary"


def test_predict_invalid_json_from_model():
    def bad_pipe(text, max_new_tokens=256, truncation=True):
        return [{"generated_text": "this is just prose, no json braces"}]
    m = HfGenerativeModel(model_id="fake/model-id")
    m._pipe = bad_pipe
    m._tried_load = True
    res = m.predict("أم خالد أكلت جيداً.")
    assert res.ok is True
    assert res.valid_json is False
    assert res.error is not None


def test_predict_load_failure_graceful():
    m = HfGenerativeModel(model_id="definitely/not/a/real/model-id-xyz")
    m._pipe = None
    m._tried_load = True
    m._load_error = "simulated load failure"
    res = m.predict("some note")
    assert res.ok is False
    assert res.valid_json is False
    assert res.error == "simulated load failure"
