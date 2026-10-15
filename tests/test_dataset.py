"""Tests for dataset generation, loading, and validation."""
import tempfile
import os
from src.dataset import generate_dataset, load_dataset, write_dataset, validate_dataset, DATASET_PATH


def test_generate_dataset_has_unique_ids():
    examples = generate_dataset(10)
    assert len(examples) == 10
    ids = [e["example_id"] for e in examples]
    assert len(set(ids)) == 10


def test_generate_dataset_covers_languages():
    examples = generate_dataset(30)
    langs = {e["language_type"] for e in examples}
    assert {"ar", "mix", "en"} <= langs


def test_validate_dataset_valid():
    examples = generate_dataset(20)
    report = validate_dataset(examples)
    assert report["valid"] is True
    assert report["count"] == 20


def test_validate_dataset_catches_duplicates():
    examples = generate_dataset(5)
    dup = dict(examples[0])
    examples.append(dup)
    report = validate_dataset(examples)
    assert report["valid"] is False
    assert report["duplicate_ids"] is True


def test_validate_dataset_catches_missing_fields():
    examples = generate_dataset(5)
    bad = dict(examples[0])
    del bad["expected_facts"]
    examples.append(bad)
    report = validate_dataset(examples)
    assert report["valid"] is False
    assert any("expected_facts" in e for e in report["errors"])


def test_write_and_load_roundtrip(tmp_path):
    examples = generate_dataset(5)
    p = tmp_path / "test.jsonl"
    write_dataset(examples, p)
    loaded = load_dataset(p)
    assert loaded == examples


def test_validate_empty_dataset():
    assert validate_dataset([])["valid"] is True
    assert validate_dataset([])["count"] == 0
