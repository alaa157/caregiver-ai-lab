"""Tests for export functions."""
import tempfile
from pathlib import Path
from src.export import export_results_jsonl, export_review_csv


def test_export_jsonl(tmp_path):
    p = tmp_path / "out.jsonl"
    rows = [{"example_id": "ex-1", "model_id": "m", "summary": "s", "valid_json": True}]
    export_results_jsonl(rows, p)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"ex-1"' in lines[0]


def test_export_csv(tmp_path):
    p = tmp_path / "out.csv"
    rows = [
        {
            "example_id": "ex-1",
            "model_id": "m",
            "input_note": "note",
            "summary": "s",
            "valid_json": True,
            "latency_s": 0.1,
            "error": "",
            "human_label": "Correct",
            "human_notes": "",
        }
    ]
    export_review_csv(rows, p)
    text = p.read_text(encoding="utf-8")
    assert "Correct" in text
    assert "ex-1" in text
