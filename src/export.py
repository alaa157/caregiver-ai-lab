"""Export helpers for experiment results and human review."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_results_jsonl(results: List[Dict[str, Any]], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_review_csv(results: List[Dict[str, Any]], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id", "model_id", "input_note", "summary",
        "valid_json", "latency_s", "error", "human_label", "human_notes",
    ]
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            structured = row.get("structured", {}) or {}
            writer.writerow(
                {
                    "example_id": row.get("example_id", ""),
                    "model_id": row.get("model_id", ""),
                    "input_note": row.get("input_note", ""),
                    "summary": structured.get("summary", row.get("summary", "")),
                    "valid_json": row.get("valid_json", ""),
                    "latency_s": row.get("latency_s", ""),
                    "error": row.get("error", ""),
                    "human_label": row.get("human_label", ""),
                    "human_notes": row.get("human_notes", ""),
                }
            )
