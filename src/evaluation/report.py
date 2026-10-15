"""Experiment report generation (Markdown + JSON)."""
from __future__ import annotations

from typing import Any, Dict, List

from src.config import SYNTHETIC_DATASET_VERSION


def generate_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    dataset_version: str,
    model_ids: List[str],
    config_snapshot: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Caregiver AI Lab — Experiment Report")
    lines.append("")
    lines.append(f"- Dataset version: {dataset_version}")
    lines.append(f"- Synthetic data notice: examples are fictional and for engineering practice only.")
    lines.append(f"- Models evaluated: {', '.join(model_ids)}")
    lines.append(f"- Examples evaluated: {len(results)}")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- Valid JSON rate: {metrics.get('json_valid_rate')}")
    lines.append(f"- Schema compliance rate: {metrics.get('schema_compliance_rate')}")
    fr = metrics.get("failure_rate", {})
    lines.append(f"- Failure rate: {fr.get('rate')} ({fr.get('failures')}/{fr.get('count')})")
    lat = metrics.get("latency", {})
    lines.append(f"- Mean latency: {lat.get('mean_s')} s ({lat.get('count')} examples)")
    fr_claims = metrics.get("unsupported_claims", {})
    lines.append(f"- Unsupported-claim rate (heuristic): {fr_claims.get('rate')}")
    fr_fact = metrics.get("fact_recall", {})
    lines.append(f"- Fact recall (heuristic): {fr_fact.get('mean_recall')}")
    unc = metrics.get("uncertainty_preservation", {})
    lines.append(f"- Uncertainty preservation: {unc.get('rate')}")
    con = metrics.get("contradiction_preservation", {})
    lines.append(f"- Contradiction preservation: {con.get('rate')}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Results are computed on a small synthetic dataset; they do not measure real-world clinical performance.")
    lines.append("- Fact recall and unsupported-claim rates are heuristics, not definitive correctness measures.")
    lines.append("- Do not use this tool for diagnosis or treatment decisions.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Config: {config_snapshot}")
    lines.append("")
    return "\n".join(lines)


def save_report(markdown: str, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown, encoding="utf-8")
