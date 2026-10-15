"""Evaluation metrics. All metric computations are real (no fabricated
numbers). Heuristics are explicitly labeled as such.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.evaluation.evidence import quote_found_in_note
from src.evaluation.uncertainty import has_uncertainty_marker
from src.normalization import word_tokens
from src.schemas import CareSummary


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def json_valid_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("valid_json") is True) / len(results)


def schema_compliance_rate(results: List[Dict[str, Any]]) -> float:
    """Share of results whose structured output validates against CareSummary."""
    if not results:
        return 0.0
    ok = 0
    for r in results:
        try:
            CareSummary(**r.get("structured", {}))
            ok += 1
        except Exception:
            pass
    return ok / len(results)


def _output_text(result: Dict[str, Any]) -> str:
    structured = result.get("structured", {}) or {}
    parts = [structured.get("summary", ""), result.get("summary", "")]
    for obs in structured.get("observations", []) or []:
        parts.extend([obs.get("statement", ""), obs.get("evidence_quote", "")])
    for c in structured.get("unclear_or_conflicting", []) or []:
        parts.extend([c.get("statement", ""), c.get("evidence_quote", "")])
    return " ".join(p for p in parts if p)


def _fact_recall_example(expected_facts: List[str], output_text: str) -> Optional[float]:
    """Heuristic fact recall: share of expected facts whose content words
    appear in the output text. Not a definitive correctness measure."""
    if not expected_facts:
        return None
    norm_output = output_text.lower()
    hits = 0
    for fact in expected_facts:
        words = word_tokens(fact)
        if not words:
            continue
        if all(w in norm_output for w in words):
            hits += 1
    return hits / len(expected_facts)


def fact_recall_rate(results: List[Dict[str, Any]], examples_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    recalls: List[float] = []
    omitted: List[str] = []
    for r in results:
        ex = examples_by_id.get(r.get("example_id", ""))
        facts = ex.get("expected_facts", []) if ex else []
        score = _fact_recall_example(facts, _output_text(r))
        if score is None:
            continue
        recalls.append(score)
        if score < 1.0 and facts:
            omitted.append(r.get("example_id", ""))
    return {
        "metric": "fact_recall_heuristic",
        "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "examples_evaluated": len(recalls),
        "omitted_fact_examples": omitted[:20],
    }


def unsupported_claim_rate(results: List[Dict[str, Any]], examples_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic: share of structured output claims that lack any evidence
    quote traceable to the input note."""
    if not results:
        return {"metric": "unsupported_claim_rate_heuristic", "rate": 0.0, "examples_evaluated": 0}
    total_claims = 0
    unsupported = 0
    flagged_ids: List[str] = []
    for r in results:
        ex = examples_by_id.get(r.get("example_id", ""))
        note = ex.get("input_note", "") if ex else r.get("input_note", "")
        structured = r.get("structured", {}) or {}
        claims: List[str] = []
        claims.append(structured.get("summary", ""))
        for obs in structured.get("observations", []) or []:
            claims.append(obs.get("statement", ""))
        for u in structured.get("unsupported_claims", []) or []:
            claims.append(u.get("statement", ""))
        for c in claims:
            if not c.strip():
                continue
            total_claims += 1
            ev = None
            for obs in structured.get("observations", []) or []:
                if obs.get("statement") == c:
                    ev = obs.get("evidence_quote", "")
                    break
            supported = bool(ev) and quote_found_in_note(ev, note)
            if not supported:
                unsupported += 1
                flagged_ids.append(r.get("example_id", ""))
    rate = unsupported / total_claims if total_claims else 0.0
    return {
        "metric": "unsupported_claim_rate_heuristic",
        "rate": round(rate, 4),
        "unsupported_claims": unsupported,
        "total_claims": total_claims,
        "examples_evaluated": len(results),
        "flagged_example_ids": flagged_ids[:20],
    }


def uncertainty_preservation_rate(results: List[Dict[str, Any]], examples_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Share of examples with reference uncertainties where the model output
    preserves uncertainty (marks observations unclear or lists conflicting items)."""
    total = 0
    preserved = 0
    for r in results:
        ex = examples_by_id.get(r.get("example_id", ""))
        refs = ex.get("uncertainties", []) if ex else []
        if not refs:
            continue
        total += 1
        structured = r.get("structured", {}) or {}
        structured_text = _output_text(r)
        flags = bool(structured.get("unclear_or_conflicting")) or any(
            o.get("uncertain") for o in structured.get("observations", []) or []
        )
        if flags or has_uncertainty_marker(structured_text):
            preserved += 1
    return {
        "metric": "uncertainty_preservation_rate",
        "rate": round(preserved / total, 4) if total else 0.0,
        "examples_with_uncertainty": total,
        "preserved": preserved,
    }


def contradiction_preservation_rate(results: List[Dict[str, Any]], examples_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    preserved = 0
    for r in results:
        ex = examples_by_id.get(r.get("example_id", ""))
        contr = ex.get("known_contradictions", []) if ex else []
        if not contr:
            continue
        total += 1
        structured_text = _output_text(r)
        has_unclear = bool((r.get("structured") or {}).get("unclear_or_conflicting"))
        if has_unclear or has_uncertainty_marker(structured_text):
            preserved += 1
    return {
        "metric": "contradiction_preservation_rate",
        "rate": round(preserved / total, 4) if total else 0.0,
        "examples_with_contradictions": total,
        "preserved": preserved,
    }


def latency_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    lats = [r.get("latency_s", 0.0) for r in results if r.get("latency_s") is not None]
    if not lats:
        return {"metric": "latency", "count": 0}
    import statistics

    return {
        "metric": "latency",
        "count": len(lats),
        "mean_s": round(sum(lats) / len(lats), 4),
        "max_s": round(max(lats), 4),
        "min_s": round(min(lats), 4),
    }


def failure_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"metric": "failure_rate", "rate": 0.0, "count": 0}
    fails = sum(1 for r in results if not r.get("ok", True) or r.get("error"))
    return {"metric": "failure_rate", "rate": round(fails / len(results), 4), "failures": fails, "count": len(results)}


def evaluate_all(results: List[Dict[str, Any]], examples_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "json_valid_rate": round(json_valid_rate(results), 4),
        "schema_compliance_rate": round(schema_compliance_rate(results), 4),
        "fact_recall": fact_recall_rate(results, examples_by_id),
        "unsupported_claims": unsupported_claim_rate(results, examples_by_id),
        "uncertainty_preservation": uncertainty_preservation_rate(results, examples_by_id),
        "contradiction_preservation": contradiction_preservation_rate(results, examples_by_id),
        "latency": latency_stats(results),
        "failure_rate": failure_rate(results),
    }
