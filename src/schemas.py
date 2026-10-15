"""Validated JSON schema for structured care-note summaries.

Every model output is normalized into this shape (or a graceful fallback when
the model returns invalid JSON). The schema deliberately keeps uncertainty and
missing evidence explicit: uncertainty is never converted into certainty, and
missing evidence is marked unsupported rather than fabricated.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """A single care observation extracted from the note."""
    category: str = Field(..., min_length=1, description="e.g. appetite, medication, sleep")
    statement: str = Field(..., min_length=1, description="plain-language statement")
    evidence_quote: str = Field(default="", description="verbatim quote from the input note")
    uncertain: bool = Field(default=False, description="true when the note expresses doubt about this fact")


class UnclearOrConflicting(BaseModel):
    """A statement the note leaves unresolved or contradicts itself about."""
    statement: str = Field(..., min_length=1)
    evidence_quote: str = Field(default="")


class UnsupportedClaim(BaseModel):
    """A claim in the output that cannot be traced to the input note."""
    statement: str = Field(..., min_length=1)
    reason: str = Field(default="no supporting evidence in the input note")


class CareSummary(BaseModel):
    """Top-level structured summary returned by any model adapter."""
    summary: str = Field(..., min_length=1)
    observations: List[Observation] = Field(default_factory=list)
    unclear_or_conflicting: List[UnclearOrConflicting] = Field(default_factory=list)
    unsupported_claims: List[UnsupportedClaim] = Field(default_factory=list)
    model_id: Optional[str] = None
    latency_s: Optional[float] = None
    valid_json: bool = True
    parse_error: Optional[str] = None

    def to_flat_text(self) -> str:
        """Flatten the structure to a single searchable string (used by heuristics)."""
        parts = [self.summary]
        for o in self.observations:
            parts.extend([o.statement, o.evidence_quote])
        for c in self.unclear_or_conflicting:
            parts.extend([c.statement, c.evidence_quote])
        for u in self.unsupported_claims:
            parts.append(u.statement)
        return " ".join(p for p in parts if p)
