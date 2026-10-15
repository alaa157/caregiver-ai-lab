"""Common model interface. Every adapter returns the same normalized shape so
the rest of the pipeline (evaluation, UI, reports) never depends on one model.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelResult:
    model_id: str
    input_note: str
    generation_config: Dict[str, Any]
    summary: str
    structured: Dict[str, Any] = field(default_factory=dict)
    evidence_quotes: List[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: Optional[str] = None
    ok: bool = True
    valid_json: bool = True

    def to_record(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "input_note": self.input_note,
            "generation_config": self.generation_config,
            "summary": self.summary,
            "structured": self.structured,
            "evidence_quotes": self.evidence_quotes,
            "latency_s": self.latency_s,
            "error": self.error,
            "ok": self.ok,
            "valid_json": self.valid_json,
        }


class ModelInterface(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def available(self) -> bool:
        """True when the model can be loaded/run in this environment."""
        ...

    @abstractmethod
    def predict(self, note: str, **kwargs) -> ModelResult:
        """Run one note through the model. Never raises; failures are recorded
        in the returned ModelResult so the app keeps working."""
        ...

    def health_check(self) -> Dict[str, Any]:
        return {"model_id": self.name(), "available": self.available()}
