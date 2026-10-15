"""Hugging Face Transformers adapter for a generative (seq2seq / causal) model.

The model ID is configurable via CAREGIVER_MODEL_ID. Loading failures,
generation failures, and invalid JSON are all captured as error statuses so the
application keeps running without the model.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import torch

from src.config import DEFAULT_MODEL_ID, HF_TOKEN, MAX_INPUT_CHARS, MAX_NEW_TOKENS
from src.models.base import ModelInterface, ModelResult
from src.normalization import normalize, split_sentences, word_tokens
from src.schemas import CareSummary

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from generated text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_structured(raw: str) -> tuple[Dict[str, Any], bool, Optional[str]]:
    """Try to validate generated text as CareSummary JSON. Never raises."""
    block = _extract_json_block(raw)
    if block is None:
        return {"raw_output": raw.strip()}, False, "no JSON object found in model output"
    try:
        data = json.loads(block)
        validated = CareSummary(**data)
        return validated.model_dump(), True, None
    except Exception as exc:  # noqa: BLE001 - report as parse failure
        return {"raw_output": raw.strip(), "parse_detail": block[:200]}, False, str(exc)


class HfGenerativeModel(ModelInterface):
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self._pipe: Any = None
        self._load_error: Optional[str] = None
        self._tried_load = False

    def name(self) -> str:
        return self.model_id

    def available(self) -> bool:
        if self._pipe is not None:
            return True
        if self._tried_load and self._load_error:
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  # noqa: WPS433

            self._tried_load = True
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, token=HF_TOKEN)
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self._tried_load = True
            self._load_error = str(exc)
            return False

    def predict(self, note: str, **kwargs) -> ModelResult:
        start = time.time()
        config: Dict[str, Any] = {
            "model_id": self.model_id,
            "max_new_tokens": kwargs.get("max_new_tokens", MAX_NEW_TOKENS),
            "truncation": True,
        }
        try:
            if not self.available():
                raise RuntimeError(self._load_error or "model failed to load")

            text = note[:MAX_INPUT_CHARS]
            out = self._pipe(
                text,
                max_new_tokens=config["max_new_tokens"],
                truncation=True,
            )
            generated = ""
            if isinstance(out, list) and out:
                first = out[0]
                generated = first.get("summary_text") or first.get("generated_text") or str(first)
            generated = str(generated).strip()

            structured, valid_json, parse_error = _parse_structured(generated)

            summary = generated
            evidence: List[str] = []
            if isinstance(structured, dict):
                summary = structured.get("summary", generated) or generated
                obs = structured.get("observations", [])
                evidence = [o.get("evidence_quote", "") for o in obs if isinstance(o, dict)]

            latency = round(time.time() - start, 4)
            return ModelResult(
                model_id=self.model_id,
                input_note=note,
                generation_config=config,
                summary=summary,
                structured=structured,
                evidence_quotes=[e for e in evidence if e],
                latency_s=latency,
                ok=True,
                valid_json=valid_json,
                error=parse_error,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            latency = round(time.time() - start, 4)
            structured, _, _ = _parse_structured("")
            return ModelResult(
                model_id=self.model_id,
                input_note=note,
                generation_config=config,
                summary="",
                structured=structured,
                evidence_quotes=[],
                latency_s=latency,
                ok=False,
                error=str(exc),
                valid_json=False,
            )
