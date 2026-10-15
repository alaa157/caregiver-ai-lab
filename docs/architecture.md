# Architecture

Caregiver AI Lab is a single-file-dependency Python project with a clear layer split: data → models → evaluation → UI → tests. No database, no microservices.

## Layers

1. **Data layer** (`src/dataset.py`, `data/`) — synthetic JSONL dataset with labeled reference annotations, plus a validator (duplicate IDs, missing fields, malformed examples).
2. **Schema layer** (`src/schemas.py`) — Pydantic `CareSummary`: summary, observations (category, statement, evidence quote, uncertain flag), unclear_or_conflicting, unsupported_claims.
3. **Normalization layer** (`src/normalization.py`) — Arabic-aware normalization (alef variants, teh marbuta, tatweel, diacritics, punctuation) and sentence splitting. Deterministic, dependency-free.
4. **Model layer** (`src/models/`) — `ModelInterface` ABC + `ModelResult` dataclass. Two adapters:
   - `RuleBasedBaseline`: extractive, keyword scoring, no network, no API key.
   - `HfGenerativeModel`: Hugging Face Transformers (configurable `CAREGIVER_MODEL_ID`), with graceful failure on load/generation/JSON errors.
5. **Evaluation layer** (`src/evaluation/`) — metrics (JSON validity, schema compliance, fact recall, unsupported claims, uncertainty/contradiction preservation, latency, failure rate), evidence tracing, report generation.
6. **Export layer** (`src/export.py`) — JSONL and CSV export for results and human review.
7. **UI layer** (`app.py`) — Gradio with 5 tabs; `gr.State` carries experiment results within a session.
8. **Tests** (`tests/`) — pytest with mocked models; no network or credentials.

## Data flow (single example)

`input note` → model adapter `predict()` → raw text → JSON extraction → Pydantic validation → `ModelResult` (with `valid_json`, `error`, `latency_s`) → UI display / evaluation metrics / report.

## Configuration

All runtime knobs live in `src/config.py` and are driven by environment variables (`CAREGIVER_MODEL_ID`, `HF_TOKEN`, `MAX_NEW_TOKENS`, …). Nothing is hardcoded.

## Colab

`notebooks/caregiver_ai_lab.ipynb` mirrors this flow cell by cell so the notebook and the library stay consistent.
