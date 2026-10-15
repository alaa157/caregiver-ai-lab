# Learning roadmap

Study in this order. Each file teaches specific AI-engineering concepts.

## 1. Start here
- `src/dataset.py` — synthetic data design, JSONL I/O, schema validation, duplicate-ID detection.
- `src/schemas.py` — Pydantic: why structured output matters, validation contracts.

## 2. Baseline first
- `src/normalization.py` — Arabic-aware text normalization; deterministic heuristics vs models.
- `src/models/base.py` — interface design: every model returns the same shape.
- `src/models/baseline.py` — rule-based extractive baseline; your "is the model worth it?" reference.

## 3. Generative models
- `src/models/huggingface_model.py` — Hugging Face inference, configurable model ID, graceful failure (no model = app still works).

## 4. Evaluation (the core)
- `src/evaluation/metrics.py` — real metrics: JSON validity, schema compliance, fact recall, unsupported claims, latency, failure rate. All heuristics labeled as such.
- `src/evaluation/evidence.py`, `uncertainty.py` — tracing claims back to source text.

## 5. UI and export
- `app.py` — Gradio multi-tab interface, session state, error boundaries.
- `src/export.py` — JSONL/CSV export for portfolios.

## 6. Testing and reproducibility
- `tests/` — pytest with mocked models; deterministic tests; no network.
- `notebooks/caregiver_ai_lab.ipynb` — Colab workflow: setup → data → baseline → model → evaluation → report → Gradio launch.

## Concepts you will touch

Baselines · tokenization vs raw text · prompt engineering · model inference · precision/recall · hallucinations · evaluation leakage · train/validation/test separation · reproducibility · why metrics can mislead · structured output · error analysis · human-in-the-loop review.

## Files NOT yet present (future extensions)

Fine-tuning scripts, API-provider adapters, experiment versioning, user accounts. These are deliberately out of scope for v1.
