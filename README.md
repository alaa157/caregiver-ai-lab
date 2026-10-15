# Caregiver AI Lab

**Egyptian Arabic AI Evaluation & Experimentation Platform** — a personal learning project for Computer & Information students.

## What it does

Caregiver AI Lab answers one question:

> Which AI model can summarize Egyptian Arabic and mixed-language daily care notes while preserving the original facts, uncertainty, and supporting evidence?

You enter a **fictional** daily care note, choose a model, generate a structured summary, compare multiple models side by side, check fact preservation, detect unsupported claims and contradictions, evaluate structured JSON output, record experiment results, and export everything for a portfolio or research report.

## What makes this different from a generic chatbot

- **Structured output, not free text**: every summary is validated against a Pydantic JSON schema (`summary`, `observations` with evidence quotes, `unclear_or_conflicting`, `unsupported_claims`).
- **Uncertainty is preserved**: the schema never converts "I don't know" into a statement of fact.
- **Evidence traceable**: every observation must carry a verbatim quote from the input note, or it is flagged unsupported.
- **Real evaluation**: fact recall, unsupported-claim rate, JSON validity, schema compliance, latency, and failure rate are computed from real runs against a labeled synthetic dataset — no made-up numbers.
- **Rule-based baseline included**: you can see whether a generative model actually beats keyword-based extraction before trusting it.
- **Synthetic data only**: the ~150-example dataset is fictional, labeled, and explicitly not clinical data.

## Project structure

```
caregiver-ai-lab/
├── app.py                      # Gradio UI (5 tabs)
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
├── .env.example                # Copy to .env and fill in secrets (never commit .env)
├── .gitignore
├── notebooks/
│   └── caregiver_ai_lab.ipynb  # Colab notebook: sequential end-to-end workflow
├── data/
│   ├── synthetic_examples.jsonl   # ~150 labeled synthetic examples (fictional)
│   └── README.md                  # Synthetic-data notice
├── src/
│   ├── config.py               # Paths, model IDs, env-driven configuration
│   ├── schemas.py              # Pydantic CareSummary schema
│   ├── normalization.py        # Arabic normalization + sentence splitting
│   ├── dataset.py              # Synthetic dataset generator, loader, validator
│   ├── models/
│   │   ├── base.py             # Common ModelInterface + ModelResult
│   │   ├── baseline.py         # Rule-based extractive baseline (no API key)
│   │   └── huggingface_model.py # HF Transformers adapter (configurable model ID)
│   ├── evaluation/
│   │   ├── metrics.py          # JSON validity, fact recall, unsupported claims, etc.
│   │   ├── evidence.py         # Evidence-quote tracing heuristic
│   │   ├── uncertainty.py      # Uncertainty-marker detection
│   │   └── report.py           # Markdown + JSON experiment report
│   └── export.py               # JSONL / CSV export
├── tests/                      # pytest suite (mocked models, no network)
├── results/                    # Generated outputs (gitignored)
└── docs/
    ├── architecture.md
    ├── evaluation_methodology.md
    └── learning_roadmap.md
```

## Stack

Python 3.11+, Pandas, NumPy, scikit-learn, Pydantic, Hugging Face Transformers, Gradio, Matplotlib (notebook), pytest. Free-tier Colab compatible; no paid API required.

## Setup

```bash
git clone https://github.com/alaa157/caregiver-ai-lab.git   # when the repo exists
cd caregiver-ai-lab
cp .env.example .env
# Edit .env if you want to use a gated Hugging Face model (HF_TOKEN=...)
pip install -r requirements.txt
```

In Google Colab, open `notebooks/caregiver_ai_lab.ipynb` and run the cells sequentially. Store `HF_TOKEN` in Colab Secrets, never in the notebook.

## Running

### Baseline only (no API key, works offline)

```bash
python -c "
from src.dataset import generate_dataset, write_dataset, DATASET_PATH
write_dataset(generate_dataset(150), DATASET_PATH)
from src.models.baseline import RuleBasedBaseline
m = RuleBasedBaseline()
r = m.predict('أم خالد أكلت جيداً ولم تنم جيداً.')
print(r.summary)
"
```

### Gradio interface

```bash
python app.py
# Open http://localhost:7860
```

Tabs: **Try a Note** · **Compare Models** · **Evaluation Dashboard** · **Error Analysis** · **Learn**.

### Automated tests

```bash
python -m pytest -q
```

Tests use mocked model responses — no network or credentials needed.

## Reproducing an experiment

1. Open the Colab notebook and run cells 1–12 in order.
2. The notebook builds/validates the dataset, runs baseline (and generative model when available), computes real metrics, and writes `results/experiment_report.md` and `results/experiment_results.jsonl`.
3. Metrics come from real execution only — nothing is fabricated.

## Understanding the metrics

- **Valid JSON rate**: share of model outputs that parse as JSON.
- **Schema compliance rate**: share that validate against the `CareSummary` Pydantic schema.
- **Fact recall (heuristic)**: share of reference important facts whose content words appear in the output. Heuristic — not a definitive correctness proof.
- **Unsupported-claim rate (heuristic)**: share of output claims lacking an evidence quote traceable to the input note.
- **Uncertainty / contradiction preservation**: share of labeled-uncertain or labeled-contradictory examples where the output marks something unclear.

## Important limitations

- This is a **learning tool, not a medical application**. It must not diagnose, recommend treatment, or imply medical correctness.
- The dataset is synthetic and fictional; results do not represent real caregivers or clinical populations.
- Metrics are heuristics; human review (Tab 4) is still necessary.
- If the configured Hugging Face model is unavailable, the baseline keeps working and the error is shown gracefully.

## Future extensions (not implemented)

- Fine-tuning a small Arabic seq2seq model on this task.
- More model adapters (API providers, local GGUF models).
- User accounts / experiment versioning.

## Learning roadmap

Start here → then expand:

| File | What it teaches |
|------|-----------------|
| `src/dataset.py` | Synthetic data design, JSONL, schema validation |
| `src/schemas.py` | Pydantic modeling, structured output contracts |
| `src/normalization.py` | Arabic text normalization, deterministic heuristics |
| `src/models/base.py` + `baseline.py` | Interface design, extractive baselines |
| `src/models/huggingface_model.py` | Local HF inference, graceful failure |
| `src/evaluation/metrics.py` | Real evaluation engineering, heuristic labeling |
| `app.py` | Gradio UI, state, error boundaries |
| `tests/` | Pytest, mocks, deterministic testing |
| `notebooks/caregiver_ai_lab.ipynb` | Colab reproducibility, notebook workflow |

## Ethics & data boundaries

- Never commit real patient or caregiver data.
- Never commit `.env` or API tokens (`HF_TOKEN` goes in Colab Secrets or `.env`, which is gitignored).
- Do not modify or push to the separate `caregiver-daily-notes` repository.
