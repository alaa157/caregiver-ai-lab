"""Caregiver AI Lab — Gradio interface.

Five tabs:
1. Try a Note — single-example structured summarization.
2. Compare Models — side-by-side outputs on the same note.
3. Evaluation Dashboard — real metrics from completed experiments.
4. Error Analysis — inspect failures and add human-review labels.
5. Learn — beginner-friendly concept explanations.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from src.config import BASELINE_MODEL_ID, DEFAULT_MODEL_ID, RESULTS_DIR  # noqa: E402
from src.dataset import (  # noqa: E402
    load_dataset,
    generate_dataset,
    write_dataset,
    validate_dataset,
    DATASET_PATH,
)
from src.models.baseline import RuleBasedBaseline  # noqa: E402
from src.models.huggingface_model import HfGenerativeModel  # noqa: E402
from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.export import export_results_jsonl, export_review_csv  # noqa: E402

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model registry ---------------------------------------------------------
BASELINE = RuleBasedBaseline()
HF_MODEL = HfGenerativeModel()
MODEL_REGISTRY: Dict[str, Any] = {BASELINE.name(): BASELINE, HF_MODEL.name(): HF_MODEL}

# --- Dataset ----------------------------------------------------------------
try:
    EXAMPLES = load_dataset(DATASET_PATH)
    _val = validate_dataset(EXAMPLES)
    if not _val["valid"]:
        EXAMPLES = generate_dataset(150)
        write_dataset(EXAMPLES, DATASET_PATH)
except Exception:
    EXAMPLES = generate_dataset(150)
    write_dataset(EXAMPLES, DATASET_PATH)

EXAMPLE_CHOICES = [(f"{ex['example_id']} ({ex['language_type']})", ex["example_id"]) for ex in EXAMPLES]


def _record_example(example_id: str) -> Dict[str, Any]:
    for ex in EXAMPLES:
        if ex["example_id"] == example_id:
            return ex
    return {}


def _run_models(note: str, model_names: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for mname in model_names:
        model = MODEL_REGISTRY.get(mname)
        if model is None:
            continue
        result = model.predict(note)
        record = result.to_record()
        record["example_id"] = f"run-{abs(hash(note)) % 100000:05d}"
        record["language_type"] = "mixed"
        results.append(record)
    return results


def _issues_for(record: Dict[str, Any], example: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not record.get("ok"):
        issues.append(f"Model error: {record.get('error')}")
        return issues
    if not record.get("valid_json"):
        issues.append("Invalid JSON returned by model")
    structured = record.get("structured", {}) or {}
    obs = structured.get("observations", [])
    for o in obs:
        if o.get("uncertain"):
            issues.append(f"Uncertainty preserved: '{o.get('statement')}'")
    if structured.get("unclear_or_conflicting"):
        issues.append("Contradiction/unclear section present")
    if not example:
        return issues
    ref_facts = example.get("expected_facts", [])
    if ref_facts and not obs:
        issues.append("No observations extracted (possible omitted facts)")
    ref_unc = example.get("uncertainties", [])
    if ref_unc and not structured.get("unclear_or_conflicting") and not any(o.get("uncertain") for o in obs):
        issues.append("Reference note contains uncertainty that the output did not mark")
    return issues


# --- Tab 1: Try a Note ------------------------------------------------------
def try_note(note: str, language: str, model_name: str):
    if not note or not note.strip():
        return ("Please enter a fictional care note.", "", "", "", "", "", "")
    model = MODEL_REGISTRY.get(model_name, BASELINE)
    result = model.predict(note)
    structured = result.structured or {}
    obs_lines = []
    for o in structured.get("observations", []):
        obs_lines.append(f"- [{o.get('category')}] {o.get('statement')} (evidence: {o.get('evidence_quote')})")
    obs_text = "\n".join(obs_lines) if obs_lines else "No observations extracted."
    unclear = structured.get("unclear_or_conflicting", [])
    unsup = structured.get("unsupported_claims", [])
    warnings_parts = []
    if any(o.get("uncertain") for o in structured.get("observations", [])):
        warnings_parts.append("⚠ Output marks some observations as uncertain — treat them as unconfirmed.")
    if unsup:
        warnings_parts.append(f"⚠ {len(unsup)} unsupported claim(s) detected in output.")
    if not result.get("valid_json"):
        warnings_parts.append("⚠ Model returned invalid JSON; showing a best-effort parse.")
    warnings_html = "<br>".join(warnings_parts) if warnings_parts else "No warnings."
    latency = f"{result.latency_s:.3f} s" if result.latency_s is not None else "n/a"
    error_text = result.error or ""
    return (
        result.summary,
        obs_text,
        json.dumps(structured, ensure_ascii=False, indent=2),
        "\n".join(o.get("evidence_quote", "") for o in structured.get("observations", [])) or "—",
        warnings_html,
        latency,
        error_text,
    )


# --- Tab 2: Compare Models --------------------------------------------------
def compare_models(note: str, model_names: List[str]):
    if not note or not note.strip():
        return ("Please enter a note.",) * 8
    if len(model_names) < 2:
        return ("Select at least two models.",) * 8
    outputs = []
    for mname in model_names[:4]:
        model = MODEL_REGISTRY.get(mname)
        result = model.predict(note) if model else None
        if not result:
            outputs.append((mname, "Model unavailable", "", "n/a", "error"))
            continue
        structured = result.structured or {}
        obs_count = len(structured.get("observations", []))
        unclear_count = len(structured.get("unclear_or_conflicting", []))
        outputs.append(
            (
                mname,
                result.summary or "(no output)",
                json.dumps(structured, ensure_ascii=False, indent=2)[:1500],
                f"{result.latency_s:.3f} s" if result.latency_s else "n/a",
                "ok" if result.ok else f"error: {result.error}",
            )
        )
    # pad to 4 slots for stable layout
    while len(outputs) < 4:
        outputs.append(("", "", "", "", ""))
    return tuple(sum((list(o) for o in outputs), []))


# --- Tab 3: Evaluation Dashboard -------------------------------------------
def run_evaluation(limit: int):
    examples = EXAMPLES[: max(1, min(int(limit), len(EXAMPLES)))]
    results: List[Dict[str, Any]] = []
    for model in [BASELINE, HF_MODEL] if HF_MODEL.available() else [BASELINE]:
        for ex in examples:
            result = model.predict(ex["input_note"])
            record = result.to_record()
            record["example_id"] = ex["example_id"]
            record["language_type"] = ex["language_type"]
            results.append(record)
    metrics = evaluate_all(results, {ex["example_id"]: ex for ex in EXAMPLES})
    summary_lines = [
        f"Examples evaluated: {len(results)}",
        f"Valid JSON rate: {metrics['json_valid_rate']}",
        f"Schema compliance: {metrics['schema_compliance_rate']}",
        f"Failure rate: {metrics['failure_rate']['rate']} ({metrics['failure_rate']['failures']}/{metrics['failure_rate']['count']})",
        f"Mean latency: {metrics['latency']['mean_s']} s",
        f"Fact recall (heuristic): {metrics['fact_recall']['mean_recall']}",
        f"Unsupported-claim rate (heuristic): {metrics['unsupported_claims']['rate']}",
        f"Uncertainty preservation: {metrics['uncertainty_preservation']['rate']}",
        f"Contradiction preservation: {metrics['contradiction_preservation']['rate']}",
    ]
    chart_df = _metrics_dataframe(metrics)
    return "\n".join(summary_lines), chart_df, results


def _metrics_dataframe(metrics: Dict[str, Any]):
    import pandas as pd

    rows = [
        ("Valid JSON rate", metrics["json_valid_rate"]),
        ("Schema compliance", metrics["schema_compliance_rate"]),
        ("Fact recall (heuristic)", metrics["fact_recall"]["mean_recall"]),
        ("Uncertainty preservation", metrics["uncertainty_preservation"]["rate"]),
        ("Contradiction preservation", metrics["contradiction_preservation"]["rate"]),
        ("Unsupported-claim rate (heuristic)", metrics["unsupported_claims"]["rate"]),
        ("Failure rate", metrics["failure_rate"]["rate"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# --- Tab 4: Error Analysis --------------------------------------------------
def list_analysis_rows(results_state: List[Dict[str, Any]]) -> List[str]:
    choices = []
    for r in results_state:
        choices.append(f"{r.get('example_id')} [{r.get('model_id')}]")
    return choices


def analyze_row(selection: str, results_state: List[Dict[str, Any]]):
    if not selection:
        return ("", "", "", "", "", "", "")
    target = selection.split(" [")[0]
    record = next((r for r in results_state if r.get("example_id") == target), None)
    example = _record_example(target)
    if not record:
        return ("Selection not found.",) * 7
    structured = record.get("structured", {}) or {}
    issues = _issues_for(record, example)
    return (
        example.get("input_note", ""),
        json.dumps(example, ensure_ascii=False, indent=2),
        record.get("summary", ""),
        json.dumps(structured, ensure_ascii=False, indent=2),
        "\n".join(issues) if issues else "No automated issues detected.",
        record.get("human_label", ""),
        record.get("human_notes", ""),
    )


def save_label(selection: str, label: str, notes: str, results_state: List[Dict[str, Any]]):
    if not selection:
        return results_state, "Select a row first."
    target = selection.split(" [")[0]
    for r in results_state:
        if r.get("example_id") == target:
            r["human_label"] = label
            r["human_notes"] = notes
            break
    return results_state, "Label saved."


def export_review(results_state: List[Dict[str, Any]]):
    if not results_state:
        return None, "No results to export."
    path = RESULTS_DIR / "human_review_export.csv"
    export_review_csv(results_state, path)
    return str(path), f"Exported {len(results_state)} rows to {path.name}"


# --- Tab 5: Learn -----------------------------------------------------------
LEARN_MARKDOWN = """
# Learn — Concepts used in this project

### Baselines
A baseline is a simple, non-generative method (here: rule-based sentence
selection). You compare complex models against it to prove they add value.

### Tokenization
Models read tokens, not characters. Arabic script plus English words in one
note can produce different tokenizations; the baseline works on raw text
sentences so you can see the difference.

### Prompt engineering
The generative adapter wraps your note in a fixed instruction. Changing the
wrapping changes outputs — this tab lets you see that sensitivity.

### Model inference
Running the model to produce a summary. Local (transformers) versus API-backed
models differ in latency, cost, and privacy.

### Precision and recall
Precision = share of extracted facts that are correct. Recall = share of
important facts from the reference that the output kept. Both are heuristics
here because exact string matching cannot prove semantic equivalence.

### Hallucinations
Statements the model states as fact but that are not present (and not
provable) in the input note. This project flags claims without evidence
quotes.

### Evaluation leakage
Never evaluate on examples you used to design prompts. The synthetic dataset
is fixed; the test split must stay untouched during development.

### Train / validation / test separation
No training happens in this prototype (inference only), but the same rule
applies: hold out a test set you only evaluate once.

### Reproducibility
Results here depend on model weights, generation settings, and the dataset
version (`v1.0-synthetic`). Record all of them in the report.

### Why metrics can be misleading
A model can score high recall by copying long chunks of the input without
understanding them. Always inspect failures in Tab 4.
"""


def build_app() -> gr.Blocks:
    css = """
    .arabic-note { direction: rtl; text-align: right; font-family: 'Amiri', 'Segoe UI', Tahoma, sans-serif; }
    .warn { color: #b45309; font-weight: 600; }
    .ok-text { color: #166534; font-weight: 600; }
    """
    with gr.Blocks(title="Caregiver AI Lab") as demo:
        gr.Markdown("# Caregiver AI Lab\nFictional Egyptian-Arabic care-note summarization — experimentation platform. Not a medical tool.")

        results_state = gr.State([])
        model_names = gr.State(list(MODEL_REGISTRY.keys()))

        with gr.Tabs():
            # ---------- Tab 1 ----------
            with gr.TabItem("1 · Try a Note"):
                note_in = gr.Textbox(label="Fictional daily care note", lines=4, placeholder="例如: أم خالد أكلت Quantity قليلة ...")
                lang_in = gr.Dropdown(["ar", "mix", "en"], value="mix", label="Language / script")
                model_in = gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value=BASELINE.name(), label="Model")
                gen_btn = gr.Button("Generate summary", variant="primary")
                out_summary = gr.Textbox(label="Summary")
                out_obs = gr.Textbox(label="Observations", lines=6)
                out_struct = gr.Code(label="Structured output (JSON)", language="json")
                out_ev = gr.Textbox(label="Evidence quotes")
                out_warn = gr.HTML(label="Warnings")
                out_lat = gr.Textbox(label="Latency")
                out_err = gr.Textbox(label="Error", interactive=False)
                gen_btn.click(
                    try_note,
                    inputs=[note_in, lang_in, model_in],
                    outputs=[out_summary, out_obs, out_struct, out_ev, out_warn, out_lat, out_err],
                )

            # ---------- Tab 2 ----------
            with gr.TabItem("2 · Compare Models"):
                cmp_note = gr.Textbox(label="Note to run through all models", lines=4)
                cmp_models = gr.CheckboxGroup(choices=list(MODEL_REGISTRY.keys()), value=list(MODEL_REGISTRY.keys()), label="Models to compare")
                cmp_btn = gr.Button("Run comparison", variant="primary")
                cmp_outs = []
                labels = ["Summary", "Structured (JSON)", "Latency", "Status"]
                for i in range(4):
                    with gr.Column():
                        cmp_outs += [
                            gr.Textbox(label=f"Model A — {labels[0]}" if i == 0 else f"Model B — {labels[0]}" if i == 1 else f"Model C — {labels[0]}" if i == 2 else f"Model D — {labels[0]}"),
                            gr.Code(label=f"{['Model A','Model B','Model C','Model D'][i]} — {labels[1]}", language="json"),
                            gr.Textbox(label=f"{['Model A','Model B','Model C','Model D'][i]} — {labels[2]}"),
                            gr.Textbox(label=f"{['Model A','Model B','Model C','Model D'][i]} — {labels[3]}", interactive=False),
                        ]
                cmp_btn.click(compare_models, inputs=[cmp_note, cmp_models], outputs=cmp_outs)

            # ---------- Tab 3 ----------
            with gr.TabItem("3 · Evaluation Dashboard"):
                eval_limit = gr.Number(value=30, label="Examples to evaluate", precision=0)
                eval_btn = gr.Button("Run evaluation on dataset", variant="primary")
                eval_text = gr.Markdown("No experiment has run yet. Click 'Run evaluation on dataset' to compute real metrics.")
                eval_chart = gr.BarPlot(x="metric", y="value", title="Experiment metrics")
                eval_state = gr.State([])

                def dashboard_wrapper(limit, state):
                    text, chart_df, results = run_evaluation(limit)
                    return text, chart_df, results

                eval_btn.click(dashboard_wrapper, inputs=[eval_limit, results_state], outputs=[eval_text, eval_chart, eval_state])

            # ---------- Tab 4 ----------
            with gr.TabItem("4 · Error Analysis"):
                analysis_dropdown = gr.Dropdown(choices=[], label="Select a result to inspect")
                ex_note = gr.Textbox(label="Input note", elem_classes=["arabic-note"])
                ex_ref = gr.Code(label="Reference annotation (JSON)", language="json")
                ex_out = gr.Textbox(label="Model summary")
                ex_struct = gr.Code(label="Model structured output (JSON)", language="json")
                ex_issues = gr.Textbox(label="Automatically detected issues", lines=5)
                ex_label = gr.Radio(["Correct", "Partially correct", "Incorrect", "Contains unsupported claims", "Loses uncertainty", "Omits important information"], label="Human review label")
                ex_notes = gr.Textbox(label="Notes", lines=2)
                save_label_btn = gr.Button("Save label")
                export_btn = gr.Button("Export reviewed results (CSV)")
                export_path = gr.File(label="Exported CSV")
                export_msg = gr.Textbox(label="Status", interactive=False)

                def refresh_choices(state):
                    return gr.update(choices=list_analysis_rows(state))

                results_state.change(refresh_choices, inputs=results_state, outputs=analysis_dropdown)
                analysis_dropdown.change(analyze_row, inputs=[analysis_dropdown, results_state], outputs=[ex_note, ex_ref, ex_out, ex_struct, ex_issues, ex_label, ex_notes])
                save_label_btn.click(save_label, inputs=[analysis_dropdown, ex_label, ex_notes, results_state], outputs=[results_state, export_msg])
                export_btn.click(export_review, inputs=results_state, outputs=[export_path, export_msg])

            # ---------- Tab 5 ----------
            with gr.TabItem("5 · Learn"):
                gr.Markdown(LEARN_MARKDOWN)

    return demo


if __name__ == "__main__":
    import json  # noqa: F401 (used in handlers above via module scope)

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
