"""Project configuration. All secrets must come from environment variables
(or Colab Secrets); nothing is hardcoded here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TESTS_DIR = ROOT / "tests"

DATASET_PATH = DATA_DIR / "synthetic_examples.jsonl"
RESULTS_PATH = RESULTS_DIR / "experiment_results.jsonl"
REPORT_PATH = RESULTS_DIR / "experiment_report.md"
REVIEW_EXPORT_PATH = RESULTS_DIR / "human_review_export.csv"

# Model selection. Override with CAREGIVER_MODEL_ID env var (e.g. in Colab Secrets).
DEFAULT_MODEL_ID = os.getenv("CAREGIVER_MODEL_ID", "Omartificial-Intelligence-Project/AraT5v2-base-102M")
BASELINE_MODEL_ID = "rule-based-extractive-baseline"

MAX_NEW_TOKENS = int(os.getenv("CAREGIVER_MAX_NEW_TOKENS", "256"))
MAX_INPUT_CHARS = int(os.getenv("CAREGIVER_MAX_INPUT_CHARS", "2000"))
REQUEST_TIMEOUT_S = int(os.getenv("CAREGIVER_REQUEST_TIMEOUT_S", "120"))

# Hugging Face access token. Only needed for gated models; read securely, never stored here.
HF_TOKEN = os.getenv("HF_TOKEN") or None

SYNTHETIC_DATASET_VERSION = "v1.0-synthetic"
DATASET_LICENSE_NOTE = (
    "This dataset is synthetic, generated for engineering practice only. "
    "It does not represent real caregivers, patients, or clinical populations."
)
