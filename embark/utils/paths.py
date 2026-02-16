"""Standardized paths for data and model artifacts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Evaluation directory (data + trained models live here) ---
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

DATA_DIR = EVALUATION_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RESULTS_DIR = DATA_DIR / "results"

TRAINED_MODELS_DIR = EVALUATION_DIR / "trained_models"

# Legacy aliases (some scripts still reference these)
MODELS_DIR = TRAINED_MODELS_DIR
MODELS_CHECKPOINTS_DIR = TRAINED_MODELS_DIR
MODELS_BEST_DIR = TRAINED_MODELS_DIR

BENCHMARK_RESULTS_DIR = DATA_RESULTS_DIR / "benchmarks"
COMPARISON_RESULTS_DIR = DATA_RESULTS_DIR / "comparisons"
