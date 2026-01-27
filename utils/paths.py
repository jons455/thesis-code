"""Standardized paths for data and model artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RESULTS_DIR = DATA_DIR / "results"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODELS_BEST_DIR = MODELS_DIR / "best"

BENCHMARK_RESULTS_DIR = DATA_RESULTS_DIR / "benchmarks"
COMPARISON_RESULTS_DIR = DATA_RESULTS_DIR / "comparisons"
