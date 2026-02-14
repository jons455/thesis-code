"""
Utility modules for PMSM Neuromorphic Benchmark
================================================

This package provides shared configuration constants:
- config: PMSM motor parameters (PMSMDefaults) and simulation defaults

Note: paths module is available but not auto-imported. Import explicitly if needed:
    from embark.utils.paths import TRAINED_MODELS_DIR
"""

from .config import DEFAULT_EPISODE_DURATION, DEFAULT_MAX_STEPS, DEFAULT_PMSM

__all__ = [
    "DEFAULT_PMSM",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_EPISODE_DURATION",
]
