"""
Benchmark utility modules.

This package provides shared utilities for the benchmark framework:
- validation: Unified input validation and error handling

"""

from .validation import (
    safe_get_numeric,
    validate_dict,
    validate_numeric_dict,
    validate_state_reference,
)

__all__ = [
    "validate_dict",
    "validate_numeric_dict",
    "validate_state_reference",
    "safe_get_numeric",
]
