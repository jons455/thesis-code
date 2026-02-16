"""
Unified validation utilities for benchmark components.

This module provides consistent error handling patterns across the codebase:
- Input validation for dicts, numeric values, and required keys
- Consistent error messages with context
- Fail-fast validation (validate upfront, then use direct access)

Error Handling Strategy:
------------------------
1. **Validate upfront**: Check all inputs at function entry
2. **Fail fast**: Raise clear errors immediately, don't silently degrade
3. **Use direct access**: After validation, use `dict[key]` (not `.get()`) for required keys
4. **Use `.get()` only**: For optional keys with sensible defaults

This ensures:
- Clear error messages when contracts are violated
- No silent failures from missing keys
- Consistent behavior across all modules
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_dict(
    mapping: dict[str, Any],
    mapping_name: str,
    required_keys: tuple[str, ...] = (),
    optional_keys: tuple[str, ...] = (),
) -> None:
    """
    Validate that an input is a dict with required keys present.

    Args:
        mapping: Dictionary to validate
        mapping_name: Name for error messages (e.g., "state", "reference")
        required_keys: Keys that must be present
        optional_keys: Keys that may be present (for documentation)

    Raises:
        TypeError: If mapping is not a dict
        KeyError: If required keys are missing

    Example:
        >>> validate_dict(state, "state", required_keys=("i_d", "i_q"))
        >>> # After validation, safe to use: state["i_d"]
    """
    if not isinstance(mapping, dict):
        raise TypeError(
            f"{mapping_name} must be a dict, got {type(mapping).__name__}."
        )

    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise KeyError(f"{mapping_name} is missing required keys: {missing}.")


def validate_numeric_dict(
    mapping: dict[str, Any],
    mapping_name: str,
    required_keys: tuple[str, ...] = (),
    check_all_values: bool = False,
) -> None:
    """
    Validate that a dict has required keys with finite numeric values.

    Args:
        mapping: Dictionary to validate
        mapping_name: Name for error messages
        required_keys: Keys that must be present and numeric
        check_all_values: If True, validate all values (not just required keys)

    Raises:
        TypeError: If mapping is not a dict or values are not numeric
        KeyError: If required keys are missing
        ValueError: If values are not finite

    Example:
        >>> validate_numeric_dict(state, "state", required_keys=("i_d", "i_q"))
        >>> # After validation, safe to use: state["i_d"]
    """
    validate_dict(mapping, mapping_name, required_keys=required_keys)

    keys_to_check = list(required_keys) if not check_all_values else list(mapping.keys())

    for key in keys_to_check:
        if key not in mapping:
            continue  # Skip optional keys when check_all_values=True

        value = mapping[key]
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{mapping_name}['{key}'] must be numeric, got {type(value).__name__}."
            )
        if not np.isfinite(float(value)):
            raise ValueError(
                f"{mapping_name}['{key}'] must be finite, got {value!r}."
            )


def validate_state_reference(
    state: dict[str, Any],
    reference: dict[str, Any],
    tracked_keys: list[str],
    metric_name: str,
    time_key: str | None = None,
) -> float | None:
    """
    Validate state and reference dicts for tracking metrics.

    Validates:
    - Both are dicts
    - Required keys exist (tracked_keys in state, tracked_keys + "_ref" in reference)
    - Values are numeric and finite
    - Optional time_key exists and is valid

    Args:
        state: State dictionary
        reference: Reference dictionary
        tracked_keys: List of keys to track (e.g., ["i_d", "i_q"])
        metric_name: Name of metric for error messages
        time_key: Optional time key to validate and parse

    Returns:
        Parsed time value if time_key provided, None otherwise

    Raises:
        TypeError: If inputs are wrong type or values aren't numeric
        KeyError: If required keys are missing
        ValueError: If values are not finite

    Example:
        >>> time = validate_state_reference(
        ...     state, reference, ["i_d", "i_q"], "TrackingITAE", time_key="time"
        ... )
        >>> # After validation, safe to use: state["i_d"], reference["i_d_ref"]
    """
    if not isinstance(state, dict):
        raise TypeError(
            f"{metric_name}: state must be a dict, got {type(state).__name__}."
        )
    if not isinstance(reference, dict):
        raise TypeError(
            f"{metric_name}: reference must be a dict, got {type(reference).__name__}."
        )

    parsed_time: float | None = None
    if time_key is not None:
        if time_key not in state:
            raise KeyError(
                f"{metric_name}: state missing required time key '{time_key}'."
            )
        try:
            parsed_time = float(state[time_key])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{metric_name}: state['{time_key}'] must be numeric."
            ) from exc
        if not np.isfinite(parsed_time):
            raise ValueError(
                f"{metric_name}: state['{time_key}'] must be finite, got {parsed_time!r}."
            )

    for key in tracked_keys:
        ref_key = f"{key}_ref"
        if key not in state:
            raise KeyError(f"{metric_name}: state missing tracked key '{key}'.")
        if ref_key not in reference:
            raise KeyError(f"{metric_name}: reference missing key '{ref_key}'.")
        try:
            state_val = float(state[key])
            ref_val = float(reference[ref_key])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{metric_name}: values for '{key}' and '{ref_key}' must be numeric."
            ) from exc
        if not (np.isfinite(state_val) and np.isfinite(ref_val)):
            raise ValueError(
                f"{metric_name}: non-finite value for '{key}' or '{ref_key}'."
            )

    return parsed_time


def safe_get_numeric(
    mapping: dict[str, Any],
    key: str,
    default: float = 0.0,
    mapping_name: str = "dict",
) -> float:
    """
    Safely get a numeric value from a dict with validation.

    Use this for optional keys where a default is acceptable.
    For required keys, use validate_numeric_dict() + direct access instead.

    Args:
        mapping: Dictionary to read from
        key: Key to read
        default: Default value if key missing or invalid
        mapping_name: Name for error messages

    Returns:
        Numeric value or default

    Example:
        >>> omega = safe_get_numeric(state, "omega", default=0.0, mapping_name="state")
    """
    if key not in mapping:
        return default

    value = mapping[key]
    try:
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default
