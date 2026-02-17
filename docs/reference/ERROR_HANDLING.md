# Error Handling Strategy

This document describes the unified error handling approach used across the benchmark codebase.

## Principles

1. **Validate upfront**: Check all inputs at function entry
2. **Fail fast**: Raise clear errors immediately, don't silently degrade
3. **Use direct access**: After validation, use `dict[key]` (not `.get()`) for required keys
4. **Use `.get()` only**: For optional keys with sensible defaults

## Validation Functions

All validation utilities are in `embark.benchmark.utils.validation`:

### `validate_numeric_dict()`

Validates that a dict has required keys with finite numeric values.

```python
from embark.benchmark.utils.validation import validate_numeric_dict

def my_function(state: StateDict):
    # Validate upfront
    validate_numeric_dict(state, "state", required_keys=("i_d", "i_q"))
    
    # After validation, safe to use direct access
    i_d = state["i_d"]  # ✅ Safe - already validated
    i_q = state["i_q"]  # ✅ Safe - already validated
```

### `validate_state_reference()`

Validates state and reference dicts for tracking metrics.

```python
from embark.benchmark.utils.validation import validate_state_reference

def update(self, state: StateDict, reference: ReferenceDict):
    # Validate upfront
    time = validate_state_reference(
        state, reference, ["i_d", "i_q"], "MyMetric", time_key="time"
    )
    
    # After validation, safe to use direct access
    error = reference["i_d_ref"] - state["i_d"]  # ✅ Safe
```

### `safe_get_numeric()`

For optional keys where a default is acceptable.

```python
from embark.benchmark.utils.validation import safe_get_numeric

# Optional key with default
omega = safe_get_numeric(state, "omega", default=0.0, mapping_name="state")
```

## Patterns by Module Type

### Controllers

**PI Controller** (`agents/pi_controller.py`):
- ✅ Validates state and reference dicts upfront
- ✅ Uses direct access after validation

**SNN Controller** (`agents/snn_controller.py`):
- ✅ Validates tensor inputs (type, shape, finiteness)
- ✅ Uses direct access after validation

### Metrics

**Tracking Metrics** (`metrics/accumulators/tracking.py`):
- ✅ Validates state/reference upfront using `validate_state_reference()`
- ✅ Uses direct access after validation: `reference["i_d_ref"]`

**NeuroBench Adapters** (`contrib/neurobench/metric_adapters.py`):
- ✅ Uses try/except for graceful degradation (external library compatibility)
- ✅ Returns empty dicts on failure (appropriate for adapter pattern)

### Tasks

**PMSM Task** (`tasks/pmsm_current_control.py`):
- ✅ Validates action, state, reference dicts upfront
- ✅ Uses direct access after validation

### Processors

**PWM Processor** (`processors/pwm.py`):
- ✅ Validates configuration in `__post_init__`
- ✅ Validates action tensor size
- ✅ Uses `.get()` for optional keys (e.g., `voltages.get("v_d", 0.0)`)

## Error Message Format

All validation errors follow consistent patterns:

```python
# Type errors
f"{mapping_name} must be a dict, got {type(mapping).__name__}."

# Missing keys
f"{mapping_name} is missing required keys: {missing}."

# Invalid values
f"{mapping_name}['{key}'] must be numeric, got {type(value).__name__}."
f"{mapping_name}['{key}'] must be finite, got {value!r}."
```

## When to Use Each Pattern

| Pattern | Use Case | Example |
|---------|----------|---------|
| `validate_numeric_dict()` + direct access | Required keys | `state["i_d"]` |
| `safe_get_numeric()` | Optional keys with defaults | `omega = safe_get_numeric(state, "omega", 0.0)` |
| Try/except | External library compatibility | NeuroBench adapters |
| `.get()` with defaults | Optional keys in processors | `voltages.get("v_d", 0.0)` |

## Migration Guide

### Before (Inconsistent)

```python
# Some modules validate, some don't
def update(self, state: StateDict, reference: ReferenceDict):
    # No validation - crashes on KeyError
    error = reference["i_d_ref"] - state["i_d"]
```

### After (Unified)

```python
from embark.benchmark.utils.validation import validate_state_reference

def update(self, state: StateDict, reference: ReferenceDict):
    # Validate upfront
    validate_state_reference(state, reference, ["i_d"], "MyMetric")
    
    # Safe direct access after validation
    error = reference["i_d_ref"] - state["i_d"]
```

## Benefits

1. **Consistent behavior**: All modules follow the same pattern
2. **Clear errors**: Users get helpful error messages immediately
3. **No silent failures**: Missing keys raise KeyError, not return None
4. **Type safety**: Validation ensures values are numeric and finite
5. **Maintainability**: Single source of truth for validation logic
