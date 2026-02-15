# Rate-Encoding SNN Benchmark Interface

**Date:** February 15, 2026  
**Status:** ✅ Implemented

This document defines a minimal but complete benchmark interface for **all rate-encoding SNN controllers**. The goal is to support any rate-based SNN architecture without unnecessary complexity.

## Framework Scope

**Primary framework:** PyTorch  
- All processors (`RateSNNStateProcessor`, `RateSNNActionProcessor`) use `torch.Tensor`
- Neural controllers use `SNNControllerWrapper` (PyTorch `nn.Module` wrapper)
- Hardware-in-the-loop exception: `RemoteAkidaPolicy` implements framework-agnostic `Controller` interface directly

**Extension path:** For other frameworks (JAX, TensorFlow, Keras), implement the `Controller` protocol directly instead of using `TensorController` + adapter pattern. See `RemoteAkidaPolicy` as reference.

---

## Design Principles

1. **Rate-encoding agnostic**: Support any SNN that uses rate coding (continuous in, continuous out)
2. **Minimal components**: Remove redundancy, unify similar processors
3. **Configurable, not specialized**: One processor with flags beats multiple specialized ones
4. **Stateful when needed**: Support temporal features and incremental outputs
5. **NeuroBench compatible**: Expose model internals for metrics (spikes, SyOps)

---

## What Rate-Encoding Models Can Need

### Input Features (StateProcessor responsibility)

| Feature Category | Examples | Models Using |
|------------------|----------|--------------|
| **Raw currents** | i_d, i_q | All |
| **Raw references** | i_d_ref, i_q_ref | v12 |
| **Errors** | e_d = i_ref - i | All |
| **Speed** | n, omega | All |
| **Previous actions** | u_d_prev, u_q_prev | v12 (incremental) |
| **Derivatives** | de_d, de_q, dn | v9 |
| **EMA filters** | EMA(e, α) at multiple timescales | v9, v12 |
| **Integrals** | ∫e dt (accumulated error) | Possible future |

### Output Modes (ActionProcessor responsibility)

| Mode | Description | Models Using |
|------|-------------|--------------|
| **Absolute** | Output is final voltage (v_d, v_q) | v5, v9 |
| **Incremental** | Output is Δu, accumulated over time | v12 |
| **PWM** | Voltage → duty cycle with dead-time | Hardware deployment |

### Normalization (Both processors)

| Scheme | Description |
|--------|-------------|
| **Symmetric** | x / x_max → [-1, 1] |
| **Asymmetric** | (x - min) / (max - min) → [0, 1] or [-1, 1] |
| **Custom bounds** | Per-feature min/max |
| **Error gain** | e × gain / i_max (amplifies small errors) |

---

## Proposed Interface

### 1. RateSNNStateProcessor (replaces SNNStateProcessor, MinMaxProcessor)

One configurable processor that handles all rate-encoding input requirements:



| Current Component | Reason to Remove | Replacement |
|-------------------|------------------|-------------|
| `SNNStateProcessor` | Specialized for v5 only | `RateSNNStateProcessor` with v5 config |
| `MinMaxProcessor` | Generic, not SNN-specific | Keep for non-SNN use OR merge into Rate processor |
| `StandardScalerProcessor` | Generic, not SNN-specific | Keep for non-SNN use |
| `IdentityStateProcessor` | Keep | Useful for debugging |
| `IdentityActionProcessor` | Keep | Useful for debugging |
| `LinearActionProcessor` | Superseded | `RateSNNActionProcessor(incremental=False)` |
| `PWMActionProcessor` | Keep | Hardware deployment path |

---

## Feature Matrix

| Feature | v5 | v9 | v12 | RateSNNStateProcessor Flag |
|---------|----|----|-----|---------------------------|
| i_d, i_q | ✓ | ✓ | ✓ | `include_currents=True` |
| i_d_ref, i_q_ref | - | - | ✓ | `include_references=True` |
| e_d, e_q | ✓ | ✓ | ✓ | `include_errors=True` |
| n | ✓ | ✓ | ✓ | `include_speed=True` |
| u_d_prev, u_q_prev | - | - | ✓ | `include_prev_action=True` |
| de_d, de_q, dn | - | ✓ | - | `include_derivatives=True` |
| EMA slow (α=0.98) | - | ✓ | ✓ | `include_ema_slow=True` |
| EMA fast (α=0.70) | - | ✓ | ✓ | `include_ema_fast=True` |
| Integral | - | - | - | `include_integral=True` (future) |
| **Total features** | 5 | 12 | 13 | `output_dim` property |

| Output Mode | v5 | v9 | v12 | RateSNNActionProcessor Config |
|-------------|----|----|-----|------------------------------|
| Absolute (u_d, u_q) | ✓ | ✓ | - | `incremental=False` |
| Incremental (Δu) | - | - | ✓ | `incremental=True, delta_max=0.2` |

---

## Implementation Summary

### ✅ Phase 1: Unified processors created
- `embark/benchmark/processors/rate_snn.py`
  - `RateSNNStateProcessor` — configurable state processor with feature flags
  - `RateSNNActionProcessor` — supports absolute and incremental output modes
  - 102 unit tests covering all feature combinations
- Exported from `embark/benchmark/processors/__init__.py`
- No factory functions — users configure flags directly

### ✅ Phase 2: Adapter updated
- `embark/benchmark/adapters/tensor_adapter.py`
  - `reset()` propagates to both processors
  - `set_prev_action()` feedback loop for incremental models
  - Processors are stateful across timesteps

### ✅ Phase 3: Harness verified
- `embark/benchmark/harness/closed_loop.py` already calls `controller.reset()`
- Reset propagation works through adapter → processors

### ✅ Phase 4: Controllers cleaned up
- Removed `ANNControllerWrapper` (not benchmarking ANNs)
- Kept `SNNControllerWrapper` (PyTorch SNNs)
- Kept `RemoteAkidaPolicy` (hardware-in-the-loop deployment)

### ✅ Phase 5: Tests passing
- All 102 new tests pass
- All 133 existing tests pass (zero regressions)
- Verified backward compatibility with existing processor tests

---

## Usage Example

```python
from embark.benchmark import TensorControllerAdapter
from embark.benchmark.controllers import SNNControllerWrapper
from embark.benchmark.processors import RateSNNStateProcessor, RateSNNActionProcessor

# Your PyTorch SNN model
my_snn = MySNNModel()

# Wrap to add reset(), get_state(), set_state() and extract spike info
wrapped_snn = SNNControllerWrapper(my_snn)

# Configure state processor for your model's input features
state_proc = RateSNNStateProcessor(
    include_currents=True,      # i_d, i_q
    include_errors=True,        # e_d, e_q
    include_speed=True,         # n (rpm)
    include_references=True,    # i_d_ref, i_q_ref (if your model needs them)
    include_prev_action=True,   # u_d_prev, u_q_prev (for incremental models)
    include_derivatives=True,   # de_d, de_q, dn (for derivative features)
    include_ema_slow=True,      # EMA(e, α=0.98) for slow error tracking
    include_ema_fast=True,      # EMA(e, α=0.70) for fast error tracking
    include_integral=False,     # ∫e dt (future feature)
    i_max=20.0,                 # Current normalization bound
    n_max=3000.0,               # Speed normalization bound
    error_gain=2.0,             # Amplify small errors (e × gain / i_max)
)

# Configure action processor for your model's output mode
action_proc = RateSNNActionProcessor(
    incremental=False,          # True for incremental (Δu), False for absolute (u)
    u_max=48.0,                 # Voltage normalization bound
    delta_max=0.2,              # Max Δu per step (only for incremental=True)
)

# Create adapter (implements Controller interface)
controller = TensorControllerAdapter(
    controller=wrapped_snn,
    state_processor=state_proc,
    action_processor=action_proc,
)

# Use with harness
from embark.benchmark import ClosedLoopHarness, PMSMCurrentControlTask

task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
controller.configure(task.physics_engine.config, task)

harness = ClosedLoopHarness(task=task, controller=controller, metrics=[...])
results = harness.run()
```

## Summary

**Before (6+ specialized processors):**
- `SNNStateProcessor` (hardcoded for one model variant)
- `MinMaxProcessor` (generic, kept for non-SNN use)
- `StandardScalerProcessor` (generic, kept for non-SNN use)
- `LinearActionProcessor` (superseded)
- `PWMActionProcessor` (kept for hardware deployment)
- `IdentityStateProcessor/ActionProcessor` (kept for debugging)

**After (2 configurable + 3 special-purpose):**
- `RateSNNStateProcessor` — handles ALL rate-SNN input variations via feature flags
- `RateSNNActionProcessor` — handles absolute AND incremental output modes
- `PWMActionProcessor` — hardware deployment
- `IdentityStateProcessor/ActionProcessor` — debugging
- Old processors kept for backward compatibility

