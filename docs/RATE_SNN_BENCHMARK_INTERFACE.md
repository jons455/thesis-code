# Rate-Encoding SNN Benchmark Interface

**Date:** February 14, 2026  
**Status:** Design Plan

This document defines a minimal but complete benchmark interface for **all rate-encoding SNN controllers**. The goal is to support any rate-based SNN architecture (v5, v9, v12, future variants) without unnecessary complexity.

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

```python
@dataclass
class RateSNNStateProcessor(StateProcessor):
    """
    Universal state processor for rate-encoding SNNs.
    
    Configure via flags which features to include. All temporal features
    (EMA, derivatives, integrals) are computed internally and reset per episode.
    """
    
    # === Feature Selection ===
    include_currents: bool = True           # [i_d, i_q] normalized
    include_references: bool = False        # [i_d_ref, i_q_ref] as separate features
    include_errors: bool = True             # [e_d, e_q] = (ref - meas) * gain
    include_speed: bool = True              # [n] normalized
    include_prev_action: bool = False       # [u_d_prev, u_q_prev] for incremental models
    include_derivatives: bool = False       # [de_d, de_q, dn] first differences
    include_ema_slow: bool = False          # [ema_e_d, ema_e_q] slow filter
    include_ema_fast: bool = False          # [ema_e_d, ema_e_q] fast filter
    include_integral: bool = False          # [int_e_d, int_e_q] accumulated error
    
    # === Normalization Parameters ===
    error_gain: float = 10.0                # Amplification for error signals
    n_max: float = 4000.0                   # Speed normalization (RPM)
    u_max: float = 48.0                     # Voltage normalization
    ema_slow_alpha: float = 0.98            # Slow EMA time constant
    ema_fast_alpha: float = 0.70            # Fast EMA time constant
    integral_decay: float = 0.999           # Anti-windup for integral
    clip_features: bool = True              # Clip all features to [-1, 1]
    
    # === Internal State (managed automatically) ===
    _i_max: float = field(default=1.0, init=False, repr=False)
    _prev_e_d: float = field(default=0.0, init=False, repr=False)
    _prev_e_q: float = field(default=0.0, init=False, repr=False)
    _prev_n: float = field(default=0.0, init=False, repr=False)
    _ema_slow_d: float = field(default=0.0, init=False, repr=False)
    _ema_slow_q: float = field(default=0.0, init=False, repr=False)
    _ema_fast_d: float = field(default=0.0, init=False, repr=False)
    _ema_fast_q: float = field(default=0.0, init=False, repr=False)
    _int_e_d: float = field(default=0.0, init=False, repr=False)
    _int_e_q: float = field(default=0.0, init=False, repr=False)
    _prev_action: tuple[float, float] = field(default=(0.0, 0.0), init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def configure(self, physics_config: SystemConfig, task: ClosedLoopTask) -> None:
        self._i_max = physics_config.i_max
        self.reset()
    
    def reset(self) -> None:
        """Reset temporal state for new episode."""
        self._prev_e_d = 0.0
        self._prev_e_q = 0.0
        self._prev_n = 0.0
        self._ema_slow_d = 0.0
        self._ema_slow_q = 0.0
        self._ema_fast_d = 0.0
        self._ema_fast_q = 0.0
        self._int_e_d = 0.0
        self._int_e_q = 0.0
        self._prev_action = (0.0, 0.0)
        self._initialized = False
    
    def set_prev_action(self, u_d_norm: float, u_q_norm: float) -> None:
        """Called by adapter to feed back previous action."""
        self._prev_action = (u_d_norm, u_q_norm)
    
    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        features = []
        
        # Currents
        i_d = state["i_d"] / self._i_max
        i_q = state["i_q"] / self._i_max
        if self.include_currents:
            features.extend([i_d, i_q])
        
        # References (separate from errors)
        if self.include_references:
            i_d_ref = reference["i_d_ref"] / self._i_max
            i_q_ref = reference["i_q_ref"] / self._i_max
            features.extend([i_d_ref, i_q_ref])
        
        # Errors
        e_d_raw = (reference["i_d_ref"] - state["i_d"]) / self._i_max * self.error_gain
        e_q_raw = (reference["i_q_ref"] - state["i_q"]) / self._i_max * self.error_gain
        e_d = max(-1.0, min(1.0, e_d_raw)) if self.clip_features else e_d_raw
        e_q = max(-1.0, min(1.0, e_q_raw)) if self.clip_features else e_q_raw
        if self.include_errors:
            features.extend([e_d, e_q])
        
        # Speed
        omega = state.get("omega", 0.0)
        n_rpm = omega * 60.0 / (2.0 * math.pi)
        n = n_rpm / self.n_max
        if self.include_speed:
            features.append(n)
        
        # Previous action (for incremental models)
        if self.include_prev_action:
            features.extend([self._prev_action[0], self._prev_action[1]])
        
        # Initialize temporal state on first call
        if not self._initialized:
            self._prev_e_d = e_d
            self._prev_e_q = e_q
            self._prev_n = n
            self._ema_slow_d = e_d
            self._ema_slow_q = e_q
            self._ema_fast_d = e_d
            self._ema_fast_q = e_q
            self._initialized = True
        
        # Derivatives
        if self.include_derivatives:
            de_d = e_d - self._prev_e_d
            de_q = e_q - self._prev_e_q
            dn = n - self._prev_n
            features.extend([de_d, de_q, dn])
        
        # EMA filters
        if self.include_ema_slow:
            self._ema_slow_d = self.ema_slow_alpha * self._ema_slow_d + (1 - self.ema_slow_alpha) * e_d
            self._ema_slow_q = self.ema_slow_alpha * self._ema_slow_q + (1 - self.ema_slow_alpha) * e_q
            features.extend([self._ema_slow_d, self._ema_slow_q])
        
        if self.include_ema_fast:
            self._ema_fast_d = self.ema_fast_alpha * self._ema_fast_d + (1 - self.ema_fast_alpha) * e_d
            self._ema_fast_q = self.ema_fast_alpha * self._ema_fast_q + (1 - self.ema_fast_alpha) * e_q
            features.extend([self._ema_fast_d, self._ema_fast_q])
        
        # Integral
        if self.include_integral:
            self._int_e_d = self.integral_decay * self._int_e_d + e_d
            self._int_e_q = self.integral_decay * self._int_e_q + e_q
            int_e_d_clip = max(-1.0, min(1.0, self._int_e_d * 0.01))  # Scale down
            int_e_q_clip = max(-1.0, min(1.0, self._int_e_q * 0.01))
            features.extend([int_e_d_clip, int_e_q_clip])
        
        # Update previous values
        self._prev_e_d = e_d
        self._prev_e_q = e_q
        self._prev_n = n
        
        return torch.tensor(features, dtype=torch.float32)
    
    @property
    def output_dim(self) -> int:
        dim = 0
        if self.include_currents: dim += 2
        if self.include_references: dim += 2
        if self.include_errors: dim += 2
        if self.include_speed: dim += 1
        if self.include_prev_action: dim += 2
        if self.include_derivatives: dim += 3
        if self.include_ema_slow: dim += 2
        if self.include_ema_fast: dim += 2
        if self.include_integral: dim += 2
        return dim
```

### 2. RateSNNActionProcessor (replaces LinearActionProcessor for SNNs)

Unified action processor supporting both absolute and incremental modes:

```python
@dataclass
class RateSNNActionProcessor(ActionProcessor):
    """
    Universal action processor for rate-encoding SNNs.
    
    Supports absolute output (v5, v9) and incremental output (v12).
    Tracks previous action for feedback to StateProcessor.
    """
    
    # === Mode ===
    incremental: bool = False               # If True, output is Δu (accumulated)
    
    # === Limits ===
    u_max: float = 48.0                     # Voltage limit
    delta_max: float = 0.2                  # Max Δu per step (for incremental mode)
    
    # === Internal State ===
    _u_prev: tuple[float, float] = field(default=(0.0, 0.0), init=False, repr=False)
    
    def configure(self, physics_config: SystemConfig) -> None:
        self.u_max = getattr(physics_config, "u_max", self.u_max)
        self.reset()
    
    def reset(self) -> None:
        """Reset for new episode."""
        self._u_prev = (0.0, 0.0)
    
    def __call__(self, action: torch.Tensor, physics_config: SystemConfig) -> ActionDict:
        action_list = action.detach().cpu().flatten().tolist()
        
        if self.incremental:
            # Incremental mode: action is [Δu_d_norm, Δu_q_norm]
            delta_u_d = max(-self.delta_max, min(self.delta_max, action_list[0]))
            delta_u_q = max(-self.delta_max, min(self.delta_max, action_list[1]))
            
            # Accumulate
            u_d = self._u_prev[0] + delta_u_d * self.u_max
            u_q = self._u_prev[1] + delta_u_q * self.u_max
            
            # Clamp to limits
            u_d = max(-self.u_max, min(self.u_max, u_d))
            u_q = max(-self.u_max, min(self.u_max, u_q))
        else:
            # Absolute mode: action is [u_d_norm, u_q_norm] in [-1, 1]
            u_d = action_list[0] * self.u_max
            u_q = action_list[1] * self.u_max
            
            # Clamp
            u_d = max(-self.u_max, min(self.u_max, u_d))
            u_q = max(-self.u_max, min(self.u_max, u_q))
        
        self._u_prev = (u_d, u_q)
        return {"v_d": u_d, "v_q": u_q}
    
    @property
    def prev_action_normalized(self) -> tuple[float, float]:
        """For feeding back to StateProcessor."""
        return (self._u_prev[0] / self.u_max, self._u_prev[1] / self.u_max)
```

### 3. Updated TensorControllerAdapter

Add reset propagation and action feedback:

```python
@dataclass
class TensorControllerAdapter:
    # ... existing fields ...
    
    def reset(self) -> None:
        """Reset controller and all processors."""
        self.controller.reset()
        
        # Reset processors if they support it
        if hasattr(self.state_processor, "reset"):
            self.state_processor.reset()
        if hasattr(self.action_processor, "reset"):
            self.action_processor.reset()
        
        self._last_observation = None
        self._last_action_tensor = None
    
    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        # ... existing config check ...
        
        # Feed previous action to state processor (for incremental models)
        if hasattr(self.state_processor, "set_prev_action"):
            if hasattr(self.action_processor, "prev_action_normalized"):
                u_prev = self.action_processor.prev_action_normalized
                self.state_processor.set_prev_action(u_prev[0], u_prev[1])
        
        # ... rest of existing pipeline ...
```

---

## Factory Functions for Common Configurations

```python
def create_v5_processor() -> RateSNNStateProcessor:
    """5 features: i_d, i_q, e_d, e_q, n"""
    return RateSNNStateProcessor(
        include_currents=True,
        include_errors=True,
        include_speed=True,
        error_gain=10.0,
    )

def create_v9_processor() -> RateSNNStateProcessor:
    """12 features: currents, errors, speed, derivatives, EMAs"""
    return RateSNNStateProcessor(
        include_currents=True,
        include_errors=True,
        include_speed=True,
        include_derivatives=True,
        include_ema_slow=True,
        include_ema_fast=True,
        error_gain=4.0,
    )

def create_v12_processor() -> RateSNNStateProcessor:
    """13 features: currents, refs, errors, speed, prev_action, EMAs"""
    return RateSNNStateProcessor(
        include_currents=True,
        include_references=True,
        include_errors=True,
        include_speed=True,
        include_prev_action=True,
        include_ema_slow=True,
        include_ema_fast=True,
        include_derivatives=False,  # v12 drops derivatives
        error_gain=4.0,
    )

def create_v5_action_processor() -> RateSNNActionProcessor:
    return RateSNNActionProcessor(incremental=False)

def create_v12_action_processor() -> RateSNNActionProcessor:
    return RateSNNActionProcessor(incremental=True, delta_max=0.2)
```

---

## Components to REMOVE

| Current Component | Reason to Remove | Replacement |
|-------------------|------------------|-------------|
| `SNNStateProcessor` | Specialized for v5 only | `RateSNNStateProcessor` with v5 config |
| `MinMaxProcessor` | Generic, not SNN-specific | Keep for non-SNN use OR merge into Rate processor |
| `StandardScalerProcessor` | Generic, not SNN-specific | Keep for non-SNN use |
| `IdentityStateProcessor` | Keep | Useful for debugging |
| `IdentityActionProcessor` | Keep | Useful for debugging |
| `LinearActionProcessor` | Superseded | `RateSNNActionProcessor(incremental=False)` |
| `PWMActionProcessor` | Keep | Hardware deployment path |

**Simplified processor exports:**

```python
# embark/benchmark/processors/__init__.py

# Universal rate-SNN processors
from .rate_snn import RateSNNStateProcessor, RateSNNActionProcessor
from .rate_snn import create_v5_processor, create_v9_processor, create_v12_processor

# Hardware deployment
from .pwm import PWMActionProcessor, PWMConverter

# Debugging/identity
from .identity import IdentityStateProcessor, IdentityActionProcessor

# Legacy (deprecated, will remove in v2.0)
from .normalizers import SNNStateProcessor  # alias to create_v5_processor()
from .decoders import LinearActionProcessor  # alias to RateSNNActionProcessor
```

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

## Implementation Checklist

### Phase 1: Create unified processors

```
[ ] Create embark/benchmark/processors/rate_snn.py
    [ ] RateSNNStateProcessor class
    [ ] RateSNNActionProcessor class
    [ ] Factory functions (create_v5_processor, etc.)
    [ ] Unit tests for each configuration

[ ] Update embark/benchmark/processors/__init__.py
    [ ] Export new processors
    [ ] Deprecation aliases for old processors
```

### Phase 2: Update adapter

```
[ ] Update embark/benchmark/adapters/tensor_adapter.py
    [ ] Add reset() propagation to processors
    [ ] Add set_prev_action feedback loop
    [ ] Update docstrings
```

### Phase 3: Update harness

```
[ ] Update embark/benchmark/harness/closed_loop.py
    [ ] Call controller.reset() at episode start
    [ ] Ensure processors reset properly
```

### Phase 4: Deprecate old processors

```
[ ] Mark SNNStateProcessor as deprecated (alias to create_v5_processor)
[ ] Mark LinearActionProcessor as deprecated (alias to RateSNNActionProcessor)
[ ] Update all docs to use new processors
```

### Phase 5: Test with all model versions

```
[ ] Verify v5 model works with create_v5_processor()
[ ] Verify v9 model works with create_v9_processor()
[ ] Train v12 model and verify with create_v12_processor()
[ ] Compare metrics (should be identical to old processors)
```

---

## Summary

**Before (6 specialized processors):**
- SNNStateProcessor (v5 only)
- MinMaxProcessor (generic)
- StandardScalerProcessor (generic)
- LinearActionProcessor
- PWMActionProcessor
- IdentityStateProcessor/ActionProcessor

**After (3 core + 2 special-purpose):**
- `RateSNNStateProcessor` — handles ALL rate-SNN input variations
- `RateSNNActionProcessor` — handles absolute AND incremental output
- `PWMActionProcessor` — hardware deployment
- `IdentityStateProcessor/ActionProcessor` — debugging

This design is:
- **Complete**: Handles v5, v9, v12, and any future rate-encoding variant
- **Minimal**: 2 main processors instead of 4+
- **Configurable**: Flags enable/disable features without code changes
- **NeuroBench compatible**: No changes needed to metric pipeline
