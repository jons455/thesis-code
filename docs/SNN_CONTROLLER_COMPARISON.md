# SNN Controller Comparison for Embark Benchmark

**Date:** February 14, 2026  
**Status:** Ready for Implementation

This document describes the two SNN controller types available for benchmarking and what preprocessing is required for each.

---

## Overview

The embark benchmark supports two fundamentally different SNN coding schemes:

| Property | Population Analog (v5) | Pulse-Based (v9) |
|----------|------------------------|------------------|
| **Model File** | `snn_v5/improved_high_alpha/best_model.pt` | `v9/v9_no_tanh.pt` |
| **Architecture** | `LearnedLinearSNNController` | `FeedForwardRateSNN` |
| **Input Features** | 5 (basic) | 12 (temporal augmented) |
| **Coding Scheme** | Population Rate Coding | Event-Driven Temporal Coding |
| **Output Mechanism** | Linear readout with EMA | Spike rate accumulation |
| **Parameters** | ~10,816 | ~21,922 |
| **Neurons** | 228 | 288 |
| **Benchmark Ready** | Yes | Needs custom preprocessor |

---

## Type 1: Population Analog Readout (v5)

### Description

Population Analog Readout uses **rate coding** where neural populations accumulate spikes over time. A linear readout layer averages spike counts to produce continuous voltage outputs. This approach is biologically inspired by population vector decoding used in motor cortex studies.

### Architecture

```
Input (5 features)
    │
    ▼
┌─────────────────────┐
│  Hidden Layer 1     │  64 neurons, Leaky LIF (β=0.9)
│  Linear + LIF       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Hidden Layer 2     │  64 neurons, Leaky LIF (β=0.9)
│  Linear + LIF       │
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  MultiTimescale Readout             │
│  ├─ Fast path: Linear(64 → 2)       │
│  └─ Slow path: Linear(64 → 2)       │
│           with EMA (α=0.98)         │
│  Output = (fast + slow) × scale     │
└─────────────────────────────────────┘
    │
    ▼
Output (2 features: u_d, u_q)
```

### Input Features (5 total)

The model expects 5 normalized features:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | `i_d` | d-axis measured current | `i_d / i_max` |
| 1 | `i_q` | q-axis measured current | `i_q / i_max` |
| 2 | `e_d` | d-axis tracking error | `(i_d_ref - i_d) / i_max × error_gain`, clipped to [-1, 1] |
| 3 | `e_q` | q-axis tracking error | `(i_q_ref - i_q) / i_max × error_gain`, clipped to [-1, 1] |
| 4 | `n` | Normalized motor speed | `n_rpm / n_max` (n_max = 4000 RPM) |

### State Processor

Uses the existing `SNNStateProcessor` with `error_gain=10.0`:

```python
from embark.benchmark.processors.normalizers import SNNStateProcessor

state_processor = SNNStateProcessor(error_gain=10.0)
```

### Why It Works

- **High alpha (0.98)** provides strong integral control action through longer memory in the slow readout path
- **Dual-timescale readout** combines fast response with steady-state accuracy
- **Population-level decoding** is robust to individual neuron variability

### Benchmark Usage

```python
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.processors.normalizers import SNNStateProcessor
from embark.benchmark.processors.decoders import LinearActionProcessor

# Create agent
agent = SNNControllerAgent(
    checkpoint_path="evaluation/trained_models/snn_v5/improved_high_alpha/best_model.pt",
    device="cpu",
    track_spikes=True,
)

# Create adapter with standard processor
adapter = TensorControllerAdapter(
    controller=agent,
    state_processor=SNNStateProcessor(error_gain=10.0),
    action_processor=LinearActionProcessor(),
)

# Ready for benchmark
```

---

## Type 2: Pulse-Based Switching (v9)

### Description

Pulse-Based Switching uses **event-driven temporal coding** where spike timing matters. The model uses dual-population encoding (ON/OFF neurons) to handle signed values and accumulates spikes over 48 rate steps to produce stable outputs.

### Architecture

```
Input (12 features)
    │
    ▼
┌─────────────────────────┐
│  Dual-Population Encode │  x → [x_pos, x_neg] (24 neurons)
│  ON/OFF splitting       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Hidden Layer 1     │  128 neurons, Leaky LIF (β=0.96)
│  Linear + LIF       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Hidden Layer 2     │  96 neurons, Leaky LIF (β=0.90)
│  Linear + LIF       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Hidden Layer 3     │  64 neurons, Leaky LIF (β=0.82)
│  Linear + LIF       │
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Rate Accumulation (48 steps)       │
│  spike_sum / rate_steps → rate      │
│  Linear readout (no tanh in v9)     │
└─────────────────────────────────────┘
    │
    ▼
Output (2 features: u_d, u_q)
```

### Input Features (12 total)

The model expects 12 temporally-augmented features:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | `i_d` | d-axis measured current | `i_d / i_max` |
| 1 | `i_q` | q-axis measured current | `i_q / i_max` |
| 2 | `e_d` | d-axis tracking error | `(i_d_ref - i_d) / i_max × error_gain` |
| 3 | `e_q` | q-axis tracking error | `(i_q_ref - i_q) / i_max × error_gain` |
| 4 | `n` | Normalized motor speed | `n_rpm / n_max` |
| 5 | `de_d` | d-axis error derivative | `e_d[t] - e_d[t-1]` |
| 6 | `de_q` | q-axis error derivative | `e_q[t] - e_q[t-1]` |
| 7 | `e_d_ema_slow` | Slow EMA of d-error | EMA with α=0.98 |
| 8 | `e_q_ema_slow` | Slow EMA of q-error | EMA with α=0.98 |
| 9 | `e_d_ema_fast` | Fast EMA of d-error | EMA with α=0.70 |
| 10 | `e_q_ema_fast` | Fast EMA of q-error | EMA with α=0.70 |
| 11 | `dn` | Speed derivative | `n[t] - n[t-1]` |

### State Processor (REQUIRED: Custom Implementation)

The existing `SNNStateProcessor` only provides 5 features. For v9, you need a custom `TemporalStateProcessor`:

```python
from dataclasses import dataclass, field
import math
import torch
from embark.benchmark.interfaces import (
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
)


@dataclass
class TemporalStateProcessor(StateProcessor):
    """
    State processor for v9 Pulse-Based SNN with temporal augmentation.
    
    Provides 12 features matching v9 training:
    - 5 basic features (like SNNStateProcessor)
    - 2 error derivatives (de_d, de_q)
    - 4 EMA filtered errors (slow α=0.98, fast α=0.70)
    - 1 speed derivative (dn)
    
    Parameters
    ----------
    error_gain : float
        Amplification factor for error signals (default 4.0 for v9).
    n_max : float
        Maximum speed in RPM for normalization (default 4000.0).
    ema_slow_alpha : float
        Alpha for slow EMA filter (default 0.98).
    ema_fast_alpha : float
        Alpha for fast EMA filter (default 0.70).
    """
    
    error_gain: float = 4.0
    n_max: float = 4000.0
    ema_slow_alpha: float = 0.98
    ema_fast_alpha: float = 0.70
    
    # Internal state (mutable defaults handled via field)
    _i_max: float = field(default=1.0, repr=False)
    _prev_e_d: float = field(default=0.0, repr=False)
    _prev_e_q: float = field(default=0.0, repr=False)
    _prev_n: float = field(default=0.0, repr=False)
    _ema_slow_d: float = field(default=0.0, repr=False)
    _ema_slow_q: float = field(default=0.0, repr=False)
    _ema_fast_d: float = field(default=0.0, repr=False)
    _ema_fast_q: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask
    ) -> None:
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
        self._initialized = False

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        # Basic features (same as SNNStateProcessor)
        i_d = state["i_d"] / self._i_max
        i_q = state["i_q"] / self._i_max
        
        e_d_raw = (reference["i_d_ref"] - state["i_d"]) / self._i_max * self.error_gain
        e_q_raw = (reference["i_q_ref"] - state["i_q"]) / self._i_max * self.error_gain
        
        # Clip errors to [-1, 1]
        e_d = max(-1.0, min(1.0, e_d_raw))
        e_q = max(-1.0, min(1.0, e_q_raw))
        
        # Speed normalization (omega rad/s → RPM → normalized)
        omega = state.get("omega", 0.0)
        n_rpm = omega * 60.0 / (2.0 * math.pi)
        n = n_rpm / self.n_max
        
        # Initialize on first call
        if not self._initialized:
            self._prev_e_d = e_d
            self._prev_e_q = e_q
            self._prev_n = n
            self._ema_slow_d = e_d
            self._ema_slow_q = e_q
            self._ema_fast_d = e_d
            self._ema_fast_q = e_q
            self._initialized = True
        
        # Compute derivatives
        de_d = e_d - self._prev_e_d
        de_q = e_q - self._prev_e_q
        dn = n - self._prev_n
        
        # Update EMA filters
        # EMA formula: ema_new = α × ema_old + (1 - α) × value
        self._ema_slow_d = self.ema_slow_alpha * self._ema_slow_d + (1 - self.ema_slow_alpha) * e_d
        self._ema_slow_q = self.ema_slow_alpha * self._ema_slow_q + (1 - self.ema_slow_alpha) * e_q
        self._ema_fast_d = self.ema_fast_alpha * self._ema_fast_d + (1 - self.ema_fast_alpha) * e_d
        self._ema_fast_q = self.ema_fast_alpha * self._ema_fast_q + (1 - self.ema_fast_alpha) * e_q
        
        # Store for next iteration
        self._prev_e_d = e_d
        self._prev_e_q = e_q
        self._prev_n = n
        
        # Build 12-feature vector
        return torch.tensor([
            i_d, i_q,                               # [0-1] Measured currents
            e_d, e_q,                               # [2-3] Tracking errors
            n,                                      # [4]   Normalized speed
            de_d, de_q,                             # [5-6] Error derivatives
            self._ema_slow_d, self._ema_slow_q,     # [7-8] Slow EMA errors
            self._ema_fast_d, self._ema_fast_q,     # [9-10] Fast EMA errors
            dn,                                     # [11]  Speed derivative
        ], dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return 12
```

### Why It Works

- **No tanh compression** (v9 fix) allows learning correct voltage biases
- **Dual-population encoding** (ON/OFF neurons) handles signed values naturally
- **Temporal augmentation** provides derivative and integral-like features
- **Rate-based accumulation** over 48 steps provides stable output

### v9 Improvements over v8

Three fixes for systematic offset bias:
- **Fix A:** Remove tanh on output (most impactful)
- **Fix B:** Initialize readout bias from training data means
- **Fix C:** Add offset-penalty loss term

### Benchmark Usage

```python
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.processors.normalizers import TemporalStateProcessor  # Custom!
from embark.benchmark.processors.decoders import LinearActionProcessor

# Create agent
agent = SNNControllerAgent(
    checkpoint_path="evaluation/trained_models/v9/v9_no_tanh.pt",
    device="cpu",
    track_spikes=True,
)

# Create adapter with temporal processor
adapter = TensorControllerAdapter(
    controller=agent,
    state_processor=TemporalStateProcessor(error_gain=4.0),  # Note: v9 uses gain=4.0
    action_processor=LinearActionProcessor(),
)

# Ready for benchmark
```

---

## Comparison Table

| Aspect | Population Analog (v5) | Pulse-Based (v9) |
|--------|------------------------|------------------|
| **Coding Principle** | Rate coding with population averaging | Temporal coding with spike timing |
| **Biological Analogy** | Motor cortex population vectors | Event-driven sensory processing |
| **Integration Method** | Dual EMA in readout layer | Temporal features + rate accumulation |
| **Input Complexity** | 5 features (simple) | 12 features (requires derivatives/EMA) |
| **State Requirements** | Membrane potentials only | Membrane + temporal history |
| **Inference Cost** | Single forward pass | 48 rate steps per inference |
| **Hardware Suitability** | Akida-compatible | Requires temporal preprocessing |
| **Typical Use Case** | Continuous smooth control | Event-driven reactive control |

---

## Implementation Checklist

### For v5 (Population Analog) - Ready Now

- [x] Model trained and saved
- [x] Architecture wrapper exists (`LearnedLinearSNNController`)
- [x] State processor exists (`SNNStateProcessor`)
- [x] Action processor compatible (`LinearActionProcessor`)
- [x] Can run benchmark immediately

### For v9 (Pulse-Based) - Needs Work

- [x] Model trained and saved
- [x] Architecture wrapper exists (`FeedForwardRateSNNWrapper`)
- [ ] **State processor needed** (`TemporalStateProcessor`) ← Implementation above
- [x] Action processor compatible (`LinearActionProcessor`)
- [ ] Add processor to `embark/benchmark/processors/normalizers.py`
- [ ] Test with quick benchmark

---

## Running the Benchmarks

### After implementing TemporalStateProcessor:

```bash
# Benchmark v5 (Population Analog Readout)
poetry run python -m evaluation.core.run_evaluation \
    --model evaluation/trained_models/snn_v5/improved_high_alpha/best_model.pt \
    --quick

# Benchmark v9 (Pulse-Based Switching)
poetry run python -m evaluation.core.run_evaluation \
    --model evaluation/trained_models/v9/v9_no_tanh.pt \
    --quick
```

### Expected Metrics to Compare

| Metric | Description | Indicates |
|--------|-------------|-----------|
| `mae_i_q` | Mean absolute error on q-current | Control accuracy |
| `max_error_i_q` | Worst-case tracking error | Safety margin |
| `settling_time` | Time to reach 2% error band | Dynamic response |
| `total_spikes` | Spike count over episode | Energy efficiency |
| `syops_per_step` | Synaptic operations per control step | Computational cost |
| `activation_sparsity` | Fraction of silent neurons | Hardware efficiency |

---

## Research Questions for Comparison

1. **Accuracy:** Which coding scheme achieves better tracking accuracy?
2. **Efficiency:** Which requires fewer spikes/synaptic operations?
3. **Robustness:** Which handles disturbances and load changes better?
4. **Dynamics:** Which provides faster settling time?
5. **Hardware:** Which maps better to neuromorphic hardware (Akida)?

---

## References

### Model Classes
- **v5:** `evaluation/pytorch_snn/models/learned_linear.py` → `LearnedLinearSNNController`
- **v9:** `evaluation/pytorch_snn/models/feedforward_rate.py` → `FeedForwardRateSNN`

### Training Notebooks
- **v5:** `notebooks/train_snn_methods_v5.ipynb`
- **v9:** `notebooks/train_snn_v8.ipynb` (v9 is an improved v8)

### Benchmark Integration
- **Agent:** `embark/benchmark/agents.py` → `SNNControllerAgent`
- **Processors:** `embark/benchmark/processors/normalizers.py`
- **Evaluation:** `evaluation/core/run_evaluation.py`
