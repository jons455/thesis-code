# Benchmark Module

NeuroBench integration layer for PMSM current control benchmark.

## Purpose

This module bridges the GEM (gym-electric-motor) PMSM simulation with the NeuroBench closed-loop benchmark framework. It provides:

- **Gymnasium-compatible environment** wrapping GEM's PMSM simulation
- **Controller agents** (PI baseline, future SNN)
- **Processor functions** for encoding/decoding (normalization, spike encoding)

## Quick Start

```python
from benchmark import PMSMEnv, PIControllerAgent

# Create environment and agent
env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=2.0)
agent = PIControllerAgent()

# Run episode
state, info = env.reset()
agent.reset()

for _ in range(500):
    action = agent(state)  # [u_d, u_q] normalized
    state, reward, done, truncated, info = env.step(action)
    if done:
        break

# Get results
print(f"Final i_q: {info['i_q']:.4f} A (ref: {info['i_q_ref']:.4f} A)")
```

## Files

| File | Purpose |
|------|---------|
| `pmsm_env.py` | `PMSMEnv` — Gymnasium wrapper for GEM PMSM |
| `agents.py` | `PIControllerAgent`, `PIControllerTorchAgent` — Controller implementations |
| `processors.py` | Encoding/decoding functions (normalize, rate encode, etc.) |
| `run_benchmark.py` | Validation script |
| `tests/` | Unit tests for components |

## PMSMEnv

Gymnasium-compatible wrapper for GEM PMSM current control.

**Observation Space** (normalized to [-1, 1]):
```
[i_d, i_q, e_d, e_q]
  │    │    │    └── q-axis current error (ref - measured)
  │    │    └─────── d-axis current error
  │    └──────────── q-axis current (measured)
  └───────────────── d-axis current (measured)
```

**Action Space** (normalized to [-1, 1]):
```
[u_d, u_q]
  │    └── q-axis voltage command
  └─────── d-axis voltage command
```

**Parameters**:
```python
PMSMEnv(
    n_rpm=1000,           # Motor speed [rpm]
    i_d_ref=0.0,          # d-axis current reference [A]
    i_q_ref=2.0,          # q-axis current reference [A]
    scenario='step_response',  # Benchmark scenario
    max_steps=2000,       # Steps per episode
)
```

## PIControllerAgent

Classical PI controller with Technical Optimum tuning.

**Features**:
- Decoupling compensation (back-EMF feedforward)
- Anti-windup on integrators
- Voltage magnitude limiting

**Interface**:
```python
agent = PIControllerAgent()
agent.reset()                    # Reset integrator states
action = agent(state)            # Get control action
```

## Processors (Future Expansion)

Currently provides basic functions:
- `normalize_state()` — Normalize currents/errors by limits
- `denormalize_action()` — Convert normalized voltage to physical
- `rate_encode()` — Convert continuous value to spikes
- `population_decode()` — Decode spikes to continuous value

**Planned** (for SNN support):
- `DeltaEncodingPreprocessor` — Convert errors to delta-errors
- `IntegratorPostprocessor` — Accumulate voltage kicks
- `IdentityPreprocessor/Postprocessor` — Pass-through for PI

## Running the Benchmark

```bash
cd benchmark
python run_benchmark.py
```

Expected output:
```
Simple Integration Test (without NeuroBench)
============================================
Results:
  Steps completed: 500
  Total reward: -X.XXXX
  Final tracking error: 0.00 mA
  Time in target: 453 steps
  i_d final: 0.0000 A (ref: 0.0000 A)
  i_q final: 2.0000 A (ref: 2.0000 A)

[OK] Simple test PASSED
```

## NeuroBench Integration

For full NeuroBench closed-loop benchmark:

```python
from neurobench.benchmarks import BenchmarkClosedLoop
from neurobench.models import TorchAgent
from benchmark import PMSMEnv, PIControllerTorchAgent

env = PMSMEnv()
agent = TorchAgent(PIControllerTorchAgent())

benchmark = BenchmarkClosedLoop(
    agent=agent,
    environment=env,
    weight_update=False,
    preprocessors=[],
    postprocessors=[],
    metric_list=[...],
)

results, avg_time = benchmark.run(nr_interactions=10, max_length=500)
```

## Motor Parameters

Default motor parameters (validated against MATLAB/Simulink):

| Parameter | Value | Unit |
|-----------|-------|------|
| Pole pairs | 3 | - |
| R_s | 0.543 | Ω |
| L_d | 1.13 | mH |
| L_q | 1.42 | mH |
| Ψ_PM | 16.9 | mWb |
| I_max | 10.8 | A |
| U_DC | 48 | V |
| Control freq | 10 | kHz |

## See Also

- `docs/ARCHITECTURE.md` — Full system architecture
- `docs/BENCHMARK_METRICS.md` — Metrics documentation
- `pmsm-pem/` — GEM simulation scripts
