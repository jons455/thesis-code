# System Architecture: Neuromorphic PMSM Control Benchmark

**Date**: 2026-01-13  
**Version**: WP2 Complete  
**Branch**: `wp2-neurobench-integration`

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BENCHMARK PIPELINE                                     │
│                                                                                  │
│  ┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │   Reference    │    │   Controller    │    │     Motor       │              │
│  │   Generator    │───▶│   (PI / SNN)    │───▶│   Simulation    │              │
│  │                │    │                 │    │     (GEM)       │              │
│  └────────────────┘    └────────▲────────┘    └────────┬────────┘              │
│                                 │                      │                        │
│                                 │    Feedback Loop     │                        │
│                                 └──────────────────────┘                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         METRICS COLLECTION                               │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │   │
│  │  │Control Metrics│  │ Neuromorphic  │  │   NeuroBench  │                │   │
│  │  │ (ITAE, etc.)  │  │  (SyOps, etc) │  │   Standard    │                │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Motor Simulation Layer (GEM)

**Package**: `gym-electric-motor` (GEM)  
**Environment**: `Cont-CC-PMSM-v0` (Continuous Current Control PMSM)

```python
# Motor Parameters (validated against MATLAB/Simulink)
motor_parameter = {
    'p': 3,              # Pole pairs
    'r_s': 0.543,        # Stator resistance [Ω]
    'l_d': 0.00113,      # d-axis inductance [H]
    'l_q': 0.00142,      # q-axis inductance [H]
    'psi_p': 0.0169,     # PM flux linkage [Wb]
}

limit_values = {
    'i': 10.8,           # Max current [A]
    'u': 48.0,           # DC-link voltage [V]
    'omega': 314.16,     # Max angular velocity [rad/s]
}
```

**Control Frequency**: 10 kHz (Ts = 100 µs)

### 2.2 Environment Wrapper (PMSMEnv)

**File**: `pmsm-pem/benchmark/pmsm_env.py`  
**Purpose**: Bridge between GEM and NeuroBench

```
PMSMEnv (Gymnasium Interface)
├── Observation Space: [i_d, i_q, e_d, e_q] normalized to [-1, 1]
├── Action Space: [u_d, u_q] normalized to [-1, 1]
├── Reference Generator: Step response, operating point sweep
├── Coordinate Transform: dq ↔ abc (Park/Clarke)
└── Metrics Tracking: time_in_range, episode_data
```

**Data Flow**:
```
Agent Output        PMSMEnv              GEM Environment
[u_d, u_q]  ──────▶ dq→abc ──────────▶  Motor Physics
(normalized)        transform           (state update)
                                              │
                                              ▼
[i_d, i_q,  ◀────── Normalize ◀──────────  [i_sd, i_sq, ...]
 e_d, e_q]          + Errors               (14 state values)
```

### 2.3 Controller Agents

**File**: `pmsm-pem/benchmark/agents.py`

#### PI Controller (Baseline)

```python
class PIControllerAgent:
    """
    Technical Optimum tuning:
    Kp = L / (2*Ts)
    Ki = R / (2*Ts)
    
    With decoupling and anti-windup.
    """
    def __call__(self, state) -> action:
        # state: [i_d, i_q, e_d, e_q] normalized
        # action: [u_d, u_q] normalized
```

#### SNN Controller (To be implemented in WP3)

```python
class SNNControllerAgent:
    """
    snnTorch LIF network.
    
    Architecture (planned):
    - Input: 4 neurons (i_d, i_q, e_d, e_q)
    - Hidden: 64-128 LIF neurons
    - Output: 2 neurons (u_d, u_q from membrane potential)
    
    Training: Imitation learning from PI trajectories.
    """
```

### 2.4 NeuroBench Integration

**Package**: `neurobench` (installed from 2025_GC branch, 2026-01-13)  
**Key Class**: `BenchmarkClosedLoop`

```python
from neurobench.benchmarks import BenchmarkClosedLoop
from neurobench.models import SNNTorchAgent

# Wrap SNN for NeuroBench
agent = SNNTorchAgent(trained_snn_model)

# Create benchmark
benchmark = BenchmarkClosedLoop(
    agent=agent,
    environment=env,
    weight_update=False,
    preprocessors=[],
    postprocessors=[],
    metric_list=[
        [Footprint, ConnectionSparsity],      # Static
        [ActivationSparsity, SynapticOperations]  # Workload
    ]
)

# Run
results, avg_time = benchmark.run(nr_interactions=50, max_length=500)
```

---

## 3. File Structure

```
thesis-code/
├── docs/
│   ├── ARCHITECTURE.md          # This file
│   ├── BENCHMARK_METRICS.md     # Metrics documentation
│   ├── WORK_PROGRESS.md         # Progress log
│   ├── CONTROLLER_VERIFICATION.md
│   ├── GEM_KONFIGURATION.md
│   └── GEM_LEARNINGS.md
│
├── pmsm-pem/                    # Main Python package
│   ├── benchmark/               # NEW - NeuroBench integration
│   │   ├── __init__.py
│   │   ├── pmsm_env.py         # PMSMEnv Gymnasium wrapper
│   │   ├── agents.py           # PI baseline, SNN placeholder
│   │   ├── processors.py       # Spike encoding (placeholder)
│   │   └── run_benchmark.py    # Validation script
│   │
│   ├── metrics/                 # Existing metrics framework
│   │   ├── benchmark_metrics.py # ~1100 lines
│   │   └── METRICS_DOCUMENTATION.md
│   │
│   ├── simulation/              # GEM simulation scripts
│   │   ├── simulate_pmsm.py    # GEM standard controller
│   │   └── run_operating_point_tests.py
│   │
│   ├── validation/              # MATLAB comparison
│   │   ├── compare_simulations.py
│   │   └── compare_operating_points.py
│   │
│   ├── export/                  # Simulation results
│   │   ├── gem_standard/       # GEM controller data
│   │   ├── train/              # 580+ PI trajectories
│   │   └── archive/            # Archived runs
│   │
│   └── venv/                    # Python virtual environment
│
└── pmsm-matlab/                 # MATLAB reference implementation
    ├── foc_pmsm.slx            # Simulink model
    └── export/validation/       # MATLAB validation data
```

---

## 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL LOOP                                    │
│                          (runs at 10 kHz)                                    │
│                                                                              │
│   ┌─────────┐   ┌──────────────────────────────────────────────────────┐   │
│   │Reference│   │                    PMSMEnv                            │   │
│   │Generator│   │  ┌─────────┐    ┌────────────┐    ┌─────────────┐   │   │
│   │         │──▶│  │ Compute │    │  Inverse   │    │    GEM      │   │   │
│   │ id_ref  │   │  │ Errors  │    │ Park/Clarke│    │   Motor     │   │   │
│   │ iq_ref  │   │  │         │    │ Transform  │    │  Dynamics   │   │   │
│   └─────────┘   │  └────┬────┘    └─────▲──────┘    └──────┬──────┘   │   │
│                 │       │               │                   │          │   │
│                 │       ▼               │                   ▼          │   │
│                 │  ┌─────────┐    ┌─────┴──────┐    ┌─────────────┐   │   │
│                 │  │Normalize│    │ Controller │    │   Extract   │   │   │
│                 │  │  State  │───▶│  (PI/SNN)  │    │   State     │   │   │
│                 │  │         │    │            │    │             │   │   │
│                 │  └─────────┘    └────────────┘    └─────────────┘   │   │
│                 └──────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

State Vector (from GEM):
[omega, torque, i_a, i_b, i_c, i_sd, i_sq, u_a, u_b, u_c, u_sd, u_sq, epsilon, u_sup]
   0      1      2    3    4     5     6    7    8    9    10    11     12      13

Normalized Observation (to controller):
[i_d/i_max, i_q/i_max, e_d/i_max, e_q/i_max]
    0           1          2          3
```

---

## 5. Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Control frequency | 10 kHz | Sampling rate |
| Timestep (Ts) | 100 µs | Control period |
| Episode length | 500-2000 steps | 50-200 ms |
| Operating points | 6+ combinations | id/iq sweep |
| Speed range | 500-2500 rpm | Mechanical speed |

---

## 6. Validation Results (Current)

### PI Controller Baseline (2026-01-13)

| Metric | Value | Status |
|--------|-------|--------|
| Final tracking error | 0.00 mA | ✅ |
| Steps in target (2%) | 453/500 | ✅ |
| i_d tracking | 0.0000 A (ref: 0.0) | ✅ |
| i_q tracking | 2.0000 A (ref: 2.0) | ✅ |

### Comparison with MATLAB (Previous validation)

| Metric | GEM vs MATLAB | Status |
|--------|---------------|--------|
| Current tracking error | < 1e-11 A | ✅ Equivalent |
| All operating points | 6/6 passed | ✅ |
| Voltage offset | ~68% (normalization) | ⚠️ Known |

---

## 7. Next Steps (WP3)

1. **SNN Architecture Design**
   - Input: 4 neurons (direct or rate-coded)
   - Hidden: 64-128 LIF neurons (snnTorch)
   - Output: 2 neurons (membrane potential → voltage)

2. **Training Pipeline**
   - Load PI trajectories from `export/train/` (580+ files)
   - Supervised imitation learning (MSE loss)
   - Validate closed-loop stability

3. **Benchmark Execution**
   - Run BenchmarkClosedLoop with SNN
   - Collect NeuroBench metrics (SyOps, sparsity)
   - Compare to PI baseline

