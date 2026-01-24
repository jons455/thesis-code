# System Architecture: Neuromorphic PMSM Control Benchmark

This document helps me keeping track if the architectural aspects of my software. There is also a [draw.io model](https://app.diagrams.net/#G1W4HkU8qH2lNLPS4p-ilH75E5m6x5HMOT#%7B%22pageId%22%3A%22fyiX_BZgRGonI7MBzxk7%22%7D). 


## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           BENCHMARK PIPELINE                                    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              EPISODE LOOP                                │   │
│  │                                                                          │   │
│  │   state ──▶ [PreProc] ──▶ Agent ──▶ [PostProc] ──▶ action               │   │
│  │     ▲                                                  │                 │   │
│  │     │            ┌──────────────────────┐              │                 │   │
│  │     └────────────│    ENVIRONMENT       │◀─────────────┘                 │   │
│  │                  │    (GEM/PMSMEnv)     │                                │   │
│  │                  └──────────────────────┘                                │   │
│  │                             │                                            │   │
│  │                             ▼                                            │   │
│  │                  ┌──────────────────────┐                                │   │
│  │                  │   METRICS RECORDER   │                                │   │
│  │                  └──────────────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         METRICS COMPUTATION                              │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │   │
│  │  │Control Metrics│  │ Neuromorphic  │  │   NeuroBench  │                │   │
│  │  │ (ITAE, etc.)  │  │  (SyOps, etc) │  │   Standard    │                │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: Layered Architecture

The benchmark separates concerns into distinct layers:
- **Preprocessor**: Transforms env state → agent input (e.g., delta encoding)
- **Agent**: The controller being benchmarked (PI, SNN, ANN)
- **Postprocessor**: Transforms agent output → env action (e.g., integrator)
- **Recorder**: Logs data for metrics computation

This allows mixing and matching different controllers with different encoding schemes.



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

**File**: `benchmark/pmsm_env.py`
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

**File**: `benchmark/agents.py`

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

#### SNN Controller (Biological SNN Architecture)

The SNN uses a **biological architecture** to solve the steady-state problem:
- **SNN**: Learns fast dynamics (like P/D terms) - fires when error *changes*
- **Implicit Integrator**: Output neurons use **Slow-Leak LIF** dynamics ($\beta \approx 1.0$) to effectively integrate spikes and hold voltage at steady state.

```python
class SNNControllerAgent:
    """
    Biological SNN for PMSM current control.
    
    Architecture:
    Input [4] ──▶ Hidden [64] ──▶ Output [2] (Slow-Leak LIF) ──▶ Voltage
       (Spikes)      (Spikes)           (Membrane Potential)
       
    The Output neurons act as the "Integrator".
    
    Temporal Upsampling:
    Runs N internal inference steps (e.g., 10) for every 1 control step
    to allow spike propagation and settling.
    """
    
    def __call__(self, state) -> np.ndarray:
        # 1. Expand input for temporal dimension
        # 2. Run SNN for N steps
        # 3. Return final membrane potential as continuous action
        return self.snn(state)
```

**Why Biological?**
| Problem | Hybrid Issue | Biological Solution |
|||--|
| Complexity | External math block adds "non-neural" code | All-neural implementation |
| Steady state | Drift if integrator is separate | Membrane potential naturally holds state |
| Efficiency | Two separate blocks | Unified network |

### 2.4 Processor Layer (Pre/Post Processing)

**File**: `benchmark/processors.py`

Processors transform data between the environment and agent. This enables:
- Different encoding schemes (direct, delta, spike)
- Controller-agnostic benchmark pipeline
- Easy experimentation with architectures

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSOR CHAIN                                  │
│                                                                          │
│  Environment           Preprocessor           Agent           Postprocessor          Environment
│  [i_d,i_q,e_d,e_q] ──▶ DeltaEncoding ──▶ SNN ──▶ Integrator ──▶ [u_d,u_q]
│                        [i_d,i_q,Δe_d,Δe_q]    [kick_d,kick_q]           
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Preprocessors

| Preprocessor | Input | Output | Use Case |
|--|-|--|-|
| `IdentityPreprocessor` | state | state | PI controller |
| `DeltaEncodingPreprocessor` | [i,e] | [i,Δe] | Hybrid SNN |
| `SpikeEncodingPreprocessor` | continuous | spikes | Fully spiking SNN |

#### Postprocessors

| Postprocessor | Input | Output | Use Case |
||-|--|-|
| `IdentityPostprocessor` | [u_d,u_q] | [u_d,u_q] | PI controller |
| `IntegratorPostprocessor` | [kick_d,kick_q] | [u_d,u_q] | Hybrid SNN |
| `SpikeDecodingPostprocessor` | spikes | [u_d,u_q] | Fully spiking SNN |

#### Configuration

```python
@dataclass
class ProcessorConfig:
    """Centralized configuration to avoid magic numbers."""
    
    # Motor limits
    i_max: float = 10.8      # Maximum current [A]
    u_max: float = 48.0      # Maximum voltage [V]
    
    # Timing
    dt: float = 1e-4         # Control timestep [s]
    
    # Preprocessing
    max_delta: float = None  # Optional delta clamping
    
    # Postprocessing (Integrator)
    anti_windup: bool = True
    
    # Spike encoding
    num_neurons_per_input: int = 10
    max_spike_rate: float = 100.0  # Hz
```

### 2.5 Design Decisions & Gotchas

#### ✅ Decision: Temporal Upsampling (Sub-stepping)
The SNN needs time to settle, but the control loop is fixed at 10kHz (100µs).
**Solution**: Run **N=10 inference steps** for every 1 control step.
- Input is repeated for 10 ticks.
- Spikes propagate through layers.
- Output membrane potential is read at the 10th tick.

#### ⚠️ Gotcha 1: Integrator Time Trap

The SNN must be trained to predict **Δu per timestep**, NOT du/dt:

| SNN Output | Training Target | Postprocessor |
||--||
| **Δu per step** ✅ | `u[t] - u[t-1]` | `u_acc += kick` |
| du/dt ❌ | `(u[t] - u[t-1]) / dt` | `u_acc += kick * dt` |

**Decision**: Use Δu per step. Simpler, avoids dt dependency.

#### ⚠️ Gotcha 2: First Step Shock

With delta encoding, the first timestep has a massive delta if reference jumps:
```
t=0: error = 0
t=1: error = 10A (step reference)
delta = 10A - 0 = 10A  ← Huge spike!
```

**Decision**: Accept it (physically correct). Add optional `max_delta` clamping for debugging.

#### ⚠️ Gotcha 3: Anti-Windup

The integrator must not accumulate beyond voltage limits:

```python
# Correct: Clamp accumulator, not just output
if abs(self.u_acc) >= 1.0:
    self.u_acc = np.clip(self.u_acc, -1.0, 1.0)
```

#### ✅ Decision: Data Copy

Always use `.copy()` when recording state arrays:
```python
self.states.append(state.copy())  # ✅ Not state (mutable reference)
```

### 2.7 NeuroBench Integration

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

### 2.8 Controller Configurations

Different controllers require different processor chains:

| Controller | Preprocessor | Postprocessor | Notes |
||--||-|
| PI (baseline) | Identity | Identity | Direct state→action |
| Biological SNN | Normalization | Identity | Direct voltage output |
| Fully Spiking SNN | SpikeEncoding | SpikeDecoding | All-spike pathway |
| ANN (baseline) | Identity | Identity | Fair DL comparison |

Example configurations:

```python
# PI Controller - no processing
runner_pi = EpisodeRunner(
    env=PMSMEnv(),
    agent=PIControllerAgent(),
    preprocessor=IdentityPreprocessor(),
    postprocessor=IdentityPostprocessor(),
)

# Hybrid SNN - delta encoding + integrator
runner_snn = EpisodeRunner(
    env=PMSMEnv(),
    agent=load_snn('hybrid_snn.pt'),
    preprocessor=DeltaEncodingPreprocessor(config),
    postprocessor=IntegratorPostprocessor(config),
)
```



## 3. File Structure

```
thesis-code/
├── benchmark/                   # NeuroBench integration (standalone)
│   ├── __init__.py
│   ├── pmsm_env.py             # PMSMEnv Gymnasium wrapper
│   ├── agents.py               # PI baseline, SNN wrapper
│   ├── processors.py           # Pre/Post processors (encoding, integrator)
│   ├── runner.py               # EpisodeRunner orchestration
│   ├── config.py               # ProcessorConfig, BenchmarkConfig
│   └── run_benchmark.py        # Validation script
│
├── metrics/                     # Metrics framework (standalone)
│   ├── __init__.py
│   ├── benchmark_metrics.py    # ~1100 lines of metrics
│   ├── test_metrics.py         # Unit tests
│   └── METRICS_DOCUMENTATION.md
│
├── snn/                         # SNN models (external training)
│   ├── __init__.py
│   ├── models.py               # snnTorch network definitions
│   ├── dataset.py              # PyTorch Dataset for PI trajectories
│   └── train.py                # Training script (imitation learning)
│
├── pmsm-pem/                    # GEM PMSM simulation
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
├── pmsm-matlab/                 # MATLAB reference implementation
│   ├── foc_pmsm.slx            # Simulink model
│   └── export/validation/       # MATLAB validation data
│
└── docs/                        # Documentation
    ├── README.md               # Docs overview
    ├── ARCHITECTURE.md         # This file
    ├── BENCHMARK_METRICS.md    # Metrics documentation
    ├── SIMULATION.md           # GEM configuration & validation
    ├── WORK_PROGRESS.md        # Progress log
    └── archive/                # Old/superseded docs
```



## 4. Data Flow Diagram

### 4.1 Environment Layer (PMSMEnv)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENVIRONMENT LAYER                               │
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
│                 │  ┌─────────┐         │             ┌─────────────┐   │   │
│                 │  │Normalize│         │             │   Extract   │   │   │
│                 │  │  State  │         │             │   State     │   │   │
│                 │  └────┬────┘         │             └──────┬──────┘   │   │
│                 └───────│──────────────│────────────────────│──────────┘   │
│                         │              │                    │               │
│                         ▼              │                    ▼               │
│                    [i_d,i_q,e_d,e_q]   │           GEM state vector        │
│                                        │                                    │
└────────────────────────────────────────│────────────────────────────────────┘
                                         │
                                         ▼
                                  [u_d, u_q] action
```

### 4.2 Processor Layer (Benchmark Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROCESSOR LAYER                                 │
│                          (wraps Environment + Agent)                         │
│                                                                              │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│   │  PMSMEnv   │    │   PRE-     │    │   AGENT    │    │   POST-    │     │
│   │  (state)   │───▶│ PROCESSOR  │───▶│  (PI/SNN)  │───▶│ PROCESSOR  │──┐  │
│   │            │    │            │    │            │    │            │  │  │
│   └────────────┘    └────────────┘    └────────────┘    └────────────┘  │  │
│         ▲                                                                │  │
│         │                         action                                 │  │
│         └────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Example: SNN Pipeline                                                     │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────┐                         │
│   │[i_d,i_q,    │   │ Normalization│   │   SNN    │                         │
│   │ e_d,e_q]    │──▶│ (Implicit)   │──▶│ u_d,u_q  │                         │
│   │             │   │              │   │ (Direct) │                         │
│   └─────────────┘   └──────────────┘   └──────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 State/Action Vectors

**GEM State Vector** (14 values):
```
[omega, torque, i_a, i_b, i_c, i_sd, i_sq, u_a, u_b, u_c, u_sd, u_sq, epsilon, u_sup]
   0      1      2    3    4     5     6    7    8    9    10    11     12      13
```

**PMSMEnv Observation** (4 values, normalized):
```
[i_d/i_max, i_q/i_max, e_d/i_max, e_q/i_max]
    0           1          2          3
```

**Preprocessed State** (depends on preprocessor):
```
Identity:      [i_d, i_q, e_d, e_q]      ← for PI controller
Normalization: [i_d, i_q, e_d, e_q]      ← for Biological SNN
SpikeEncoding: [spikes × 4×N neurons]    ← for Fully Spiking SNN
```

**Agent Output** (depends on agent type):
```
PI/ANN:     [u_d, u_q]         ← direct voltage
Biological SNN: [u_d, u_q]      ← direct voltage (slow-leak integration)
Spiking:    [spikes × 2×M]      ← spike trains
```

**Postprocessed Action** (to environment):
```
Always: [u_d, u_q] normalized to [-1, 1]
```



## 5. Simulation Parameters

| Parameter | Value | Description |
|--|-|-|
| Control frequency | 10 kHz | Sampling rate |
| Timestep (Ts) | 100 µs | Control period |
| Episode length | 500-2000 steps | 50-200 ms |
| Operating points | 6+ combinations | id/iq sweep |
| Speed range | 500-2500 rpm | Mechanical speed |



## 6. Validation Results (Current)

### PI Controller Baseline (2026-01-13)

| Metric | Value | Status |
|--|-|--|
| Final tracking error | 0.00 mA | ✅ |
| Steps in target (2%) | 453/500 | ✅ |
| i_d tracking | 0.0000 A (ref: 0.0) | ✅ |
| i_q tracking | 2.0000 A (ref: 2.0) | ✅ |

### Comparison with MATLAB (Previous validation)

| Metric | GEM vs MATLAB | Status |
|--||--|
| Current tracking error | < 1e-11 A | ✅ Equivalent |
| All operating points | 6/6 passed | ✅ |
| Voltage offset | ~68% (normalization) | ⚠️ Known |



## 7. Next Steps (WP3)

### 7.1 Implement Processor Layer

**Status**: Implicitly implemented, Refactoring to explicit classes pending (see METHODOLOGY.md)

| File | Purpose | Status |
|||--|
| `benchmark/config.py` | ProcessorConfig dataclass | 🔜 TODO |
| `benchmark/processors.py` | Explicit Pre/Postprocessor classes | 🔜 Refactor needed |
| `benchmark/runner.py` | EpisodeRunner class | 🔜 TODO |

**Existing**: Normalization currently fused in `PMSMEnv`.
**Needed**: Extract to `NormalizationPreprocessor`.

### 7.2 SNN Development

**Note**: SNN training is separate from the benchmark pipeline.
The pipeline accepts any pre-trained `.pt` model file.

| Component | Description | Status |
|--|-|--|
| SNN Architecture | Biological SNN (Slow-Leak LIF) | ✅ Implemented |
| Training Target | Direct Voltage Control | ✅ Decided |
| Training Data | 580+ PI trajectories in `pmsm-pem/export/train/` | ✅ Available |
| SNN Folder | `snn/` directory structure | 🔜 Cleanup needed |
| Training Script | `snn/train.py` | ✅ Basic version |

### 7.3 Benchmark Execution (WP4)

Once the processor layer and a trained SNN are available:

1. **Configure benchmark scenarios**
   - Step responses at various operating points (1.0s episodes)
   - Operating point sweep (500-2500 rpm)
   - Disturbance rejection tests

2. **Run all controllers**
   - PI baseline (IdentityPreprocessor + IdentityPostprocessor)
   - Biological SNN (NormalizationPreprocessor + IdentityPostprocessor)
   - Optional: ANN baseline (fair comparison)

3. **Collect metrics**
   - Control quality: RMSE, ITAE, Max Error, settling time, overshoot
   - Stability: Control Smoothness (TV) - critical for detecting SNN chattering
   - Neuromorphic: SyOps/step, activation sparsity, energy estimate

4. **Generate comparison report**
   - Tables: PI vs SNN vs ANN
   - Plots: step responses, Pareto fronts (RMSE vs SyOps)



## 8. Design Principles

1. **Separation of Concerns**
   - Environment layer: Physics simulation only
   - Processor layer: Encoding/decoding only
   - Agent layer: Control logic only
   - Metrics layer: Measurement only

2. **Controller Agnostic**
   - Any controller implementing `__call__(state) → action` and `reset()` works
   - Processors are swappable for experimentation

3. **Training vs Benchmarking Split**
   - Training (offline): Uses DataLoader, generates `.pt` files
   - Benchmarking (online): Uses EpisodeRunner, measures performance
   - Same Agent class works in both contexts

4. **Reproducibility**
   - All configurations in dataclasses
   - Seeds for RNG in environment and encoding
   - Logged episode data for post-hoc analysis
