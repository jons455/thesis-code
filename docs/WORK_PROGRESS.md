# Work Progress Log

Documentation of implementation progress for the neuromorphic PMSM controller benchmark. 

This log is relevant for the Implementation chapter in the thesis.



## Project Context

### Why This Project?

This project evaluates **Spiking Neural Networks (SNNs)** as an alternative to classical PI controllers for PMSM current control. The goal is to find the best trade-off between:
- **Control Quality**: Can an SNN track current references as accurately as a PI controller?
- **Neuromorphic Efficiency**: Can SNNs provide energy savings on neuromorphic hardware?

### The Approach

1. **Simulation-First**: Use GEM (gym-electric-motor) to simulate the PMSM, avoiding the need for real hardware during development
2. **Imitation Learning**: Train the SNN to mimic PI controller behavior (supervised learning from expert trajectories)
3. **Closed-Loop Evaluation**: Test the trained SNN in closed-loop control using NeuroBench's standardized framework
4. **Fair Comparison**: Measure both control quality and computational cost using consistent metrics



## Component Overview

### pmsm-pem/ — Motor Simulation

**Purpose**: Simulate the PMSM motor physics using GEM (gym-electric-motor).

**Why GEM?**
- Physically accurate motor models (validated against MATLAB/Simulink)
- OpenAI Gym interface (compatible with NeuroBench)
- No MATLAB license needed
- Active maintenance by University of Paderborn

**Key Files**:
| File | Purpose |
|||
| `simulation/simulate_pmsm.py` | Run GEM simulation with standard controller |
| `simulation/run_operating_point_tests.py` | Generate training data across operating points |
| `validation/compare_simulations.py` | Validate GEM matches MATLAB results |
| `export/train/` | 580+ PI trajectories for SNN training |

**Output**: CSV files with (time, i_d, i_q, u_d, u_q, n) at 10 kHz

### benchmark/ — NeuroBench Integration

**Purpose**: Interface layer between GEM and NeuroBench for closed-loop SNN evaluation.

**Why This Layer?**
- GEM uses its own interface; NeuroBench expects Gymnasium
- Need to handle coordinate transforms (dq ↔ abc)
- Need normalized observations/actions for neural networks
- Need flexible processor chains for different encoding schemes

**Key Files**:
| File | Purpose |
|||
| `pmsm_env.py` | Gymnasium wrapper around GEM PMSM environment |
| `agents.py` | PI controller baseline, future SNN controller |
| `processors.py` | Encoding/decoding functions (rate encoding, delta encoding) |
| `run_benchmark.py` | Validation script for the integration |

**Data Flow**:
```
GEM State → PMSMEnv → [Preprocessor] → Agent → [Postprocessor] → PMSMEnv → GEM Action
```

### metrics/ — Evaluation Framework

**Purpose**: Compute control quality and neuromorphic efficiency metrics.

**Why Custom Metrics?**
- Standard ML metrics (accuracy, loss) don't capture control quality
- Need domain-specific metrics: ITAE (penalizes steady-state drift), settling time, overshoot
- Need neuromorphic metrics: SyOps (synaptic operations), activation sparsity
- Control Smoothness (Total Variation) catches "chattering" that RMSE misses

**Key Metrics**:
| Metric | Purpose | Why It Matters |
|--||-|
| RMSE | Overall tracking accuracy | Standard metric, easy to compare |
| ITAE | Time-weighted error | Catches SNN steady-state drift |
| Settling Time | How fast the controller responds | SNNs can be faster than PI |
| Control Smoothness (TV) | Voltage variation per step | Catches SNN chattering |
| SyOps | Computational operations | Proxy for energy consumption |
| Activation Sparsity | % of silent neurons | Higher = more efficient |

### data-preperation/ — Legacy Data Tools

**Purpose**: Tools for merging and preparing simulation data.

**Status**: Partially obsolete. The Edge Impulse approach was explored but not used. The merge scripts may still be useful for data analysis.

**Note**: Edge Impulse guides are archived in `docs/archive/`.



## 2026-01-15

### Documentation Cleanup

**What was done**:
- Aligned IMPLEMENTATION_CHECKLIST.md with ARCHITECTURE.md
- Fixed file path references (benchmark is at root level, not pmsm-pem/benchmark/)
- Updated WP status: WP2 marked as Complete, WP3 ready to start
- Added component overview to WORK_PROGRESS.md

**Why**: The implementation checklist had outdated file paths and inconsistent status tracking. The architecture document had evolved but the checklist hadn't been updated to match.



## 2026-01-13

### WP2: NeuroBench Integration & Interface Development ✅ COMPLETE

**Goal**: Create the interface layer so SNNs can be evaluated in closed-loop control.

**What was done**:
- Installed NeuroBench 2025_GC branch (the version with closed-loop support)
- Created `benchmark/` folder with Gymnasium-compatible environment wrapper
- Implemented PI controller as baseline (proves the pipeline works)
- Validated: PI achieves 0.00 mA tracking error through the new pipeline

**Why this matters**:
- The benchmark pipeline is now ready for any controller (PI, ANN, SNN)
- The PI baseline provides a reference point for comparison
- NeuroBench integration enables standardized neuromorphic metrics

**Key Results**:
```
PI Controller through PMSMEnv:
  i_d final: 0.0000 A (ref: 0.0 A)
  i_q final: 2.0000 A (ref: 2.0 A)
  Time in target: 453/500 steps (91%)
```

**Key Files Created**:
- `benchmark/pmsm_env.py` — PMSMEnv Gymnasium wrapper
- `benchmark/agents.py` — PIControllerAgent, PIControllerTorchAgent
- `benchmark/processors.py` — Basic processor functions
- `benchmark/run_benchmark.py` — Validation script



## Pre-2026-01-13 (Completed)

### WP1: Simulation Environment & Baseline ✅ COMPLETE

**Goal**: Establish a validated simulation environment and generate training data.

**What was done**:
- Configured GEM PMSM simulation with real motor parameters
- Validated against MATLAB/Simulink (tracking error < 1e-11 A at steady state)
- Generated 580+ PI controller trajectories across multiple operating points
- Implemented comprehensive metrics framework (~1100 lines)

**Why this matters**:
- Validated simulation means we can trust the training data
- PI trajectories provide expert demonstrations for imitation learning
- Metrics framework enables consistent evaluation across experiments

**Key Achievement**: GEM Standard Controller produces **identical** steady-state currents as MATLAB/Simulink FOC implementation.

**Training Data Generated**:
| Operating Point | i_d [A] | i_q [A] | Speed [rpm] |
|--|||-|
| Baseline | 0 | 2 | 1000 |
| Medium load | 0 | 5 | 1000 |
| High load | 0 | 8 | 1000 |
| Field weakening | -3 | 2 | 1000 |
| Combined | -3 | 5 | 1000 |
| High FW | -5 | 5 | 1000 |
| + Speed variations | 500, 1500, 2500 rpm |



## WP3: SNN Implementation (Next)

**Goal**: Train an SNN to control the PMSM and compare to PI baseline.

**Approach**: Hybrid SNN-Integrator Architecture
- SNN learns Δu (voltage changes) not absolute voltages
- External integrator accumulates Δu into steady voltage
- This solves the "spike decay at steady state" problem

**Priority Tasks**:
1. **Implement Processor Layer** (from ARCHITECTURE.md)
   - `benchmark/config.py` — ProcessorConfig dataclass
   - `benchmark/processors.py` — Class-based processors (DeltaEncoding, Integrator)
   - `benchmark/runner.py` — EpisodeRunner class

2. **Create SNN Training Pipeline**
   - `snn/` folder structure
   - Dataset loading for PI trajectories
   - Hybrid SNN model with snnTorch LIF neurons

3. **Train & Integrate**
   - Imitation learning: SNN predicts Δu from (i_d, i_q, Δe_d, Δe_q)
   - Closed-loop validation through benchmark pipeline

**Success Criteria**:
- SNN tracks reference with RMSE < 0.1 A (within 10× of PI)
- SNN shows >80% activation sparsity (efficient)
- Control Smoothness (TV) within 2× of PI (no excessive chattering)



*Last updated: 2026-01-15*
