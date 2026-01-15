# Implementation Checklist

This document helps me to keep the overview on what is left to do in the remaining time to achieve the goal of developing a MVP like end-to-end pipeline. 

> Extracted from Work Packages - Implementation tasks only (no writing/documentation)



## WP1: Simulation Environment & Baseline ✅ COMPLETED

- [x] Configure GEM simulation framework with PMSM parameters
- [x] Validate parameters against MATLAB/Simulink reference model
- [x] Implement metrics framework (NeuroBench-based)
  - [x] Control metrics: ITAE, IAE, ISE, MAE, RMSE
  - [x] Dynamics metrics: rise time, settling time, overshoot
  - [x] Neuromorphic metrics: SyOps, activation sparsity
- [x] Generate PI-controller baseline trajectories
- [x] Achieve tracking errors < 0.01 A
- [x] Export training data (580+ CSV files in `pmsm-pem/export/train/`)



## WP2: NeuroBench Integration & Interface Development ✅ COMPLETED

### 2.1 NeuroBench Setup ✅
- [x] Install NeuroBench `2025_GC` branch (2026-01-13, commit c8dfd47)
- [x] Verify `BenchmarkClosedLoop` class is available

### 2.2 Environment Wrapper ✅
- [x] Create `PMSMEnv` wrapper adapting GEM → NeuroBench Gymnasium interface
  - [x] Implement `reset()` returning (observation, info)
  - [x] Implement `step(action)` returning (obs, reward, done, truncated, info)
  - [x] Define observation space: `[i_d, i_q, e_d, e_q]` normalized
  - [x] Define action space: `[u_d, u_q]` normalized to [-1, 1]
  - [x] GEM integration with Park/Clarke transforms
  - [x] Episode data recording for analysis
- [x] `PMSMConfig` dataclass with validated motor parameters
- [x] `OperationsConfig` for NeuroBench compatibility

### 2.3 Agent Wrappers ✅ PI DONE / 🔲 SNN = WP3
- [x] Create `PIControllerAgent` as reference baseline
  - [x] Technical Optimum tuning (Kp, Ki from motor parameters)
  - [x] Decoupling compensation (back-EMF)
  - [x] Anti-windup on integrators
  - [x] `__call__(state) -> action` interface
  - [x] `reset()` for state initialization
- [x] Create `PIControllerTorchAgent` (PyTorch wrapper for NeuroBench)
- [x] `SNNControllerAgent` placeholder exists (raises NotImplementedError)

### 2.4 Pre/Post Processors ✅ BASIC FUNCTIONS
- [x] `normalize_state()` function
- [x] `denormalize_action()` function
- [x] `rate_encode()` spike encoding (for SNN)
- [x] `population_decode()` spike decoding (for SNN)

### 2.5 Pipeline Validation ✅
- [x] Simple integration test (PI + PMSMEnv) - WORKING
  - PI achieves 0.00 mA tracking error
  - 453/500 steps within 2% settling threshold
- [x] NeuroBench `BenchmarkClosedLoop` runs (minor hook compatibility issues)



## WP3: SNN Training & Closed-Loop Validation 🔲 NOT STARTED

### 3.1 Processor Layer Implementation (from ARCHITECTURE.md)

**Note**: The architecture defines a processor layer for flexible encoding/decoding. This enables the Hybrid SNN-Integrator architecture.

- [ ] Create `benchmark/config.py`
  - [ ] `ProcessorConfig` dataclass (motor limits, dt, anti_windup settings)
- [ ] Expand `benchmark/processors.py` with class-based processors
  - [ ] `IdentityPreprocessor` - for PI controller (pass-through)
  - [ ] `DeltaEncodingPreprocessor` - for Hybrid SNN ([i_d, i_q, e_d, e_q] → [i_d, i_q, Δe_d, Δe_q])
  - [ ] `SpikeEncodingPreprocessor` - for fully spiking SNN (optional)
  - [ ] `IdentityPostprocessor` - for PI controller (pass-through)
  - [ ] `IntegratorPostprocessor` - for Hybrid SNN (accumulates kicks → voltage)
  - [ ] `SpikeDecodingPostprocessor` - for fully spiking SNN (optional)
- [ ] Create `benchmark/runner.py`
  - [ ] `EpisodeRunner` class orchestrating env + preprocessor + agent + postprocessor

### 3.2 SNN Model Development
- [ ] Create `snn/` folder structure
  - [ ] `snn/__init__.py`
  - [ ] `snn/models.py` - snnTorch network definitions
  - [ ] `snn/dataset.py` - PyTorch Dataset for PI trajectories
  - [ ] `snn/train.py` - Training script (imitation learning)
- [ ] Design LIF network using snnTorch
  - [ ] Input layer: 4 neurons (i_d, i_q, Δe_d, Δe_q)
  - [ ] Hidden layer(s): TBD neurons with LIF dynamics
  - [ ] Output layer: 2 neurons (kick_d, kick_q)
- [ ] Training target: Δu = u[t] - u[t-1] per timestep (NOT du/dt!)

### 3.3 Imitation Learning
- [ ] Load PI-controller trajectory data from `pmsm-pem/export/train/`
- [ ] Preprocess data (normalization, windowing, delta computation)
- [ ] Define loss function (MSE on output voltage kicks)
- [ ] Train Hybrid SNN to imitate PI-controller Δu
- [ ] Validate on held-out trajectories

### 3.4 Closed-Loop Integration
- [ ] Implement `HybridSNNAgent` class
  - [ ] Stateful neuron management across timesteps
  - [ ] Membrane potential persistence between calls
  - [ ] `reset()` for neuron state initialization
- [ ] Configure runner with DeltaEncodingPreprocessor + IntegratorPostprocessor
- [ ] Run `BenchmarkClosedLoop` with Hybrid SNN controller
- [ ] Verify closed-loop stability
- [ ] Compare step response: SNN vs PI

### 3.5 Initial Results
- [ ] Generate step response comparison plots
- [ ] Record tracking error metrics (RMSE, ITAE, settling time, overshoot)
- [ ] Record neuromorphic metrics (SyOps, sparsity)
- [ ] Verify Control Smoothness (TV) metric - SNN must not chatter



## WP4: Systematic Evaluation & Baseline Comparison 🔲 NOT STARTED

### 4.1 Benchmark Scenarios
- [ ] **Step Response**: Multiple reference step sizes
- [ ] **Operating Point Sweep**: 
  - [ ] Low speed (500 rpm)
  - [ ] Medium speed (1500 rpm)  
  - [ ] High speed (2500 rpm)
  - [ ] Field-weakening region (>2500 rpm)
- [ ] **Disturbance Rejection**: Load torque steps

### 4.2 Controller Comparison
- [x] **PI Controller** (baseline) - IMPLEMENTED
- [ ] **ANN Controller** (optional dense baseline)
  - [ ] Same architecture as SNN but with ReLU activations
  - [ ] Train with same imitation learning approach
- [ ] **Hybrid SNN Controller** - from WP3

### 4.3 Metrics Collection
- [ ] Run all controllers through all scenarios (1.0s episodes)
- [ ] Collect control performance metrics per scenario:
  - [ ] RMSE, ITAE, Max Error
  - [ ] Settling time, Overshoot
  - [ ] Control Smoothness (TV) - critical for SNN
- [ ] Collect neuromorphic efficiency metrics:
  - [ ] SyOps/step
  - [ ] Activation sparsity
- [ ] Calculate energy estimates using published data:
  - [ ] Loihi 2 characterization (~23 pJ/SyOp)
  - [ ] SpiNNaker 2 characterization (~10 pJ/SyOp)

### 4.4 Results Aggregation
- [ ] Create comparison tables (PI vs SNN vs ANN)
- [ ] Generate visualization plots (step responses, Pareto fronts)
- [ ] Statistical significance testing (if multiple seeds)



## WP5: Export & Contribution Packaging 🔲 NOT STARTED

### 5.1 NIR Export (Stretch Goal)
- [ ] Export trained SNN to NIR format
- [ ] Validate NIR model can be reloaded
- [ ] Document hardware portability

### 5.2 Statistical Analysis
- [ ] Run multiple seeds for variance estimation
- [ ] Calculate confidence intervals for key metrics
- [ ] Identify statistically significant differences

### 5.3 NeuroBench Contribution
- [ ] Package PMSM task as reproducible benchmark
- [ ] Ensure all dependencies are documented
- [ ] Verify reproducibility on clean environment



## Quick Reference: Key Files

| Component | Location | Status |
|--|-|--|
| PMSM Simulation | `pmsm-pem/simulation/simulate_pmsm.py` | ✅ |
| Metrics Framework | `metrics/benchmark_metrics.py` | ✅ |
| Benchmark Env | `benchmark/pmsm_env.py` | ✅ |
| PI Agent | `benchmark/agents.py` | ✅ |
| SNN Agent | `benchmark/agents.py` | 🔲 Placeholder |
| Processors (functions) | `benchmark/processors.py` | ✅ Basic |
| Processors (classes) | `benchmark/processors.py` | 🔲 TODO |
| ProcessorConfig | `benchmark/config.py` | 🔲 TODO |
| EpisodeRunner | `benchmark/runner.py` | 🔲 TODO |
| Benchmark Runner | `benchmark/run_benchmark.py` | ✅ |
| SNN Models | `snn/models.py` | 🔲 TODO |
| SNN Training | `snn/train.py` | 🔲 TODO |
| Training Data | `pmsm-pem/export/train/*.csv` | ✅ |



## Progress Tracking

| Work Package | Status | Completion |
|--|--||
| WP1 | ✅ Complete | 100% |
| WP2 | ✅ Complete | 100% |
| WP3 | 🔲 Not Started | 0% |
| WP4 | 🔲 Not Started | 0% |
| WP5 | 🔲 Not Started | 0% |



## Next Priority Tasks

1. **Implement Processor Layer** (WP3.1)
   - `ProcessorConfig` dataclass in `benchmark/config.py`
   - Class-based processors in `benchmark/processors.py`
   - `EpisodeRunner` in `benchmark/runner.py`

2. **Create SNN Training Pipeline** (WP3.2-3.3)
   - Set up `snn/` folder with model definitions
   - Implement dataset loading for PI trajectories
   - Train Hybrid SNN with imitation learning

3. **Integrate & Validate** (WP3.4-3.5)
   - Wrap trained SNN in HybridSNNAgent
   - Run closed-loop benchmark
   - Compare to PI baseline



## Architecture Notes

### Hybrid SNN-Integrator (from ARCHITECTURE.md)

The SNN uses a **hybrid architecture** to solve the steady-state problem:
- **SNN**: Learns fast dynamics (like P/D terms) - fires when error *changes*
- **External Integrator**: Provides memory (like I term) - holds voltage at steady state

```
Environment → DeltaEncoding → SNN → Integrator → Environment
[i_d,i_q,     [i_d,i_q,        [kick_d,   [u_d,u_q]
 e_d,e_q]      Δe_d,Δe_q]       kick_q]
```

**Key Design Decisions:**
- SNN predicts Δu per timestep (NOT du/dt) → simpler, avoids dt dependency
- Integrator accumulates kicks into steady voltage
- Anti-windup prevents integrator saturation



Last Updated: 2026-01-15
