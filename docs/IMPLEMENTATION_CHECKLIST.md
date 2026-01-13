# Implementation Checklist

> Extracted from Work Packages - Implementation tasks only (no writing/documentation)

---

## WP1: Simulation Environment & Baseline ✅ COMPLETED

- [x] Configure GEM simulation framework with PMSM parameters
- [x] Validate parameters against MATLAB/Simulink reference model
- [x] Implement metrics framework (NeuroBench-based)
  - [x] Control metrics: ITAE, IAE, ISE, MAE, RMSE
  - [x] Dynamics metrics: rise time, settling time, overshoot
  - [x] Neuromorphic metrics: SyOps, activation sparsity
- [x] Generate PI-controller baseline trajectories
- [x] Achieve tracking errors < 0.01 A
- [x] Export training data (580+ CSV files in `export/train/`)

---

## WP2: NeuroBench Integration & Interface Development 🔄 IN PROGRESS

### 2.1 NeuroBench Setup
- [ ] Install NeuroBench `2025_GC` branch with closed-loop support
- [ ] Verify `BenchmarkClosedLoop` class is available and working

### 2.2 Environment Wrapper ✅ COMPLETED
- [x] Create `PMSMEnv` wrapper adapting GEM → NeuroBench Gymnasium interface
  - [x] Implement `reset()` returning (observation, info)
  - [x] Implement `step(action)` returning (obs, reward, done, truncated, info)
  - [x] Define observation space: `[i_d, i_q, e_d, e_q]` normalized
  - [x] Define action space: `[u_d, u_q]` normalized to [-1, 1]
  - [x] GEM integration with Park/Clarke transforms
  - [x] Episode data recording for analysis
- [x] `PMSMConfig` dataclass with validated motor parameters
- [x] `OperationsConfig` for NeuroBench compatibility

### 2.3 Agent Wrappers ✅ PI DONE / 🔲 SNN PENDING
- [x] Create `PIControllerAgent` as reference baseline
  - [x] Technical Optimum tuning (Kp, Ki from motor parameters)
  - [x] Decoupling compensation (back-EMF)
  - [x] Anti-windup on integrators
  - [x] `__call__(state) -> action` interface
  - [x] `reset()` for state initialization
- [x] Create `PIControllerTorchAgent` (PyTorch wrapper for NeuroBench)
- [ ] Create `SNNTorchAgent` wrapper (placeholder exists)
  - [ ] Implement stateful neuron management across timesteps
  - [ ] Handle membrane potential persistence between calls
  - [ ] Implement `reset()` for neuron state initialization

### 2.4 Pre/Post Processors ✅ BASIC FRAMEWORK DONE
- [x] `normalize_state()` function
- [x] `denormalize_action()` function
- [x] `rate_encode()` spike encoding (for SNN)
- [x] `population_decode()` spike decoding (for SNN)

### 2.5 Pipeline Validation 🔄 PARTIAL
- [x] Simple integration test (PI + PMSMEnv) - WORKING
- [ ] Run `BenchmarkClosedLoop` with PI-controller successfully
- [ ] Verify metrics match standalone simulation results
- [ ] Confirm closed-loop pipeline works end-to-end with NeuroBench

---

## WP3: SNN Training & Closed-Loop Validation 🔲 NOT STARTED

### 3.1 Network Architecture
- [ ] Design LIF network using snnTorch
  - [ ] Input layer: 4 neurons (i_d, i_q, e_d, e_q)
  - [ ] Hidden layer(s): TBD neurons with LIF dynamics
  - [ ] Output layer: 2 neurons (u_d, u_q)
- [ ] Implement rate coding for spike encoding
- [ ] Implement rate decoding for voltage output

### 3.2 Imitation Learning
- [ ] Load PI-controller trajectory data from `export/train/`
- [ ] Preprocess data (normalization, windowing)
- [ ] Define loss function (MSE on output voltages)
- [ ] Train SNN to imitate PI-controller behavior
- [ ] Validate on held-out trajectories

### 3.3 Closed-Loop Integration
- [ ] Integrate trained SNN into `SNNTorchAgent`
- [ ] Run `BenchmarkClosedLoop` with SNN controller
- [ ] Verify closed-loop stability
- [ ] Compare step response: SNN vs PI

### 3.4 Initial Results
- [ ] Generate step response comparison plots
- [ ] Record tracking error metrics (ITAE, settling time, overshoot)
- [ ] Record neuromorphic metrics (SyOps, sparsity)

---

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
- [ ] **SNN Controller** - from WP3

### 4.3 Metrics Collection
- [ ] Run all controllers through all scenarios
- [ ] Collect control performance metrics per scenario
- [ ] Collect neuromorphic efficiency metrics
- [ ] Calculate energy estimates using published data:
  - [ ] Loihi 2 characterization
  - [ ] SpiNNaker 2 characterization

### 4.4 Results Aggregation
- [ ] Create comparison tables
- [ ] Generate visualization plots
- [ ] Statistical significance testing (if multiple runs)

---

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

---

## Quick Reference: Key Files

| Component | Location | Status |
|-----------|----------|--------|
| PMSM Simulation | `pmsm-pem/simulation/simulate_pmsm.py` | ✅ |
| Metrics Framework | `metrics/benchmark_metrics.py` | ✅ |
| Benchmark Env | `benchmark/pmsm_env.py` | ✅ |
| PI Agent | `benchmark/agents.py` | ✅ |
| SNN Agent | `benchmark/agents.py` | 🔲 Placeholder |
| Processors | `benchmark/processors.py` | ✅ Basic |
| Benchmark Runner | `benchmark/run_benchmark.py` | 🔄 Partial |
| Training Data | `export/train/*.csv` | ✅ |

---

## Progress Tracking

| Work Package | Status | Completion |
|--------------|--------|------------|
| WP1 | ✅ Complete | 100% |
| WP2 | 🔄 In Progress | ~70% |
| WP3 | 🔲 Not Started | 0% |
| WP4 | 🔲 Not Started | 0% |
| WP5 | 🔲 Not Started | 0% |

---

## Next Priority Tasks

1. **Install/verify NeuroBench 2025_GC branch**
2. **Get `BenchmarkClosedLoop` working with PI controller**
3. **Then move to WP3: SNN implementation**

Last Updated: 2026-01-13
