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

## WP2: NeuroBench Integration & Interface Development

### 2.1 NeuroBench Setup
- [ ] Install NeuroBench `2025_GC` branch with closed-loop support
- [ ] Verify `BenchmarkClosedLoop` class is available

### 2.2 Environment Wrapper
- [ ] Create `PMSMEnv` wrapper adapting GEM → NeuroBench Gymnasium interface
  - [ ] Implement `reset()` returning observation
  - [ ] Implement `step(action)` returning (obs, reward, done, truncated, info)
  - [ ] Define observation space (i_d, i_q, i_d_ref, i_q_ref, omega, epsilon)
  - [ ] Define action space (u_d, u_q normalized)

### 2.3 Agent Wrapper
- [ ] Create `SNNTorchAgent` wrapper for NeuroBench
  - [ ] Implement stateful neuron management across timesteps
  - [ ] Handle membrane potential persistence between `__call__` invocations
  - [ ] Implement `reset()` for neuron state initialization

### 2.4 Pipeline Validation
- [ ] Create `PIAgent` wrapper as reference
- [ ] Run `BenchmarkClosedLoop` with PI-controller
- [ ] Verify metrics match standalone simulation results
- [ ] Confirm closed-loop pipeline works end-to-end

---

## WP3: SNN Training & Closed-Loop Validation

### 3.1 Network Architecture
- [ ] Design LIF network using snnTorch
  - [ ] Input layer: 6 neurons (i_d, i_q, i_d_ref, i_q_ref, omega, epsilon)
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

## WP4: Systematic Evaluation & Baseline Comparison

### 4.1 Benchmark Scenarios
- [ ] **Step Response**: Multiple reference step sizes
- [ ] **Operating Point Sweep**: 
  - [ ] Low speed (500 rpm)
  - [ ] Medium speed (1500 rpm)  
  - [ ] High speed (2500 rpm)
  - [ ] Field-weakening region (>2500 rpm)
- [ ] **Disturbance Rejection**: Load torque steps

### 4.2 Controller Comparison
- [ ] **PI Controller** (baseline) - already implemented
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

## WP5: Export & Contribution Packaging

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

| Component | Location |
|-----------|----------|
| PMSM Simulation | `pmsm-pem/simulation/simulate_pmsm.py` |
| Metrics Framework | `metrics/benchmark_metrics.py` |
| Benchmark Env | `benchmark/pmsm_env.py` |
| Agents | `benchmark/agents.py` |
| Training Data | `export/train/*.csv` |

---

## Progress Tracking

| Work Package | Status | Est. Effort |
|--------------|--------|-------------|
| WP1 | ✅ Complete | - |
| WP2 | 🔲 Not Started | Week 1 |
| WP3 | 🔲 Not Started | Week 2 |
| WP4 | 🔲 Not Started | Week 2-3 |
| WP5 | 🔲 Not Started | Week 3 |

Last Updated: 2026-01-13

