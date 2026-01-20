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



## WP3: SNN Training & Closed-Loop Validation 🔄 IN PROGRESS

### 3.1 Architecture Decision ✅

**Decision**: Use **Pure SNN** approach instead of Hybrid SNN-Integrator.

- [x] Evaluate architecture options (Pure SNN vs Hybrid)
- [x] Choose Pure SNN with slow-leak output neurons (β=0.995)
- [x] Document architecture in `docs/SNN_ARCHITECTURE.md`

**Rationale**: Pure SNN is simpler (no external integrator) and fully deployable on neuromorphic hardware (Akida).

### 3.2 SNN Model Development ✅
- [x] Create `snn/` folder structure
  - [x] `snn/__init__.py` - Module exports
  - [x] `snn/models.py` - SimpleSNNController (~310 lines)
  - [x] `snn/dataset.py` - PMSMDataset (~310 lines)
  - [x] `snn/train.py` - Training script with CLI (~400 lines)
- [x] Design LIF network using snnTorch
  - [x] Input layer: 4 neurons (i_d, i_q, e_d, e_q)
  - [x] Hidden layers: 64 neurons each, LIF with β=0.9
  - [x] Output layer: 2 neurons, LIF with β=0.995 (slow-leak = integrator)
- [x] Output = membrane potential (not spikes) → continuous voltage
- [x] Training target: absolute voltage [u_d, u_q] (not Δu)

### 3.3 Imitation Learning ✅ PIPELINE READY
- [x] Load PI-controller trajectory data from `pmsm-pem/export/train/`
- [x] Preprocess data (normalization, windowing)
- [x] Define loss function (MSE on output voltage)
- [x] Implement training loop with validation split
- [x] Quick test: 3 epochs, 5 files → loss decreasing ✅
- [ ] **Full training run** (all 580+ files, 100 epochs)

### 3.4 Closed-Loop Integration 🔲 NEXT
- [ ] Implement `SNNControllerAgent` in `benchmark/agents.py`
  - [ ] Load trained model from checkpoint
  - [ ] Stateful membrane potential across timesteps
  - [ ] `__call__(state) -> action` interface
  - [ ] `reset()` for neuron state initialization
- [ ] Test with PMSMEnv in closed loop
- [ ] Verify closed-loop stability (no NaN/explosion)
- [ ] Compare step response: SNN vs PI

### 3.5 Initial Results 🔲
- [ ] Generate step response comparison plots
- [ ] Record tracking error metrics (RMSE, ITAE, settling time, overshoot)
- [ ] Record neuromorphic metrics (SyOps, sparsity)
- [ ] Verify Control Smoothness (TV) metric - SNN must not chatter

### 3.6 (Optional) Hybrid Approach 🔲
Only if Pure SNN shows issues (voltage drift, instability):
- [ ] Add DeltaEncodingPreprocessor
- [ ] Add IntegratorPostprocessor
- [ ] Train with Δu target instead of absolute voltage
- [ ] Compare Pure vs Hybrid



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
|-----------|----------|--------|
| PMSM Simulation | `pmsm-pem/simulation/simulate_pmsm.py` | ✅ |
| Metrics Framework | `metrics/benchmark_metrics.py` | ✅ |
| Benchmark Env | `benchmark/pmsm_env.py` | ✅ |
| PI Agent | `benchmark/agents.py` | ✅ |
| SNN Agent | `benchmark/agents.py` | 🔲 Next |
| Processors (functions) | `benchmark/processors.py` | ✅ Basic |
| Benchmark Runner | `benchmark/run_benchmark.py` | ✅ |
| SNN Models | `snn/models.py` | ✅ SimpleSNNController |
| SNN Dataset | `snn/dataset.py` | ✅ PMSMDataset |
| SNN Training | `snn/train.py` | ✅ Complete |
| SNN Architecture Doc | `docs/SNN_ARCHITECTURE.md` | ✅ |
| Training Data | `pmsm-pem/export/train/*.csv` | ✅ 580+ files |



## Progress Tracking

| Work Package | Status | Completion |
|--------------|--------|------------|
| WP1 | ✅ Complete | 100% |
| WP2 | ✅ Complete | 100% |
| WP3 | 🔄 In Progress | ~60% |
| WP4 | 🔲 Not Started | 0% |
| WP5 | 🔲 Not Started | 0% |



## Next Priority Tasks

1. **Full Training Run** (WP3.3)
   - Run `python -m snn.train --epochs 100`
   - Monitor training curves
   - Save best model

2. **Closed-Loop Integration** (WP3.4)
   - Add `SNNControllerAgent` to `benchmark/agents.py`
   - Test with PMSMEnv
   - Verify stability

3. **Benchmark Comparison** (WP3.5)
   - Generate step response plots (SNN vs PI)
   - Collect metrics (RMSE, sparsity, etc.)



## Architecture Notes

### Pure SNN with Slow-Leak Output (Current Approach)

The SNN uses **slow-leak output neurons** to solve the steady-state problem:
- **Hidden layers**: LIF with β=0.9 (fast dynamics, respond to changes)
- **Output layer**: LIF with β=0.995 (slow leak = built-in integrator)
- **Output**: Membrane potential directly encodes voltage

```
Environment → SNN (with slow-leak output) → Environment
[i_d,i_q,     Hidden (β=0.9) → Output (β=0.995)   [u_d,u_q]
 e_d,e_q]                       ↓ membrane
```

**Key Design Decisions:**
- No external integrator needed (simpler pipeline)
- Fully deployable on neuromorphic hardware (Akida)
- Training target: absolute voltage [u_d, u_q]

### Hybrid SNN-Integrator (Alternative, if needed)

If Pure SNN shows issues, can add:
- DeltaEncodingPreprocessor → SNN predicts Δu
- IntegratorPostprocessor → accumulates kicks



Last Updated: 2026-01-20
