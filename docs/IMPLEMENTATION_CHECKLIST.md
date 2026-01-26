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

**Decision**: Use **Behavioral Cloning** with membrane potential readout (Rate-coded PI approach).

- [x] Evaluate architecture options (Direct Mapping vs Behavioral Cloning)
- [x] Choose Behavioral Cloning from PI trajectories [Stroobants2022, Zaidel2021]
- [x] Membrane potential readout (not spike count) for continuous control [Zaidel2021]
- [x] Document architecture in `docs/SNN_ARCHITECTURE.md`

**Scientific Basis** (from literature review):
- Stroobants et al. (2022): Position-coded N-PI on Loihi, 93 neurons [arXiv:2109.10199]
- Zaidel et al. (2021): Rate-coded NEF PI for robotics [PMC7887770]
- van Breukelen (2025): Multiple inference cycles per control step [IMAVS 2025/17]

### 3.2 SNN Model Development ✅
- [x] Create `snn/` folder structure
  - [x] `snn/__init__.py` - Module exports
  - [x] `snn/models.py` - MembraneSNNController (~440 lines)
  - [x] `snn/dataset.py` - PMSMDataset (~310 lines)
  - [x] `snn/train.py` - Training script with CLI (~430 lines)
- [x] Design LIF network using snnTorch
  - [x] Input layer: 4 neurons (i_d, i_q, e_d, e_q)
  - [x] Hidden layers: 64 neurons each, LIF with β=0.9
  - [x] Output layer: 2 neurons, LIF with β=0.995, reset_mechanism="none"
- [x] Output = membrane potential (not spikes) → continuous voltage [Zaidel2021]
- [x] Training target: absolute voltage [u_d, u_q] (behavioral cloning)
- [x] Multi-timestep inference: configurable `num_inference_steps` [van Breukelen2025]

### 3.3 Training Data & Imitation Learning ⚠️ NEEDS DATA GENERATION
- [x] Original data: `pmsm-pem/export/train/` — was CORRUPTED, now deleted
- [x] **Root cause identified**: PI controller state not reset on GEM env reset
- [ ] **Clean data**: `pmsm-pem/export/train_v2/` — ❌ DOES NOT EXIST (needs generation)
  - Script ready: `scripts/generate_training_data.py`
  - Command: `poetry run python scripts/generate_training_data.py --num-files 500`
- [x] Validation script: `scripts/validate_data.py` ✅ Ready
- [ ] **Full training run** with clean data ← 🔲 BLOCKED (needs data first)

### 3.4 Closed-Loop Integration ✅ COMPLETE
- [x] Implement `SNNControllerAgent` in `benchmark/agents.py`
  - [x] Load trained model from checkpoint
  - [x] Stateful membrane potential across timesteps
  - [x] `num_inference_steps` parameter (Option B from literature)
  - [x] `__call__(state) -> action` interface
  - [x] `reset()` for neuron state initialization
  - [x] `get_spike_statistics()` for neuromorphic metrics
- [x] Test with PMSMEnv in closed loop
- [x] Verify closed-loop stability (no NaN/explosion)

### 3.5 Benchmark API ✅ COMPLETE
- [x] Create `benchmark/controller_interface.py`
  - [x] `ControllerInterface` protocol (structural typing)
  - [x] `ControllerAgent` abstract base class
  - [x] `BenchmarkConfig` dataclass
  - [x] `BenchmarkResults` dataclass with `.summary()` and `.to_dict()`
  - [x] `run_benchmark()` main entry point
- [x] Test with PI controller: RMSE=78mA, settling=2.4ms
- [x] NeuroBench-compatible interface

### 3.6 (Optional) Architecture Variants 🔲
If single-network approach shows issues:
- [ ] Two separate SNNs (one per axis) — matches classical PI structure
- [ ] Shared backbone + separate heads — best of both worlds
- [ ] IWTA integration neurons [Stroobants2023, arXiv:2304.08778]



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
| Benchmark API | `benchmark/controller_interface.py` | ✅ NEW |
| PI Agent | `benchmark/agents.py` | ✅ |
| SNN Agent | `benchmark/agents.py` | ✅ Multi-timestep |
| Processors (functions) | `benchmark/processors.py` | ✅ Basic |
| Benchmark Runner | `benchmark/run_benchmark.py` | ✅ |
| SNN Models | `snn/models.py` | ✅ MembraneSNNController |
| SNN Dataset | `snn/dataset.py` | ✅ PMSMDataset |
| SNN Training | `snn/train.py` | ✅ Ready |
| SNN Architecture Doc | `docs/SNN_ARCHITECTURE.md` | ✅ |
| Training Data (OLD) | `pmsm-pem/export/train/*.csv` | ❌ Deleted (was corrupted) |
| Training Data (CLEAN) | `pmsm-pem/export/train_v2/*.csv` | ❌ MISSING (needs generation) |
| SNN Checkpoints | `snn/checkpoints/*.pt` | ❌ MISSING (needs training) |
| Data Validation | `scripts/validate_data.py` | ✅ Ready |
| Data Generator | `scripts/generate_training_data.py` | ✅ Ready |



## Progress Tracking

| Work Package | Status | Completion |
|--------------|--------|------------|
| WP1 | ✅ Complete | 100% |
| WP2 | ✅ Complete | 100% |
| WP3 | ⚠️ Blocked | ~70% (code ready, needs data + training) |
| WP4 | 🔲 Not Started | 0% |
| WP5 | 🔲 Not Started | 0% |

### Session 2026-01-21: Key Accomplishments

1. **Identified Training Data Corruption**
   - Old data had 88% wrong sign (i_q negative when ref positive)
   - Root cause: PI integrator not reset on GEM env reset

2. **Generated Clean Training Data**
   - 1000 files in `train_v2/` with 100% correct tracking
   - Validation script created

3. **Implemented Multi-Timestep Inference (Option B)**
   - `num_inference_steps` parameter in SNNControllerAgent
   - Follows literature recommendation [van Breukelen 2025]

4. **Created Clean Benchmark API**
   - `benchmark/controller_interface.py`
   - NeuroBench-compatible `run_benchmark()` function
   - Tested with PI controller: 78mA RMSE, 2.4ms settling



## Next Priority Tasks

1. **Full Training Run with Clean Data** (WP3.3)
   - Command: `poetry run python -m snn.train --epochs 100`
   - Uses clean `train_v2/` data (1000 files, 0.000000A error)

2. **Benchmark Comparison** (WP4) ← READY
   - Command: `poetry run python -m benchmark.run_benchmark --compare`
   - PI baseline validated, SNN awaiting training

3. **Document Results for Thesis**
   - Step response plots
   - Metrics tables
   - Neuromorphic efficiency analysis



## Architecture Notes

### Rate-Coded PI with Membrane Potential Readout (Current Approach)

Following Zaidel et al. (2021) [PMC7887770], the SNN uses **membrane potential readout** for continuous control output:

- **Hidden layers**: LIF with β=0.9 (fast dynamics, respond to changes)
- **Output layer**: LIF with β=0.995, reset_mechanism="none" (no spike, membrane = output)
- **Output**: Membrane potential directly encodes voltage (no rate decoding needed)

```
Environment → SNN (membrane readout) → Environment
[i_d,i_q,     Hidden (β=0.9) → Output (β=0.995, no reset)   [u_d,u_q]
 e_d,e_q]                       ↓ membrane potential
```

### Key Design Decisions (with References)

| Decision | Choice | Reference |
|----------|--------|-----------|
| Training paradigm | Behavioral Cloning (imitation) | Stroobants et al. (2022) [arXiv:2109.10199] |
| Output encoding | Membrane potential readout | Zaidel et al. (2021) [PMC7887770] |
| Network size | 64 hidden neurons (vs 60-80 in literature) | Stroobants (2023) [ACM 10.1145/3546790] |
| Multi-output | Single network, 2 outputs | Standard multi-output regression |
| Inference timesteps | Configurable (1-N per control step) | van Breukelen (2025) [IMAVS 2025/17] |
| Time constants | β=0.9 hidden, β=0.995 output | Tuned for 10kHz control loop |

### Alternative Architectures (If Needed)

From literature review, if single-network approach shows issues:

1. **Two Separate Networks** [matches classical PI structure]
   - SNN_d: [i_d, e_d] → u_d
   - SNN_q: [i_q, e_q] → u_q
   - Advantage: Decoupled, easier to train

2. **IWTA Integration** [Stroobants 2023, arXiv:2304.08778]
   - Input-Weighted Threshold Adaptation for precise integration
   - 10 neurons vs 30 for position-coded integrator

3. **Position-Coded Output** [Stroobants 2022, arXiv:2109.10199]
   - 15 neurons encode discrete voltage levels
   - WTA decoding for output
   - Used on Loihi for quadrotor altitude control

---

## References

### Primary Papers (PI-to-SNN Imitation Learning)

1. **Stroobants, S. et al. (2022)**. "Parsimonious Neuromorphic PID for Quadrotor Altitude Control."
   - arXiv:2109.10199
   - 93 neurons, position-coded, Loihi deployment
   - Demonstrated 100× energy savings vs ARM Cortex-M4

2. **Stroobants, S. et al. (2023)**. "Design and implementation of N-PID on Loihi."
   - ACM doi:10.1145/3546790.3546799
   - IWTA mechanism for integration
   - Hardware implementation details

3. **Zaidel, Y. et al. (2021)**. "Neuromorphic NEF-Based Inverse Kinematics and PID Control."
   - PMC7887770 (Front. Neurorobot.)
   - Rate-coded PI with membrane potential readout
   - 250-500 neurons per axis

4. **van Breukelen Castillo, M.F. (2025)**. "SNNs for High-Speed Continuous Control."
   - IMAVS 2025, Paper 17
   - Multiple integration cycles per control step
   - Hybrid spike-rate decoding

### Supporting Papers

5. **Burgers, T. et al. (2023)**. "Evolving SNNs to Mimic PID Control for Autonomous Blimps."
   - arXiv:2309.12937
   - Evolutionary approach to SNN control

6. **Paredes-Vallés, F. et al. (2024)**. "Fully Neuromorphic Vision and Control."
   - Science Robotics, doi:10.1126/scirobotics.adi0591
   - End-to-end neuromorphic drone control

---

Last Updated: 2026-01-23
