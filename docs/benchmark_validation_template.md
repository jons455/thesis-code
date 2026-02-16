# EMBARK Benchmark Pipeline Validation Report

**Date:** [YYYY-MM-DD]  
**Evaluator:** [Your Name]  
**Benchmark Version:** [e.g., v1.0.0]  
**Commit Hash:** [git commit hash]

---

## Executive Summary

**Primary Goal:** Validate that the EMBARK benchmark pipeline reliably distinguishes controller performance, captures relevant metrics, and integrates hardware-in-the-loop (HIL) inference correctly.

**Models Used:**
- **Best performer:** [model name/version] - Purpose: demonstrate benchmark captures good performance
- **Intermediate performer:** [model name/version] - Purpose: show benchmark distinguishes mid-level quality
- **Poor/failing performer:** [model name/version] - Purpose: validate benchmark flags failures
- **HIL model:** [model name/version] - Purpose: validate Akida hardware integration

**Key Findings:** [2-3 sentences summarizing whether the benchmark passed validation]

---

## 1. Discriminative Power Validation

### Hypothesis H1
**The benchmark produces significantly different metrics for good vs intermediate vs poor controllers, with consistent ranking across scenarios.**

### 1.1 Multi-Model Performance Comparison

Run the full benchmark suite (6 scenarios) for all three models and record mean values:

| Model | Scenario | MAE_q (A) | ITAE_q (A·s²) | Settling Time (ms) | Overshoot (%) | Safety Violations |
|-------|----------|-----------|---------------|-------------------|---------------|-------------------|
| **Best** | step_low_speed_500rpm_2A | | | | | |
| | step_mid_speed_1500rpm_2A ⭐ | | | | | |
| | step_high_speed_2500rpm_2A | | | | | |
| | multi_step_bidirectional_1500rpm | | | | | |
| | four_quadrant_transition_1500rpm | | | | | |
| | field_weakening_2500rpm | | | | | |
| **Intermediate** | step_low_speed_500rpm_2A | | | | | |
| | step_mid_speed_1500rpm_2A ⭐ | | | | | |
| | step_high_speed_2500rpm_2A | | | | | |
| | multi_step_bidirectional_1500rpm | | | | | |
| | four_quadrant_transition_1500rpm | | | | | |
| | field_weakening_2500rpm | | | | | |
| **Poor** | step_low_speed_500rpm_2A | | | | | |
| | step_mid_speed_1500rpm_2A ⭐ | | | | | |
| | step_high_speed_2500rpm_2A | | | | | |
| | multi_step_bidirectional_1500rpm | | | | | |
| | four_quadrant_transition_1500rpm | | | | | |
| | field_weakening_2500rpm | | | | | |

**Note:** ⭐ = Primary reference scenario for detailed comparisons

### 1.2 Ranking Consistency

**In how many scenarios does the expected ranking (Best < Intermediate < Poor) hold for MAE_q?**  
Answer: ___ / 6 scenarios

**Success criterion:** ≥5/6 scenarios show consistent ranking  
**Result:** [ ] PASS  [ ] FAIL

**Are performance gaps substantial (>30% difference between best and intermediate)?**  
Answer: [ ] YES  [ ] NO  
Details: ___

**Does the ranking hold for ITAE_q (transient performance)?**  
Answer: ___ / 6 scenarios show consistent ranking

### 1.3 Statistical Significance

**Run-to-run variability:** Run the mid-speed scenario 10 times and compute coefficient of variation (CV = std/mean) for MAE_q:
- Best model, step_mid_speed_1500rpm_2A: CV = ___
- Intermediate model, step_mid_speed_1500rpm_2A: CV = ___
- Poor model, step_mid_speed_1500rpm_2A: CV = ___

**Success criterion:** CV < 0.10 for all models (demonstrates deterministic behavior)  
**Result:** [ ] PASS  [ ] FAIL

**Are differences statistically significant?**  
Perform paired t-test on step_mid_speed_1500rpm_2A MAE_q (best vs intermediate):
- t-statistic: ___
- p-value: ___
- Significant at α=0.05? [ ] YES  [ ] NO

**Note:** Low CV is expected since the benchmark is deterministic (no noise by default)

---

## 2. Failure Mode Detection

### Hypothesis H2
**The benchmark flags catastrophic controller failures through multiple independent metrics.**

### 2.1 Poor Model Failure Characterization

**Which scenario(s) does the poor model fail on?**  
Answer: ___

**For the worst-performing scenario, record:**
- MAE_q: ___ A
- ITAE_q: ___ A·s²
- Max_error_q: ___ A
- Settling_time_i_q: ___ (s or inf)
- RMS_q (steady-state): ___ A
- Safety violations: ___ (count)
- Violation reason: ___

**Are multiple failure indicators present?**
- [ ] MAE_q >1.0 A (>10× typical PI baseline)
- [ ] Safety violations >0
- [ ] Settling_time_i_q = inf (never settles)
- [ ] Max_error_q >5 A (>2× step size)
- [ ] ITAE_q >10 A·s² (poor transient)

**Success criterion:** ≥3 indicators triggered  
**Result:** [ ] PASS  [ ] FAIL

### 2.2 Failure Pattern Analysis

**When does the failure occur?**
- [ ] Immediately at step 0
- [ ] After transient (~50-100 steps)
- [ ] During steady-state (>200 steps)
- [ ] Only on specific transitions (e.g., negative steps)

**What is the failure mode?**
- [ ] Oscillation
- [ ] Divergence (growing error)
- [ ] Saturation (controller output maxed)
- [ ] Sign-flip instability
- [ ] Other: ___

**Does the failure align with known model limitations?**  
Answer: ___

---

## 3. Construct Validity (Metrics Measure What They Claim)

### Hypothesis H3
**Metrics align with physical expectations and engineering intuition.**

### 3.1 Metric Correlation Analysis

Run the best model across all 6 scenarios and compute correlation between metrics:

| Metric Pair | Pearson r | Expected Relationship | Validates? |
|-------------|-----------|----------------------|------------|
| MAE_q vs ITAE_q | | Strong positive (r >0.8) | [ ] YES  [ ] NO |
| MAE_q vs Max_error_q | | Strong positive (r >0.7) | [ ] YES  [ ] NO |
| Settling_time vs Overshoot | | Moderate positive (r >0.4) | [ ] YES  [ ] NO |
| MAE_q vs RMS_q | | Weak (RMS is steady-state only) | [ ] YES  [ ] NO |
| ITAE_q vs Settling_time | | Moderate positive (r >0.5) | [ ] YES  [ ] NO |

**Success criterion:** All expected relationships hold  
**Result:** [ ] PASS  [ ] FAIL

**Note:** MAE is full-episode average, RMS is steady-state only (after 50ms), so weak correlation is expected

### 3.2 Scenario Difficulty Gradient

**Does MAE_q increase with speed for the best model?**

| Scenario | Speed (rpm) | MAE_q (A) | Expected Ordering |
|----------|-------------|-----------|-------------------|
| step_low_speed_500rpm_2A | 500 | | Easiest (low back-EMF) |
| step_mid_speed_1500rpm_2A | 1500 | | ↓ |
| step_high_speed_2500rpm_2A | 2500 | | Hardest (voltage limits) |

**Is there a monotonic increase in difficulty?**  
Answer: [ ] YES  [ ] NO  [ ] MOSTLY (explain): ___

**Success criterion:** Clear difficulty gradient visible OR low-speed is hardest (parameter sensitivity)  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Low-speed may actually be harder due to parameter sensitivity; high-speed is limited by voltage saturation

### 3.3 Neuromorphic Metrics Validity (SNN Models Only)

**For each SNN model, record (from NeuroBench adapters):**

| Model | Parameters | Footprint (KB) | Activation Sparsity (%) | SyOps/step | Total SyOps | Physically Reasonable? |
|-------|------------|----------------|-------------------------|------------|-------------|------------------------|
| Best | | | | | | [ ] YES  [ ] NO |
| Intermediate | | | | | | [ ] YES  [ ] NO |
| Poor | | | | | | [ ] YES  [ ] NO |

**Checks:**
- [ ] Activation sparsity in 50-90% range (typical for rate-coded SNNs)
- [ ] Footprint ≈ 4 bytes/param (FP32) or 1-2 bytes/param (quantized)
- [ ] SyOps/step scales with network size (larger networks → more ops)
- [ ] SyOps/step < theoretical max (parameters × rate_steps)
- [ ] Connection sparsity reported (if using sparse weights)

**Expected ranges for typical rate-SNN:**
- Input size: 5-13 features
- Hidden layers: 32-128 neurons × 2-3 layers
- SyOps/step: 10k-500k (depending on size and sparsity)
- Footprint: 10-200 KB

**Success criterion:** All checks pass  
**Result:** [ ] PASS  [ ] FAIL

---

## 4. Reproducibility

### Hypothesis H4
**Results are deterministic and consistent across runs and configurations.**

### 4.1 Run-to-Run Consistency

**Run the best model on step_mid_speed_1500rpm_2A ten times consecutively:**

| Run | MAE_q (A) | ITAE_q (A·s²) | Settling Time (s) | Overshoot (%) |
|-----|-----------|---------------|-------------------|---------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |
| 9 | | | | |
| 10 | | | | |
| **Mean** | | | | |
| **Std Dev** | | | | |
| **CV (%)** | | | | |

**Success criterion:** CV <1% for all metrics (benchmark is deterministic by default)  
**Result:** [ ] PASS  [ ] FAIL

**Note:** If CV >1%, check for:
- Non-deterministic model (dropout, stochastic neurons)
- Random seed not set
- Floating-point numerical instability

### 4.2 Configuration Sensitivity

**Run the best model on step_mid_speed_1500rpm_2A with BenchmarkConfig variations:**

| Configuration | MAE_q (A) | Change vs Baseline | Ranking Preserved? |
|---------------|-----------|-------------------|-------------------|
| Baseline (tau=1e-4, no noise, no dead-time) | | — | — |
| noise_current_std=0.1 | | | [ ] YES  [ ] NO |
| noise_current_std=0.5 | | | [ ] YES  [ ] NO |
| noise_speed_std=1.0 | | | [ ] YES  [ ] NO |
| use_dead_time=True | | | [ ] YES  [ ] NO |
| tau=2e-4 (5 kHz instead of 10 kHz) | | | [ ] YES  [ ] NO |

**Observations:**
- Does performance degrade smoothly with added noise? [ ] YES  [ ] NO
- Does ranking (best > intermediate > poor) hold under all configs? [ ] YES  [ ] NO
- Are changes physically reasonable? [ ] YES  [ ] NO
- Is config serialized in results JSON? [ ] YES  [ ] NO

**Success criterion:** Rankings stable, smooth degradation, config reproducible  
**Result:** [ ] PASS  [ ] FAIL

### 4.3 Platform Independence (Optional)

**If testing on multiple platforms, run best model on step_mid_speed_1500rpm_2A:**

| Platform | OS | Python | PyTorch | gym-electric-motor | MAE_q (A) | Difference from Linux |
|----------|----|---------|---------|--------------------|-----------|----------------------|
| Linux | | | | | | — |
| Windows | | | | | | ___ % |
| macOS | | | | | | ___ % |

**Success criterion:** Difference <1% across platforms (deterministic simulation)  
**Result:** [ ] PASS  [ ] FAIL  [ ] NOT TESTED

**Note:** Small differences (<0.1%) may occur due to floating-point implementation differences

---

## 5. External Validity (Scenario Coverage)

### 5.1 Operating Regime Coverage

**Do the 6 scenarios span the key control dimensions?**

| Dimension | Coverage | Scenarios |
|-----------|----------|-----------|
| Speed range | [ ] Low (500 RPM) [ ] Mid (1500 RPM) [ ] High (2500 RPM) | 1, 2, 3, 4, 5, 6 |
| Torque direction | [ ] Positive [ ] Negative [ ] Zero-crossing | 4, 5 |
| Transient type | [ ] Single step [ ] Multi-step [ ] Torque reversal | 1-3, 4, 5 |
| d-axis control | [ ] i_d=0 (standard) [ ] Field-weakening (i_d<0) | All, 6 |
| Current magnitude | [ ] 2A [ ] Variable | All |

**Coverage completeness:**
- Speed envelope: 500-2500 RPM ✓
- Four-quadrant operation: Motoring, generating, braking ✓
- d-q coupling: Field-weakening scenario ✓
- Dynamic tracking: Multi-step and four-quadrant ✓

**Missing coverage (if any):**
- Sinusoidal tracking (bandwidth characterization)
- Variable speed (constant speed per scenario)
- Load disturbances (deterministic simulation)
- Parameter variations (fixed motor parameters)

**Are there important operating points not covered?**  
Answer: ___

### 5.2 Safety Boundary Testing

**Do scenarios push controllers toward realistic limits?**

| Limit | Max Value in Scenarios | Default Limit | Fraction Used | Controllers Hitting Limit |
|-------|------------------------|---------------|---------------|---------------------------|
| Current (A) | 2.0 (step), up to 2.83 (field-weak) | 20 A | ~10-14% | |
| Voltage (V) | Depends on speed/current | 48 V (DC bus) | ~50-80% at 2500 RPM | |
| Speed (rpm) | 2500 | 4000 | 62.5% | |

**Observations:**
- Current limits are conservative (2A << 20A rated) - focuses on control quality, not saturation
- Voltage limits are stressed at high speed (2500 RPM) - realistic for field-weakening
- Speed range covers typical PMSM operating envelope

**Success criterion:** High-speed scenarios approach voltage limits (>60% utilization)  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Check `max_voltage_utilization` metric in high-speed scenarios (3, 6)

---

## 6. HIL Integration Validation (Akida Server)

### Hypothesis H5 & H6
**Akida HIL produces similar control performance to simulation, with accurately captured latency metrics.**

**HIL Model Used:** [model name/version]  
**Akida Hardware:** [e.g., Akida 1.0, Akida 2.0]  
**Server Setup:** [IP:port, network topology]  
**Quantization:** [e.g., 8-bit, 4-bit]

### 6.1 HIL End-to-End Functionality

**Connection and inference checks:**
- [ ] TCP connection to Akida server establishes successfully
- [ ] Model loads on Akida without errors
- [ ] All 6 scenarios complete without crashes
- [ ] Results saved to JSON with HIL metadata
- [ ] Connection cleanup occurs properly

**Success criterion:** All checks pass  
**Result:** [ ] PASS  [ ] FAIL

### 6.2 HIL vs Simulation Performance Comparison

**Run the HIL model on step_mid_speed_1500rpm_2A in both modes:**

| Mode | MAE_q (A) | ITAE_q (A·s²) | Settling Time (s) | Overshoot (%) | Safety Violations |
|------|-----------|---------------|-------------------|---------------|-------------------|
| PyTorch Simulation | | | | | |
| Akida HIL | | | | | |
| **Difference (%)** | | | | | |

**Success criterion:** Difference <20%, no new safety violations  
**Result:** [ ] PASS  [ ] FAIL

**If difference >20%, investigate:**
- [ ] Quantization effects (check weight/activation bit-width: 8-bit, 4-bit, 2-bit)
- [ ] Preprocessing mismatch (error_gain, n_max normalization constants)
- [ ] State processor configuration (feature flags must match training)
- [ ] Action processor mode (incremental vs absolute)
- [ ] Network topology difference (layers, connections)
- [ ] Inference correctness (compare step-by-step outputs for first 100 steps)

**Investigation notes:** ___

**Quantization validation:**
- [ ] Compare weight distributions (PyTorch FP32 vs Akida quantized)
- [ ] Check for activation saturation in quantized model
- [ ] Verify input normalization matches exactly (error_gain=10.0, n_max=4000.0)

### 6.3 Latency Metrics Capture

**For a single Akida HIL run on step_mid_speed_1500rpm_2A, extract latency metrics from JSON:**

**Round-trip latency (includes network + chip):**
| Metric | Value | Unit | Expected Range | Valid? |
|--------|-------|------|----------------|--------|
| mean_latency_ms | | ms | 0.5 - 10 | [ ] YES  [ ] NO |
| p95_latency_ms | | ms | 1 - 15 | [ ] YES  [ ] NO |
| p99_latency_ms | | ms | 2 - 20 | [ ] YES  [ ] NO |
| max_latency_ms | | ms | 3 - 50 | [ ] YES  [ ] NO |
| jitter_ms (std dev) | | ms | <5 | [ ] YES  [ ] NO |
| total_inference_time_s | | s | ~0.3 (3000 steps × latency) | [ ] YES  [ ] NO |

**On-chip latency (if available from controller_info):**
| Metric | Value | Unit | Expected Range | Valid? |
|--------|-------|------|----------------|--------|
| chip_mean_us | | µs | 10 - 500 | [ ] YES  [ ] NO |
| chip_median_us | | µs | 10 - 500 | [ ] YES  [ ] NO |
| chip_p95_us | | µs | 20 - 800 | [ ] YES  [ ] NO |
| chip_p99_us | | µs | 30 - 1000 | [ ] YES  [ ] NO |
| chip_max_us | | µs | 50 - 2000 | [ ] YES  [ ] NO |
| chip_min_us | | µs | 5 - 100 | [ ] YES  [ ] NO |

**Are all latency fields populated (non-zero)?**  
Answer: [ ] YES  [ ] NO  
If NO, which fields are missing? ___

**Success criterion:** All required metrics present and in valid ranges  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Round-trip includes TCP/IP overhead; on-chip latency is the actual inference time on Akida

### 6.4 Real-Time Constraint Validation

**For 10 kHz control (100 µs cycle time = 0.1 ms), check on-chip latency:**
- Chip mean latency <50 µs (half the cycle): [ ] PASS  [ ] FAIL  
  Actual: ___ µs
- Chip p99 latency <100 µs (one cycle): [ ] PASS  [ ] FAIL  
  Actual: ___ µs

**For round-trip latency (including network):**
- Mean latency <5 ms (acceptable for HIL testing): [ ] PASS  [ ] FAIL  
  Actual: ___ ms
- P99 latency <10 ms: [ ] PASS  [ ] FAIL  
  Actual: ___ ms

**Does the benchmark flag timing violations?**  
Answer: [ ] YES  [ ] NO  [ ] N/A (no violations occurred)

**Success criterion:** On-chip latency meets real-time requirements (<100 µs) OR violations are flagged  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Round-trip latency includes TCP/IP overhead and is not real-time. On-chip latency is the critical metric for deployment.

### 6.5 Latency-Performance Correlation

**Run HIL model across all 6 scenarios and analyze latency vs performance:**

| Scenario | MAE_q (A) | chip_p99_us (µs) | Round-trip p99_ms (ms) |
|----------|-----------|------------------|------------------------|
| step_low_speed_500rpm_2A | | | |
| step_mid_speed_1500rpm_2A | | | |
| step_high_speed_2500rpm_2A | | | |
| multi_step_bidirectional_1500rpm | | | |
| four_quadrant_transition_1500rpm | | | |
| field_weakening_2500rpm | | | |

**Observations:**
- Is latency consistent across scenarios? [ ] YES  [ ] NO (should be consistent for same model)
- Is performance degradation vs simulation small? [ ] YES  [ ] NO (<20% acceptable)
- Does higher latency correlate with worse tracking? [ ] YES  [ ] NO  [ ] N/A (latency is constant)

**Success criterion:** Latency consistent across scenarios, performance degradation <20%  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Latency should be nearly constant across scenarios (same model, same network size). Performance differences are due to scenario difficulty, not latency variation.

### 6.6 HIL Reproducibility

**Repeat HIL run 5 times on step_mid_speed_1500rpm_2A:**

| Run | MAE_q (A) | chip_mean_us (µs) | chip_p99_us (µs) | Round-trip mean_ms (ms) |
|-----|-----------|-------------------|------------------|-------------------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| **Mean** | | | | |
| **Std Dev** | | | | |
| **CV (%)** | | | | |

**Success criterion:** CV <5% for performance, CV <20% for latency  
**Result:** [ ] PASS  [ ] FAIL

**Note:** 
- Performance (MAE_q) should be highly reproducible (<5% CV) - same model, deterministic simulation
- Latency may vary due to network conditions and OS scheduling (10-20% CV acceptable)
- On-chip latency should be more stable than round-trip latency

### 6.7 HIL Documentation Completeness

**Can someone else reproduce the setup?**
- [ ] Akida server version documented
- [ ] Network configuration documented (IP, port, firewall rules)
- [ ] Model quantization/compilation steps documented
- [ ] Driver/SDK versions recorded in results JSON
- [ ] Troubleshooting guide exists for common issues

**Success criterion:** ≥4/5 items documented  
**Result:** [ ] PASS  [ ] FAIL

---

## 7. Benchmark Usability

### 7.1 Runtime Performance

**Measure wall-clock time for full suite (6 scenarios with standard durations):**

| Model | Runtime (seconds) | Acceptable? (<2 min target) |
|-------|-------------------|-----------------------------|
| Best (SNN) | | [ ] YES  [ ] NO |
| Intermediate (SNN) | | [ ] YES  [ ] NO |
| Poor (SNN) | | [ ] YES  [ ] NO |
| PI Baseline | | [ ] YES  [ ] NO |
| HIL (Akida) | | [ ] YES  [ ] NO |

**Scenario durations (for reference):**
- Scenarios 1-3: 3000 steps (0.3s each)
- Scenario 4: 10000 steps (1.0s)
- Scenario 5: 9000 steps (0.9s)
- Scenario 6: 6000 steps (0.6s)
- **Total simulation time:** 3.7s across all scenarios

**Quick scenarios (QUICK_SCENARIOS - 2 scenarios) runtime:** ___ seconds

**Success criterion:** Full suite <2 min, Quick <30 sec  
**Result:** [ ] PASS  [ ] FAIL

**Note:** Runtime depends on model complexity, not episode length. HIL adds network latency.

### 7.2 Output Clarity

**Give the printed summary table to a colleague unfamiliar with the project.**

Questions to ask them:
1. Can you identify which model is best? [ ] YES  [ ] NO
2. Can you tell which scenario is hardest? [ ] YES  [ ] NO
3. Can you spot the failing model? [ ] YES  [ ] NO
4. Is the output overwhelming (too many metrics)? [ ] YES  [ ] NO

**Success criterion:** ≥3/4 YES (except question 4 should be NO)  
**Result:** [ ] PASS  [ ] FAIL

### 7.3 Error Handling

**Test error cases:**
- [ ] Missing model checkpoint → clear error message
- [ ] Wrong input size (processor dim ≠ model input) → helpful message explaining mismatch
- [ ] Akida server unreachable → graceful timeout and error
- [ ] Invalid BenchmarkConfig parameter → validation error before running
- [ ] Safety violation during run → scenario terminates with clear reason
- [ ] Missing required checkpoint keys (input_size, state_dict) → clear error

**Example error messages to test:**
```python
# Wrong input size
state_processor.output_dim = 5
model.input_size = 12
# Should error: "Processor output dim (5) != model input (12)"

# Missing checkpoint key
checkpoint = {"state_dict": {...}}  # missing "input_size"
# Should error: "Checkpoint missing required key: input_size"
```

**Success criterion:** All error cases handled gracefully with actionable messages  
**Result:** [ ] PASS  [ ] FAIL

---

## 8. Overall Validation Summary

### Core Hypotheses Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Discriminative power | [ ] PASS  [ ] FAIL | Ranking consistent in ___/6 scenarios |
| H2: Failure detection | [ ] PASS  [ ] FAIL | ___/4 failure indicators triggered |
| H3: Construct validity | [ ] PASS  [ ] FAIL | Metrics correlate as expected |
| H4: Reproducibility | [ ] PASS  [ ] FAIL | CV <5% across runs |
| H5: HIL correctness | [ ] PASS  [ ] FAIL | Performance difference <20% |
| H6: HIL timing capture | [ ] PASS  [ ] FAIL | All latency metrics valid |

**Overall Benchmark Validation:** [ ] PASS  [ ] FAIL

### Critical Issues Found

1. ___
2. ___
3. ___

### Recommended Actions

- [ ] Issue #1: ___
- [ ] Issue #2: ___
- [ ] Issue #3: ___

### Conclusion

[2-3 paragraphs summarizing whether the benchmark is scientifically valid and ready for publication/public use. Address: (1) Does it reliably distinguish controllers? (2) Are metrics trustworthy? (3) Does HIL work correctly? (4) What limitations remain?]

---

## Appendix: Detailed Results

### A. Full Metric Tables

[Paste complete JSON outputs or CSV files here]

**Example metrics to include for each scenario:**
- Control performance: `mae_i_q`, `mae_i_d`, `itae_i_q`, `itae_i_d`, `max_error_i_q`, `max_error_i_d`
- Dynamics: `settling_time_i_q`, `overshoot`, `rms_i_q`, `rms_i_d`
- Latency: `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `chip_mean_us`, `chip_p99_us`
- Neuromorphic: `total_syops`, `syops_per_step`, `activation_sparsity`, `footprint`, `connection_sparsity`
- Episode info: `steps`, `safety_terminated`, `violation_reason`

### B. Configuration Used

```yaml
# BenchmarkConfig
tau: 1e-4  # 10 kHz control frequency (100 µs sampling)
noise_current_std: 0.0  # No noise (default)
noise_speed_std: 0.0    # No noise (default)
use_dead_time: false    # No dead-time (default)

# PMSMConfig (motor parameters)
motor_type: "PMSM"
R_s: 0.78  # Ω (stator resistance)
L_d: 0.0027  # H (d-axis inductance)
L_q: 0.0027  # H (q-axis inductance)
psi_p: 0.0155  # Wb (permanent magnet flux)
J: 0.001  # kg·m² (moment of inertia)
u_dc: 48.0  # V (DC bus voltage)

# Safety limits
max_current_a: 20.0  # A
max_voltage_v: 48.0  # V (u_dc)
n_max_rpm: 4000.0  # RPM
```

### C. Hardware/Software Environment

**Simulation environment:**
- **CPU:** ___
- **RAM:** ___
- **OS:** ___
- **Python:** ___ (e.g., 3.9, 3.10, 3.11)
- **PyTorch:** ___ (e.g., 2.0.0, 2.1.0)
- **gym-electric-motor:** ___ (e.g., 1.0.0)
- **snntorch:** ___ (if using snnTorch models)
- **EMBARK version:** ___ (git commit hash)

**HIL environment (if applicable):**
- **Akida hardware:** ___ (e.g., Akida 1.0, Akida 2.0)
- **Akida SDK version:** ___
- **Network setup:** ___ (e.g., TCP IP:port, local/remote)
- **Quantization:** ___ (e.g., 8-bit, 4-bit, 2-bit)
- **Model compilation date:** ___

### D. Random Seeds Used

[List all random seeds for reproducibility]

---

# User Guide: How Often to Run the Benchmark Suite

## 1. Development Phase (Active Model Development)

### Quick Evaluation (QUICK_SCENARIOS - 2 scenarios)
**Frequency:** After every major code change  
**When to run:**
- Modified network architecture (layers, neurons, encoding)
- Changed training hyperparameters
- Updated state/action processors (feature flags, normalization)
- Fixed bugs in controller logic

**Runtime:** ~15-30 seconds per model  
**Purpose:** Rapid feedback on whether changes improve/degrade performance

**Command:**
```python
from embark.benchmark import BenchmarkSuite, QUICK_SCENARIOS

suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=True)
summary = suite.run(controller=my_controller, name="MySNN-dev")
suite.print_summary(summary)
```

---

### Full Evaluation (STANDARD_SCENARIOS - 6 scenarios)
**Frequency:** Before committing to main branch  
**When to run:**
- Completed a training run you want to keep
- Before publishing results or writing a paper section
- End of each development sprint/week
- Before tagging a release version

**Runtime:** ~1-2 minutes per model  
**Purpose:** Comprehensive coverage across all operating conditions

**Command:**
```python
from embark.benchmark import BenchmarkSuite, STANDARD_SCENARIOS

suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS, verbose=True)
summary = suite.run(controller=my_controller, name="MySNN-v1")
suite.print_summary(summary)
suite.save_results(summary, "results/mysnn_v1_benchmark.json")
```

---

## 2. Benchmark Pipeline Validation

### Initial Validation (This Checklist)
**Frequency:** Once per benchmark version  
**When to run:**
- After implementing new metrics
- After adding/modifying scenarios
- After major refactoring of the evaluation harness
- Before submitting a benchmark paper

**Time required:** 2-4 hours (includes analysis and filling this report)  
**Purpose:** Ensure the benchmark itself is scientifically valid

**What to run:**
1. Full evaluation for 3 diverse models (best, intermediate, poor)
2. HIL evaluation for 1 model
3. Reproducibility tests (10 consecutive runs)
4. Configuration sensitivity tests (4-5 configs)

---

### Regression Testing
**Frequency:** After any changes to benchmark code  
**When to run:**
- Modified scenario definitions (ScenarioDefinition changes)
- Changed metric calculations (MetricAccumulator implementations)
- Updated physics engine parameters (PMSMConfig)
- Changed state/action normalization (processor changes)

**Runtime:** ~2 minutes  
**Purpose:** Ensure changes don't break existing functionality

**What to run:**
```python
# Run a reference model and compare to saved baseline
import json
from embark.benchmark import BenchmarkSuite, STANDARD_SCENARIOS

suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)
summary = suite.run(controller=reference_controller, name="Reference")

# Compare to baseline
with open("results/baseline_v1.0.json", "r") as f:
    baseline = json.load(f)

# Check key metrics haven't changed >1%
for i, scenario in enumerate(summary.scenario_results):
    baseline_mae = baseline["scenarios"][i]["metrics"]["mae_i_q"]
    current_mae = scenario.metrics["mae_i_q"]
    diff_pct = abs(current_mae - baseline_mae) / baseline_mae * 100
    assert diff_pct < 1.0, f"Regression in {scenario.scenario_name}: {diff_pct:.2f}%"
```

---

## 3. Hardware-in-the-Loop Testing

### HIL Development Testing
**Frequency:** After changes to deployment pipeline  
**When to run:**
- Modified Akida model compilation
- Changed quantization scheme
- Updated network connection code (TCP client)
- Changed latency measurement logic (controller_info dict)

**Runtime:** ~2-5 minutes  
**Purpose:** Verify HIL integration still works correctly

**What to run:**
```python
# Single scenario for quick feedback
from embark.benchmark import BenchmarkSuite, STANDARD_SCENARIOS

# Use only mid-speed scenario for quick test
quick_hil_test = [STANDARD_SCENARIOS[1]]  # step_mid_speed_1500rpm_2A

suite = BenchmarkSuite(scenarios=quick_hil_test, verbose=True)
summary = suite.run(controller=akida_hil_controller, name="Akida-HIL-test")

# Verify latency metrics are captured
assert summary.scenario_results[0].metrics["mean_latency_ms"] > 0
assert summary.scenario_results[0].metrics["chip_mean_us"] > 0
print(f"HIL test passed: mean latency = {summary.scenario_results[0].metrics['mean_latency_ms']:.2f} ms")
```

---

### HIL Full Validation
**Frequency:** Once per hardware/firmware version  
**When to run:**
- Updated Akida firmware
- Changed hardware setup (new chip, network topology)
- Before reporting HIL results in a paper

**Runtime:** ~5-10 minutes  
**Purpose:** Complete latency characterization and performance validation

**What to run:**
```python
# Full suite with all 6 scenarios
from embark.benchmark import BenchmarkSuite, STANDARD_SCENARIOS

suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS, verbose=True)
summary = suite.run(controller=akida_hil_controller, name="Akida-HIL-full")

# Save results with latency analysis
suite.save_results(summary, "results/akida_hil_full_validation.json")

# Verify all latency metrics captured
for result in summary.scenario_results:
    assert result.metrics["mean_latency_ms"] > 0, f"Missing latency in {result.scenario_name}"
    assert result.metrics["chip_mean_us"] > 0, f"Missing chip timing in {result.scenario_name}"
    
print(f"HIL validation complete: {len(summary.scenario_results)} scenarios")
```

---

## 4. Multi-Model Comparison

### Comparative Evaluation
**Frequency:** When comparing different approaches  
**When to run:**
- Comparing training methods (open-loop vs closed-loop)
- Comparing architectures (different hidden sizes, rate_steps)
- Ablation studies (with/without specific input features)
- Benchmarking against baselines (PI controller)

**Runtime:** ~5-10 minutes for 3-5 models  
**Purpose:** Generate publication-ready comparison tables

**What to run:**
```python
from embark.benchmark import BenchmarkSuite, STANDARD_SCENARIOS, PIControllerAgent

# Create controllers
controllers = [
    ("SNN-v5-basic", snn_v5_controller),
    ("SNN-v10-temporal", snn_v10_controller),
    ("SNN-v12-full", snn_v12_controller),
    ("PI-baseline", pi_controller),
]

suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS, verbose=True)
results = []

for name, controller in controllers:
    summary = suite.run(controller=controller, name=name)
    results.append(summary)
    suite.save_results(summary, f"results/{name}_comparison.json")

# Print comparison table
print("\n=== Model Comparison ===")
print(f"{'Model':<20} {'MAE_q (A)':<12} {'Settling (s)':<12} {'SyOps/step':<12}")
for summary in results:
    scenario_2 = summary.scenario_results[1]  # mid-speed reference scenario
    mae = scenario_2.metrics["mae_i_q"]
    settling = scenario_2.metrics.get("settling_time_i_q", float("inf"))
    syops = scenario_2.metrics.get("syops_per_step", 0)
    print(f"{summary.controller_name:<20} {mae:<12.4f} {settling:<12.4f} {syops:<12.0f}")
```

---

## 5. Paper/Publication Workflow

### Pre-Submission Validation
**Frequency:** Before submitting a paper  
**Checklist:**
1. [ ] Run full evaluation (10+ runs) for all models in paper
2. [ ] Complete this validation checklist for the benchmark
3. [ ] Run HIL evaluation if claiming hardware results
4. [ ] Verify reproducibility (repeat key experiments 2-3 times)
5. [ ] Check that all figures/tables match latest results
6. [ ] Ensure configuration and random seeds are documented

**Time required:** 1-2 days  
**Purpose:** Ensure all reported numbers are current, correct, and reproducible

---

### Post-Publication Archival
**Frequency:** Once per published paper  
**What to save:**
- [ ] All model checkpoints used in paper
- [ ] Complete JSON results for all evaluations
- [ ] Exact benchmark version (git commit hash)
- [ ] Configuration files and random seeds
- [ ] README with reproduction instructions

**Purpose:** Enable others (and future you) to reproduce results

---

## 6. Continuous Integration (Optional)

### Automated Nightly Builds
**Frequency:** Every night (if using CI/CD)  
**What to run:**
- Quick evaluation on reference models
- Basic functionality tests
- Performance regression checks

**Purpose:** Catch breaking changes early

**Example `.github/workflows/benchmark_ci.yml`:**
```yaml
name: Benchmark Tests
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run quick benchmark
        run: |
          poetry install
          poetry run python -m evaluation.analysis.evaluate_rate_snn \
            --model evaluation/trained_models/reference_model.pt \
            --quick \
            --compare-baseline results/baseline.json
```

---

## Summary Table: When to Run What

| Scenario | Frequency | Runtime | Mode | Models |
|----------|-----------|---------|------|--------|
| **Daily development** | After each code change | 15-30s | QUICK_SCENARIOS (2) | 1 (your current model) |
| **Weekly checkpoint** | End of week/sprint | 1-2 min | STANDARD_SCENARIOS (6) | 1-2 (best so far) |
| **Before commit** | Before merge to main | 2-3 min | STANDARD_SCENARIOS (6) | 1 (changed model) |
| **Benchmark validation** | Per benchmark version | 2-4 hrs | Full + checklist | 3-4 (diverse set) |
| **HIL testing** | After deployment changes | 2-5 min | 1 scenario (mid-speed) | 1 (HIL model) |
| **HIL validation** | Per hardware version | 5-10 min | STANDARD_SCENARIOS (6) | 1 (HIL model) |
| **Paper prep** | Before submission | 1-2 days | Full × multiple runs | All paper models |
| **Comparative study** | When comparing approaches | 5-10 min | STANDARD_SCENARIOS (6) | 3-5 (comparison set) |
| **Regression test** | After benchmark changes | 2 min | STANDARD_SCENARIOS (6) | 1 (reference) |

---

## Tips for Efficient Evaluation

1. **Use QUICK_SCENARIOS during iteration:** 2 scenarios (low-speed + mid-speed) give you 80% of the signal in 33% of the time
2. **Cache baseline results:** Save one "known-good" run and compare against it to spot regressions
3. **Focus on primary metrics first:** Don't get lost in 20+ metrics; watch `mae_i_q` and `settling_time_i_q` on scenario 2 (mid-speed ⭐)
4. **Automate common comparisons:** Write scripts for "compare latest vs baseline" workflows
5. **Document why you ran it:** Add metadata to your results: save with descriptive filenames like `mysnn_v10_after_normalization_fix.json`
6. **Use scenario 2 as your reference:** `step_mid_speed_1500rpm_2A` is the primary benchmark - optimize for this first
7. **Check neuromorphic metrics:** For SNNs, verify `activation_sparsity` >50% and `syops_per_step` is reasonable for your network size

---

## Red Flags That Require Immediate Re-Evaluation

Stop and re-run full validation if you see:
- [ ] Ranking flips (model A beats B in QUICK_SCENARIOS, but B beats A in STANDARD_SCENARIOS)
- [ ] Safety violations appear/disappear between runs (should be deterministic)
- [ ] Metrics change >5% with no code changes (benchmark is deterministic)
- [ ] HIL latency suddenly increases >2× (check network/hardware)
- [ ] New scenario added to STANDARD_SCENARIOS
- [ ] Physics parameters changed (PMSMConfig: R_s, L_d, L_q, psi_p, etc.)
- [ ] Metric calculations modified (MetricAccumulator implementations)
- [ ] State/action processor normalization changed (error_gain, n_max, delta_max)
- [ ] Settling time = inf on scenario 2 (mid-speed) - controller is fundamentally broken

---

**Questions or issues with this validation process?**  
Contact: [your email]  
Benchmark repository: [GitHub link]
