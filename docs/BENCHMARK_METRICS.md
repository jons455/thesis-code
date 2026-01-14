# Benchmark Metrics for Neuromorphic PMSM Current Control

**Document Purpose**: Defines the metrics framework for comparing SNN vs PI controllers.

**Date**: 2026-01-14
**Author**: Jonas
**Status**: Final Framework

---

## Executive Summary

This benchmark evaluates **Spiking Neural Network (SNN) controllers** for PMSM current control against conventional PI controllers.

**The Goal**: Find the best trade-off between **Control Fidelity** and **Neuromorphic Efficiency**.

**The Model**: A current controller that learns the mapping:
```
(i_ref, i_ist, n) → (u_d, u_q)
```

**Standardization**: All episodes run for **1.0 second** (10,000 steps at 10 kHz) to ensure comparable metrics.

---

## Metrics Overview

| Category | Metric | Unit | Purpose |
|----------|--------|------|---------|
| **Accuracy** | RMSE | A | Overall tracking quality |
| **Accuracy** | ITAE | A·s² | Detect steady-state drift (SNN weakness) |
| **Dynamics** | Settling Time | ms | Controller agility |
| **Dynamics** | Overshoot | % | Safety check |
| **Stability** | Control Smoothness (TV) | V/step | Detect chattering (SNN weakness) |
| **Neuromorphic** | SyOps/step | ops | Computational cost proxy |
| **Neuromorphic** | Activation Sparsity | % | Efficiency measure |

---

## 1. Control Quality Metrics (Accuracy)

*Does the motor actually do what it is told?*

### 1.1 RMSE (Root Mean Square Error)

**The primary tracking accuracy metric.**

| Property | Value |
|----------|-------|
| Formula | $RMSE = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (i_{ref}[t] - i_{meas}[t])^2}$ |
| Unit | Amperes [A] |
| Better | Lower |
| Normalized | ✅ Yes (independent of episode length) |

```python
error = i_ref - i_meas
rmse = np.sqrt(np.mean(error**2))
```

**Why RMSE?** Industry standard for tracking accuracy. Gives the "average" error magnitude.

**⚠️ Limitation**: Averages out short spikes. A 1ms error disappears in a 1s average.

### 1.2 ITAE (Integral of Time-weighted Absolute Error)

**Critical for detecting SNN integrator drift.**

| Property | Value |
|----------|-------|
| Formula | $ITAE = \sum_{t=1}^{N} t \cdot |e[t]| \cdot dt$ |
| Unit | A·s² |
| Better | Lower |
| Normalized | ⚠️ Depends on episode length → **Standardize to 1.0s** |

```python
# times = [0.0, 0.0001, 0.0002, ...]  (1.0s total)
itae = np.sum(times * np.abs(error) * dt)
```

**Why ITAE?** It penalizes errors that persist over time. If an SNN has slight bias, the error won't go to zero, and ITAE will grow continuously. This exposes the #1 SNN weakness: **steady-state drift**.

**⚠️ Critical**: Episode length MUST be standardized to 1.0s for comparable ITAE values.

### 1.3 Maximum Error (Optional)

**Catches the worst-case moment that RMSE hides.**

| Property | Value |
|----------|-------|
| Formula | $e_{max} = \max(|i_{ref} - i_{meas}|)$ |
| Unit | Amperes [A] |
| Better | Lower |

```python
max_error = np.max(np.abs(error))
```

---

## 2. Dynamic Response Metrics

*How fast does it react?*

### 2.1 Settling Time (T_set)

**The "agility" metric. SNNs often beat PIs here.**

| Property | Value |
|----------|-------|
| Definition | Time for output to enter and stay within ±2% error band |
| Unit | milliseconds [ms] |
| Better | Lower |
| Typical PI | 5-10 ms |
| Target SNN | < 5 ms |

```python
threshold = 0.02 * step_magnitude  # 2% band
outside_band = np.where(np.abs(error) > threshold)[0]
if len(outside_band) > 0:
    t_set = times[outside_band[-1]]
else:
    t_set = 0.0  # Already settled
```

### 2.2 Overshoot (M_p)

**Safety check. Does the SNN "kick" too hard on step changes?**

| Property | Value |
|----------|-------|
| Formula | $M_p = \frac{\max(y) - y_{final}}{y_{final} - y_{initial}} \times 100\%$ |
| Unit | Percent [%] |
| Better | Lower |
| Safe | < 10% |
| Dangerous | > 20% |

```python
peak = np.max(current_after_step)
overshoot = (peak - target) / step_magnitude * 100
```

**⚠️ Noise sensitivity**: A single sensor spike registers as overshoot. 
**Solution**: Apply 5-sample moving average before calculating max.

---

## 3. Stability Metrics (Safety)

*Is the controller destroying the hardware?*

### 3.1 Control Smoothness (Total Variation)

**The "SNN Killer" metric. Measures voltage chattering.**

| Property | Value |
|----------|-------|
| Formula | $TV = \frac{1}{N} \sum_{t=1}^{N} |u[t] - u[t-1]|$ |
| Unit | V/step (normalized) |
| Better | Lower |
| PI Baseline | ~0.01 V/step (smooth) |
| Bad SNN | ~1.0 V/step (chattering) |

```python
voltage_changes = np.diff(voltages)
tv_normalized = np.mean(np.abs(voltage_changes))

# Or compare to PI baseline
tv_ratio = tv_snn / tv_pi  # >1 means SNN is "chattier"
```

**Why this matters:**
- An SNN might track perfectly (low RMSE)
- But oscillate voltage ±24V at 10kHz
- This causes:
  - Torque ripple → mechanical vibration
  - Audible noise
  - Bearing wear
  - Wasted switching energy

**Physical filtering note**: Motor inductance acts as low-pass filter (τ = L/R ≈ 2.6ms, f_cutoff ≈ 61 Hz). Chattering above 60 Hz is electrically filtered but still wastes energy.

### 3.2 Constraint Violations

| Metric | Threshold | Unit |
|--------|-----------|------|
| Current violations | \|I\| > 10.8 A | count |
| Voltage violations | \|U\| > 48 V | count |
| di/dt violations | > 50,000 A/s | count |

---

## 4. Neuromorphic Efficiency Metrics

*Is it actually efficient?*

### 4.1 SyOps (Synaptic Operations)

**The "cost" metric. Proxy for energy consumption.**

| Property | Value |
|----------|-------|
| Formula | $SyOps = \sum_{layers} (SpikeCount_{layer} \times FanOut_{layer})$ |
| Report as | SyOps/step (normalized by episode length) |
| Better | Lower |

```python
# Per layer
syops_layer = spike_count * fan_out

# Total per inference step
syops_per_step = total_syops / num_timesteps
```

**Why SyOps matter:**
- ANNs: Every neuron computes every cycle → O(n²) operations
- SNNs: Only spiking neurons trigger computation → O(k·n) where k << n
- Lower SyOps = Lower energy on neuromorphic hardware

**Energy estimation:**
| Platform | Energy/SyOp | SyOps/step | Energy/step |
|----------|-------------|------------|-------------|
| Loihi 2 | ~23 pJ | 1000 | 23 nJ |
| SpiNNaker 2 | ~10 pJ | 1000 | 10 nJ |
| GPU (comparison) | ~1000 pJ | 1000 | 1 µJ |

### 4.2 Activation Sparsity

**How "silent" is the network?**

| Property | Value |
|----------|-------|
| Formula | $Sparsity = 1 - \frac{TotalSpikes}{TotalNeurons \times TimeSteps}$ |
| Unit | Percent [%] or ratio [0-1] |
| Better | Higher |
| Typical ANN | 0-40% |
| Good SNN | 80-95% |
| Excellent SNN | > 95% |

```python
total_possible = num_neurons * num_timesteps
sparsity = 1.0 - (total_spikes / total_possible)
```

**Why sparsity matters:**
- High sparsity = Fewer computations = Lower power
- At steady state with delta encoding: Sparsity should approach 100%

---

## 5. Benchmark Configuration

### 5.1 Standardized Episode Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Episode length** | **1.0 s** | Makes ITAE, TV comparable |
| Control frequency | 10 kHz | Standard for PMSM |
| Timestep (dt) | 100 µs | 1/10kHz |
| Steps per episode | 10,000 | 1.0s × 10kHz |
| Step response time | 0.0 s | Step at t=0 |

### 5.2 Benchmark Scenarios

| Scenario | i_d_ref | i_q_ref | Speed | Purpose |
|----------|---------|---------|-------|---------|
| Step Low | 0.0 A | 2.0 A | 1000 rpm | Basic tracking |
| Step Mid | 0.0 A | 5.0 A | 1000 rpm | Medium load |
| Step High | 0.0 A | 8.0 A | 1000 rpm | Near limit |
| Speed Sweep | 0.0 A | 2.0 A | 500-2500 rpm | Robustness |
| Flux Weakening | -2.0 A | 5.0 A | 2000 rpm | d-axis control |

### 5.3 Reporting Template

```
=======================================================================
Controller: SNN-Hybrid-64neurons
Episode: 1.0s @ 10kHz (10,000 steps)
Operating Point: id=0A, iq=2A @ 1000rpm
=======================================================================

CONTROL QUALITY
  RMSE (i_q):           0.045 A
  ITAE (i_q):           0.0023 A·s²
  Max Error (i_q):      0.32 A
  
DYNAMICS
  Settling Time:        3.2 ms
  Overshoot:            5.3%
  
STABILITY
  Control Smoothness:   0.08 V/step  (PI baseline: 0.05 V/step)
  TV Ratio vs PI:       1.6×
  Constraint Violations: 0

NEUROMORPHIC EFFICIENCY
  SyOps/step:           1,240
  Activation Sparsity:  92.3%
  Estimated Energy:     28.5 nJ/step (Loihi 2)

COMPARISON TO PI BASELINE
  RMSE ratio:           1.05× (5% worse)
  Settling time ratio:  0.64× (36% faster)
  SyOps reduction:      15× fewer operations
=======================================================================
```

---

## 6. The Trade-off Visualization

The final comparison is a **Pareto front**:

```
Control Quality (RMSE ↓)
        ▲
    0.02│           × PI Baseline (reference)
        │       
    0.04│     × SNN-Large (better control, more SyOps)
        │
    0.06│  × SNN-Medium ← BEST TRADE-OFF
        │
    0.08│      × SNN-Small (worse control, fewer SyOps)
        │
        └──────────────────────────────────────────────►
              500    1000    1500    2000    2500
                    Neuromorphic Cost (SyOps/step ↑)
```

**The winning SNN** is the one closest to the PI baseline in control quality while having significantly lower SyOps.

---

## 7. Implementation Checklist

| Component | File | Status |
|-----------|------|--------|
| RMSE computation | `benchmark_metrics.py` | ✅ |
| ITAE computation | `benchmark_metrics.py` | ✅ |
| Settling time | `benchmark_metrics.py` | ✅ |
| Overshoot | `benchmark_metrics.py` | ✅ |
| **Control Smoothness (TV)** | `benchmark_metrics.py` | ✅ |
| SyOps from spikes | `benchmark_metrics.py` | ✅ |
| Sparsity | `benchmark_metrics.py` | ✅ |
| Episode length config | `benchmark/config.py` | 🔜 |
| Standardized scenarios | `benchmark/scenarios.py` | 🔜 |

---

## References

1. **NeuroBench**: Yik et al., "NeuroBench: A Framework for Benchmarking Neuromorphic Computing", arXiv 2024
2. **gym-electric-motor**: Balakrishnan et al., JOSS 2021
3. **Technical Optimum**: Kessler, 1958 (PI tuning method)
4. **Loihi 2 Energy**: Davies et al., Intel 2021
5. **Total Variation**: Rudin-Osher-Fatemi, 1992 (signal processing)
