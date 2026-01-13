# Benchmark Metrics Framework for Neuromorphic PMSM Control

## Overview

This document describes the metrics framework for evaluating neuromorphic (SNN-based) controllers for Permanent Magnet Synchronous Motor (PMSM) current control. The metrics are organized into **two main categories**:

1. **Control Performance Metrics** - How well does the controller regulate currents?
2. **Neuromorphic Efficiency Metrics** - What are the computational/energy advantages of SNNs?

---

## 📊 Metric Structure

```
BenchmarkResult
├── AccuracyMetrics       → Tracking precision
├── DynamicsMetrics       → Transient response
├── EfficiencyMetrics     → Energy/power efficiency
├── SafetyMetrics         → Constraint handling
└── NeuromorphicMetrics   → SNN-specific (optional)
```

---

## 1. Control Performance Metrics

### 1.1 Accuracy Metrics (AccuracyMetrics)

These metrics quantify how precisely the controller tracks the current references.

#### Integral Error Metrics

| Metric | Formula | Description | Unit |
|--------|---------|-------------|------|
| **ITAE** | $\int_0^T t \cdot |e(t)| \, dt$ | Integral Time-weighted Absolute Error. Penalizes errors more as time progresses. | A·s² |
| **IAE** | $\int_0^T |e(t)| \, dt$ | Integral Absolute Error. Total accumulated error magnitude. | A·s |
| **ISE** | $\int_0^T e(t)^2 \, dt$ | Integral Squared Error. Emphasizes large errors. | A²·s |

> **Why ITAE?** ITAE is preferred for control benchmarking because it penalizes both settling time and steady-state error. A controller that settles slowly will have a higher ITAE even if it eventually reaches zero error.

#### Point Error Metrics

| Metric | Formula | Description | Unit |
|--------|---------|-------------|------|
| **MAE** | $\frac{1}{N}\sum_{k=1}^{N} |e_k|$ | Mean Absolute Error | A |
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{k=1}^{N} e_k^2}$ | Root Mean Squared Error | A |
| **MaxError** | $\max_k |e_k|$ | Maximum absolute error | A |
| **SS_error** | $\bar{e}_{t>T_{settle}}$ | Steady-state error (mean after settling) | A |

**Literature Reference:**
> "Integral performance indices such as ITAE provide a single scalar measure that captures both transient and steady-state performance, making them suitable for optimization and comparison."
> — *Åström & Murray, Feedback Systems (2008)*

---

### 1.2 Dynamic Performance Metrics (DynamicsMetrics)

Step response characteristics following IEEE/ANSI definitions.

| Metric | Definition | Unit |
|--------|------------|------|
| **Rise Time (t_r)** | Time for signal to go from 10% to 90% of final value | s |
| **Settling Time (t_s)** | Time to remain within ±2% of final value | s |
| **Overshoot (M_p)** | $\frac{y_{max} - y_{final}}{y_{final}} \times 100\%$ | % |
| **Delay Time (t_d)** | Time to reach 50% of final value | s |
| **Peak Time (t_p)** | Time to reach maximum value | s |

```
     │       ↗ Overshoot (M_p)
     │      ╱  ╲
  y  │     ╱    ╲____ ±2% band
     │    ╱          ╲____
     │   ╱
     │──╱──────────────────
     │  ↑t_r   ↑t_p    ↑t_s
     └─────────────────────── t
```

**Literature Reference:**
> "For servo applications, rise time and settling time are critical metrics as they directly impact positioning accuracy and throughput."
> — *IEEE Transactions on Industrial Electronics (2021)*

---

### 1.3 Efficiency Metrics (EfficiencyMetrics)

Energy and power metrics adapted for current control.

#### Power Calculations

| Quantity | Formula | Description |
|----------|---------|-------------|
| **Copper Losses** | $P_{Cu} = \frac{3}{2} R_s (i_d^2 + i_q^2)$ | Resistive losses in windings |
| **Electrical Power** | $P_{elec} = \frac{3}{2} (u_d i_d + u_q i_q)$ | Input power from inverter |
| **Electromagnetic Torque** | $T_{em} = \frac{3}{2} p [\Psi_{PM} i_q + (L_d - L_q) i_d i_q]$ | Produced torque |
| **Mechanical Power** | $P_{mech} = T_{em} \cdot \omega_{mech}$ | Output power |
| **Efficiency** | $\eta = \frac{P_{mech}}{P_{elec}} \times 100\%$ | Overall efficiency |

> **Note:** Your simulation uses Current Control (CC), not Torque Control (TC). This means efficiency metrics evaluate "how efficiently does the controller achieve the *given* current references" rather than "does it choose optimal currents for a given torque."

---

### 1.4 Safety Metrics (SafetyMetrics)

Constraint violation tracking for hardware protection.

| Metric | Description | Criticality |
|--------|-------------|-------------|
| **Current Violations** | Count of $|I| > I_{max}$ | 🔴 Critical - thermal damage |
| **Voltage Violations** | Count of $|U| > U_{max}$ | 🔴 Critical - insulation breakdown |
| **dI/dt Violations** | Excessive current slew rate | 🟡 Warning - EMI issues |
| **Oscillation Detection** | Unstable control behavior | 🟡 Warning - controller tuning |

```python
# Current magnitude check
i_magnitude = sqrt(i_d² + i_q²)
violations = count(i_magnitude > I_max)  # I_max = 10.8 A for your motor
```

---

## 2. Neuromorphic Metrics (NeuromorphicMetrics)

These metrics are specific to Spiking Neural Network (SNN) controllers and quantify their computational advantages over conventional Artificial Neural Networks (ANNs).

### 2.1 Key Concepts

#### Synaptic Operations (SyOps)

Unlike ANNs where every neuron computes at every timestep, SNNs only perform computations when spikes occur. This leads to the concept of **Synaptic Operations (SyOps)**:

```
SyOps = Σ (spikes × fan-out)
      = Total number of multiply-accumulate operations triggered by spikes
```

**Comparison with MACs:**
- ANN: `MACs = neurons × weights × timesteps` (dense computation)
- SNN: `SyOps = active_spikes × fan-out` (sparse computation)

The **MAC Reduction Factor** = `MACs_dense / SyOps` indicates how much computation is saved.

#### Sparsity Types

| Sparsity Type | Definition | Typical Range |
|---------------|------------|---------------|
| **Activation Sparsity** | Fraction of neurons that never spike | 30-90% |
| **Temporal Sparsity** | Fraction of timesteps without spikes per neuron | 90-99% |
| **Connection Sparsity** | Fraction of zero weights | 0-95% |
| **Effective Sparsity** | Combined effect of all sparsity types | Varies |

### 2.2 Complete Neuromorphic Metrics Table

#### Computational Efficiency (Hardware-Independent)

| Metric | Description | Unit | Source |
|--------|-------------|------|--------|
| `total_syops` | Total synaptic operations | count | NeuroBench [1] |
| `syops_per_timestep` | Average SyOps per control step | count | NeuroBench [1] |
| `syops_per_second` | SyOps throughput | SyOps/s | NeuroBench [1] |
| `mac_reduction_factor` | Computation savings vs ANN | × | Computed |
| `activation_sparsity` | Fraction of silent neurons | 0-1 | NeuroBench [1] |
| `temporal_sparsity` | Fraction of silent timesteps | 0-1 | NeuroBench [1] |
| `connection_sparsity` | Fraction of zero weights | 0-1 | NeuroBench [1] |

#### Latency Metrics (Critical for Real-Time Control)

| Metric | Description | Unit | Target |
|--------|-------------|------|--------|
| `inference_latency_mean` | Average inference time | s | < 100 µs |
| `inference_latency_max` | Worst-case latency | s | < 1 ms |
| `inference_latency_p99` | 99th percentile | s | Deterministic |
| `time_to_first_spike` | First output spike time | s | Depends on encoding |

> **For Motor Control:** Latency must be less than the control loop period (typically 100 µs for current control at 10 kHz).

#### Energy Metrics (Platform-Dependent)

| Metric | Description | Unit | Reference Values |
|--------|-------------|------|-----------------|
| `energy_per_inference` | Energy per control step | J | 1-100 nJ |
| `energy_per_syop` | Energy per synaptic operation | J | 1-25 pJ |
| `dynamic_power` | Activity-dependent power | W | mW range |
| `static_power` | Leakage power | W | µW-mW |
| `energy_delay_product` | Efficiency metric (lower = better) | J·s | - |

**Platform-Specific Energy Values:**

| Platform | Energy per SyOp | Notes |
|----------|-----------------|-------|
| Intel Loihi 2 | ~23 pJ | Digital neuromorphic |
| SpiNNaker 2 | ~10 pJ | ARM-based |
| Analog mixed-signal | <1 pJ | Experimental |
| GPU (comparison) | ~1000 pJ | Not event-driven |

---

## 3. Literature References

### Core References

**[1] NeuroBench (2023)**
> Yik, J., et al. "NeuroBench: A Framework for Benchmarking Neuromorphic Computing Algorithms and Systems." *arXiv:2304.04640*

Key contributions:
- Standardized SyOps definition
- Sparsity metrics framework
- Benchmark datasets and tasks

**[2] IEEE Motor Control Metrics (2021)**
> "Metrics for Evaluating Quality of Control in Electric Drives." *IEEE Transactions on Industrial Electronics*

Key contributions:
- ITAE for current/torque tracking
- ME (Maximum Efficiency) deviation metrics
- Dynamic response characterization

**[3] Nature Electronics - Neuromorphic Efficiency (2022)**
> "Energy-Efficient Neuromorphic Computing: From Devices to Architectures." *Nature Electronics*

Key contributions:
- Energy-per-event analysis
- Comparison with conventional computing
- Event-driven advantages

**[4] SNN for Motor Control (2023)**
> "Spiking Neural Networks for Motor Control Applications." *Frontiers in Neuroscience*

Key contributions:
- Latency requirements for real-time control
- Temporal coding for continuous signals
- Hardware deployment considerations

### Additional References

**[5] Loihi 2 Characterization**
> Davies, M., et al. "Advancing Neuromorphic Computing with Loihi 2." *IEEE ISSCC 2022*

**[6] MTPA/MTPV Strategies**
> "Maximum Torque Per Ampere Control of PMSM Drives." *IEEE IECON*

---

## 4. Usage Examples

### Basic Control Benchmark

```python
from metrics import run_benchmark, BenchmarkResult
import pandas as pd

# Load simulation data
df = pd.read_csv('simulation.csv')
# Required columns: time, i_d, i_q, u_d, u_q, n, i_d_ref, i_q_ref

# Run benchmark
result = run_benchmark(
    df,
    controller_name="GEM Standard PI",
    operating_point="id=0, iq=5A @ 1500rpm"
)

# Print summary
print(result.summary())

# Export to dict/CSV
metrics_dict = result.to_dict()
```

### Neuromorphic Benchmark

```python
from metrics import compute_neuromorphic_metrics_from_spikes
import numpy as np

# Spike trains from SNN simulation
# Shape: (num_neurons, num_timesteps)
spike_trains = np.array([...])  # Binary matrix

# Weight matrix
weights = np.array([...])  # Shape: (post, pre)

# Compute metrics
neuro_metrics = compute_neuromorphic_metrics_from_spikes(
    spike_trains=spike_trains,
    weights=weights,
    dt_snn=1e-4,  # 100 µs timestep
    platform_energy_per_syop=23e-12,  # 23 pJ for Loihi 2
)

print(f"Total SyOps: {neuro_metrics.total_syops}")
print(f"MAC Reduction: {neuro_metrics.mac_reduction_factor:.1f}×")
print(f"Activation Sparsity: {neuro_metrics.activation_sparsity*100:.1f}%")
```

### Controller Comparison

```python
from metrics import compare_controllers

results = [
    run_benchmark(df_pi, "PI Controller"),
    run_benchmark(df_ann, "ANN Controller"),
    run_benchmark(df_snn, "SNN Controller"),
]

# Generate comparison table
comparison_df = compare_controllers(results, output_dir="./results")
```

---

## 5. Metric Selection Guide

### For Your Thesis: Recommended Primary Metrics

| Category | Primary Metric | Why |
|----------|---------------|-----|
| **Accuracy** | ITAE_iq | Single scalar capturing transient + steady-state |
| **Dynamics** | Rise Time, Settling Time | Standard control metrics |
| **Efficiency** | P_copper_mean | Directly measurable losses |
| **Safety** | current_violation_rate | Critical for hardware |
| **Neuromorphic** | syops_per_timestep, activation_sparsity | Core SNN advantages |
| **Neuromorphic** | energy_per_inference | Key selling point |

### Secondary Metrics (For Detailed Analysis)

- MAE, RMSE for error distribution
- Overshoot for stability assessment
- Latency for real-time feasibility
- MAC reduction factor for ANN comparison

---

## 6. Adapting Metrics for Your Setup

Since your simulation uses **Current Control (CC)** rather than **Torque Control (TC)**:

| Original Metric | Adaptation | Notes |
|-----------------|------------|-------|
| ITAE on τ_error | ITAE on i_q_error | Direct replacement |
| ME deviation | Not applicable* | Your controller doesn't choose i_d/i_q |
| Loss deviation | Compare to reference controller | Relative comparison |

*ME deviation measures "does the controller find optimal currents for given torque" - but your setup *receives* the current references directly.

---

## 7. Future Extensions

1. **Frequency Response Metrics**: Bandwidth, phase margin (requires sweep tests)
2. **Robustness Metrics**: Performance under parameter variation
3. **Disturbance Rejection**: Response to load torque steps
4. **Hardware-in-the-Loop**: Real Loihi/SpiNNaker measurements

---

*Last Updated: January 2026*
