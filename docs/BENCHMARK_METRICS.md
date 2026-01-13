# Benchmark Metrics for Neuromorphic PMSM Control

**Document Purpose**: Meeting preparation - Overview of metrics framework for neuromorphic motor control benchmark.

**Date**: 2026-01-13
**Author**: Jonas
**Status**: WP2 Complete, metrics framework ready

---

## Executive Summary

This benchmark evaluates **Spiking Neural Network (SNN) controllers** for PMSM current control against conventional PI controllers. We combine:

1. **Control Engineering Metrics** - How well does it control the motor?
2. **Neuromorphic Computing Metrics** - How efficiently does the SNN compute?

The goal is to answer: *Can SNNs match PI control quality while being more computationally efficient?*

---

## 1. Control Performance Metrics

These metrics evaluate **Regelgüte** (control quality) - standard in control engineering.

### 1.1 Accuracy Metrics (Tracking Error)

| Metric | Formula | Unit | What it measures |
|--------|---------|------|------------------|
| **ITAE** | ∫ t·\|e(t)\| dt | A·s² | Penalizes errors that persist over time |
| **IAE** | ∫ \|e(t)\| dt | A·s | Total accumulated error |
| **ISE** | ∫ e(t)² dt | A²·s | Penalizes large errors heavily |
| **MAE** | (1/N) Σ\|eₖ\| | A | Average absolute error |
| **RMSE** | √((1/N) Σeₖ²) | A | Root mean squared error |
| **Steady-State Error** | mean(e) for t > t_settle | A | Final tracking accuracy |

**Why ITAE?** It's the standard for controller benchmarking because it penalizes both slow settling AND persistent errors.

### 1.2 Dynamic Response Metrics

| Metric | Definition | Unit | Target |
|--------|------------|------|--------|
| **Rise Time (tᵣ)** | Time from 10% to 90% of final value | ms | < 5 ms |
| **Settling Time (tₛ)** | Time to stay within ±2% of reference | ms | < 10 ms |
| **Overshoot (Mₚ)** | Peak value above reference | % | < 10% |
| **Peak Time** | Time to reach peak | ms | - |

### 1.3 Efficiency Metrics (Electrical)

| Metric | Formula | Unit | What it measures |
|--------|---------|------|------------------|
| **Copper Losses** | I²R integrated | W or J | Energy lost in windings |
| **Electrical Power** | u·i integrated | W | Total power consumed |
| **Efficiency** | P_mech / P_elec | % | Motor efficiency |

### 1.4 Safety Metrics

| Metric | What it measures |
|--------|------------------|
| **Current Violations** | Times \|I\| > I_max |
| **Voltage Violations** | Times \|U\| > U_max |
| **di/dt Violations** | Excessive current rate of change |
| **Instability Index** | Oscillation detection |

---

## 2. Neuromorphic Computing Metrics

These metrics quantify the **computational efficiency** of SNNs - following the NeuroBench standard [Yik et al., 2024].

### 2.1 Synaptic Operations (SyOps)

**The key metric for neuromorphic efficiency.**

```
SyOps = Σ (spikes × fan-out connections)
```

| Metric | Unit | Description |
|--------|------|-------------|
| **Total SyOps** | count | Total spike-triggered accumulations |
| **SyOps/timestep** | count/step | Average per control cycle |
| **SyOps/second** | SyOps/s | Rate of operations |

**Why SyOps matter:**
- In ANNs: Every neuron computes every cycle → O(n²) operations
- In SNNs: Only neurons that spike trigger computation → O(k·n) where k << n
- Lower SyOps = Lower energy on neuromorphic hardware

### 2.2 Sparsity Metrics

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| **Activation Sparsity** | 1 - (active neurons / total) | 0-1 | Higher |
| **Temporal Sparsity** | 1 - (spikes / possible spikes) | 0-1 | Higher |
| **Connection Sparsity** | 1 - (nonzero weights / total) | 0-1 | Higher |

**Typical values:**
- ANNs: 0-40% activation sparsity
- SNNs: 80-99% activation sparsity (event-driven advantage!)

### 2.3 Latency Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **Inference Latency** | µs | Time to compute one control action |
| **Max Latency** | µs | Worst-case for real-time guarantees |
| **Jitter** | µs | Variance in latency |

**Constraint**: Control loop runs at 10 kHz → Max 100 µs per inference!

### 2.4 Energy Estimation

| Metric | Formula | Unit |
|--------|---------|------|
| **Dynamic Energy** | SyOps × E_per_SyOp | pJ |
| **Static Energy** | P_static × time | pJ |
| **Total Energy** | Dynamic + Static | pJ/inference |

**Platform-specific energy per SyOp:**

| Platform | Energy/SyOp | Source |
|----------|-------------|--------|
| Loihi 2 | ~23 pJ | Intel 2021 |
| SpiNNaker 2 | ~10 pJ | Mayr 2019 |
| GPU (for comparison) | ~1000 pJ | - |

---

## 3. How We Run the Benchmark

### 3.1 Benchmark Scenarios

| Scenario | Description | What it tests |
|----------|-------------|---------------|
| **Step Response** | Sudden change in reference | Rise time, overshoot |
| **Operating Point Sweep** | Various (id, iq) combinations | Robustness across envelope |
| **Disturbance Rejection** | Load torque step | Stability, recovery |

### 3.2 Code Structure

```
pmsm-pem/
├── benchmark/                    # NeuroBench integration
│   ├── pmsm_env.py              # Gymnasium wrapper for GEM
│   ├── agents.py                # PI baseline, future SNN
│   └── run_benchmark.py         # Main benchmark script
├── metrics/                      # Our custom metrics
│   ├── benchmark_metrics.py     # ~1100 lines of metrics
│   └── METRICS_DOCUMENTATION.md # Detailed formulas
```

### 3.3 Running a Benchmark

```python
from benchmark import PMSMEnv, PIControllerAgent
from neurobench.benchmarks import BenchmarkClosedLoop

# 1. Create environment
env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=2.0)

# 2. Create controller (PI baseline or SNN)
agent = PIControllerAgent()  # or: SNNControllerAgent(trained_model)

# 3. Run benchmark
benchmark = BenchmarkClosedLoop(
    agent=agent,
    environment=env,
    metric_list=[[Footprint, ConnectionSparsity],
                 [ActivationSparsity, SynapticOperations]]
)
results = benchmark.run(nr_interactions=50, max_length=500)

# 4. Also compute control metrics
from metrics import run_benchmark
control_results = run_benchmark(episode_data, controller_name="SNN")
```

### 3.4 Expected Output

```
Controller: SNN-LIF-2Layer
Operating Point: id=0A, iq=2A @ 1000rpm

Control Metrics:
  ITAE: 0.0023 A·s²
  Rise Time: 3.2 ms
  Settling Time: 8.1 ms
  Overshoot: 5.3%
  Steady-State Error: 0.01 A

Neuromorphic Metrics:
  SyOps/step: 1,240
  Activation Sparsity: 92.3%
  Inference Latency: 45 µs
  Estimated Energy: 28.5 nJ/step

Comparison to PI Baseline:
  Control Quality: 98.2% of PI performance
  Computational Cost: 15x fewer operations
```

---

## 4. Key Questions for Discussion

1. **Which control metrics are most important?**
   - ITAE vs. settling time vs. overshoot trade-offs?

2. **How to weight control quality vs. efficiency?**
   - Pareto front? Weighted score?

3. **What sparsity levels are realistic?**
   - Literature suggests 80-95% for motor control

4. **Latency constraints?**
   - 100 µs hard limit for 10 kHz control

---

## 5. References

1. **NeuroBench**: Yik et al., "NeuroBench: A Framework for Benchmarking Neuromorphic Computing", arXiv 2024
2. **gym-electric-motor**: Balakrishnan et al., JOSS 2021
3. **Symmetrical Optimum**: Kessler, 1958 (PI tuning method)
4. **Loihi 2 Energy**: Davies et al., Intel 2021

---

## Appendix: Metric Implementation Status

| Category | Metric | Implemented | Tested |
|----------|--------|-------------|--------|
| Accuracy | ITAE, IAE, ISE | ✅ | ✅ |
| Accuracy | MAE, RMSE | ✅ | ✅ |
| Dynamics | Rise/Settling Time | ✅ | ✅ |
| Dynamics | Overshoot | ✅ | ✅ |
| Efficiency | Copper Losses | ✅ | ✅ |
| Safety | Limit Violations | ✅ | ✅ |
| Neuromorphic | SyOps | ✅ | Pending SNN |
| Neuromorphic | Sparsity | ✅ | Pending SNN |
| Neuromorphic | Energy Est. | ✅ | Pending SNN |
