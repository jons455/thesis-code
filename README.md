# EMBARK: Neuromorphic Motor Control Benchmark

**EMBARK** (Efficient Motor Benchmark for Adaptive Rate-encoding and Kinetic control) is a standardized benchmark suite for evaluating spiking neural network (SNN) controllers on permanent magnet synchronous motor (PMSM) current control tasks.

## 🎯 What is This?

A **plug-and-play benchmark** for rate-encoding SNN motor controllers that:
- Provides standardized evaluation scenarios
- Computes control performance metrics (tracking error, settling time, overshoot)
- Measures neuromorphic efficiency metrics (synaptic operations, activation sparsity)
- Enables fair comparison between different SNN architectures

## 📋 Interface Contract

### What You Provide

To benchmark your rate-encoding SNN controller, you need:

1. **Your trained PyTorch SNN model** (any rate-encoding architecture)
2. **Input feature configuration** (what state variables your model expects)
3. **Output mode** (absolute voltages or incremental deltas)

### What You Get

The benchmark returns:

| Category | Metrics |
|----------|---------|
| **Tracking Performance** | MAE, ITAE, Maximum Error |
| **Dynamics** | Settling Time, Overshoot |
| **Neuromorphic Efficiency** | Synaptic Operations, Activation Sparsity, Footprint |
| **Latency** | Mean/P95/P99 inference time |

Results are provided per-scenario with aggregate statistics across all test cases.

---

## 🚀 Quick Start Example

### Step 1: Install

```bash
pip install -e .
```

### Step 2: Prepare Your Model

```python
import torch
from embark.benchmark import (
    TensorControllerAdapter,
    SNNControllerWrapper,
    RateSNNStateProcessor,
    RateSNNActionProcessor,
)

# Load your trained SNN model
my_snn = torch.load("path/to/your_snn_model.pt")

# Wrap it (adds reset(), get_state(), set_state() methods)
wrapped_snn = SNNControllerWrapper(
    model=my_snn,
    device="cpu",
    track_spikes=True  # Enable for neuromorphic metrics
)
```

### Step 3: Configure Input Features

Tell the benchmark **what features your model expects** as input:

```python
state_proc = RateSNNStateProcessor(
    # Feature flags - enable what your model was trained on
    include_currents=True,      # i_d, i_q (2 features)
    include_errors=True,        # e_d, e_q (2 features)
    include_speed=True,         # n (1 feature)
    include_references=False,   # i_d_ref, i_q_ref (optional)
    include_prev_action=False,  # u_d_prev, u_q_prev (for incremental)
    include_derivatives=False,  # de_d, de_q, dn (temporal)
    include_ema_slow=False,     # Slow EMA filter (temporal)
    include_ema_fast=False,     # Fast EMA filter (temporal)
    
    # Normalization bounds (match your training data)
    error_gain=10.0,            # Error amplification factor
    n_max=4000.0,               # Max speed (RPM) for normalization
)

# Check that output dimension matches your model's input
print(f"State processor output dim: {state_proc.output_dim}")
# Should match your model's expected input size!
```

### Step 4: Configure Output Mode

```python
action_proc = RateSNNActionProcessor(
    incremental=False,          # False = absolute voltages, True = delta voltages
    delta_max=0.2,              # Only used if incremental=True
)
```

### Step 5: Create Controller

```python
controller = TensorControllerAdapter(
    controller=wrapped_snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
```

### Step 6: Run Benchmark Suite

```python
from embark.benchmark import BenchmarkSuite

# Create benchmark suite with standardized scenarios
suite = BenchmarkSuite()

# Run your controller through all scenarios
summary = suite.run(controller=controller, name="MySNN-v1")

# Print formatted results table
suite.print_summary(summary)

# Save results to JSON
suite.save_results(summary, "results/mysnn_v1_benchmark.json")
```

**Output Example:**

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    Benchmark Results: MySNN-v1                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Scenario                       │   MAE_i_q │  ITAE_i_q │  Settling │  OS% ║
║───────────────────────────────────────────────────────────────────────────
║ step_low_speed_500rpm_2A       │    0.038  │    1.12   │   0.018s  │  3.1%║
║ step_mid_speed_1500rpm_2A      │    0.035  │    0.98   │   0.015s  │  2.3%║
║ step_high_speed_2500rpm_2A     │    0.045  │    1.34   │   0.017s  │  3.8%║
║ multi_step_bidirectional       │    0.042  │    1.89   │   0.019s  │  4.2%║
║ four_quadrant_transition       │    0.048  │    2.01   │   0.021s  │  5.1%║
║ field_weakening_2500rpm        │    0.051  │    2.15   │   0.023s  │  4.8%║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Neuromorphic Efficiency:                                                 ║
║   Synaptic Operations (avg):  1.2M SyOps/inference                       ║
║   Activation Sparsity (avg):  67.3%                                      ║
║   Model Footprint:            43.2 KB                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 Standardized Scenarios

The benchmark suite includes **6 optimal scenarios** based on motor control benchmarking best practices, covering the full operating envelope:

| Scenario | Speed | Current | Duration | Tests |
|----------|-------|---------|----------|-------|
| **1. Low Speed Step** | 500 RPM | 0→2A i_q | 0.3s | Low-speed sensitivity, parameter robustness |
| **2. Mid Speed Step** ⭐ | 1500 RPM | 0→2A i_q | 0.3s | **Primary reference**: settling time, overshoot |
| **3. High Speed Step** | 2500 RPM | 0→2A i_q | 0.3s | Voltage limits, back-EMF, speed-dependent behavior |
| **4. Multi-Step Bidirectional** | 1500 RPM | ±2A i_q (4 steps) | 1.0s | Dynamic tracking, memory effects, consistency |
| **5. Four-Quadrant Transition** | 1500 RPM | +2→-2→0A i_q | 0.9s | Regenerative braking, torque reversal, zero-crossing |
| **6. Field-Weakening** | 2500 RPM | i_d=-2A, i_q=2A | 0.6s | d-q coupling, multivariable control, voltage saturation |

⭐ **Scenario 2** is the primary reference for all detailed comparisons (nominal operating conditions).

### Coverage Analysis

These 6 scenarios comprehensively test:

| Property | Covered By |
|----------|------------|
| Low-speed performance | Scenario 1 |
| Nominal performance | Scenario 2 ⭐ |
| High-speed performance | Scenarios 3, 6 |
| Transient response | Scenarios 1-3 |
| Dynamic tracking | Scenario 4 |
| Motoring & generating | Scenarios 4, 5 |
| Torque reversal | Scenario 5 |
| Zero-crossing | Scenario 5 |
| d-q decoupling | Scenario 6 |
| Voltage saturation | Scenarios 3, 6 |

For detailed information about scenario design, implementation, and interpretation, see **[BENCHMARK_SCENARIOS.md](docs/BENCHMARK_SCENARIOS.md)**.

---

## 🔧 Advanced: Single-Scenario Testing

For development and debugging, you can run individual scenarios:

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PMSMCurrentControlTask,
    TrackingMAE,
    TrackingITAE,
    SettlingTime,
    Overshoot,
)

# Create a single task
task = PMSMCurrentControlTask.from_config(
    n_rpm=1000,      # Motor speed (RPM)
    i_q_ref=2.0,     # Target q-axis current (A)
    i_d_ref=0.0,     # Target d-axis current (A, usually 0)
    max_steps=2000   # Simulation length
)

# Configure your controller with task
controller.configure(task.physics_engine.config, task)

# Define metrics
metrics = [
    TrackingMAE(tracked_keys=["i_q", "i_d"]),
    TrackingITAE(tracked_keys=["i_q", "i_d"]),
    SettlingTime(tracked_key="i_q", threshold=0.05),
    Overshoot(tracked_key="i_q"),
]

# Run single scenario
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()

print(f"Steps: {results['steps']}")
print(f"MAE i_q: {results['mae_i_q']:.4f} A")
print(f"Settling time: {results['settling_time']:.4f} s")
print(f"Overshoot: {results['overshoot']:.2f} %")

# Cleanup
task.physics_engine.close()
```

---

## 🧩 Feature Configuration Guide

The `RateSNNStateProcessor` supports various input feature combinations. Choose what your model was trained on:

### Basic Configuration (5 features)
```python
RateSNNStateProcessor(
    include_currents=True,   # i_d, i_q
    include_errors=True,     # e_d, e_q  
    include_speed=True,      # n
)
# Output: [i_d_norm, i_q_norm, e_d, e_q, n_norm] → 5 features
```

### With Derivatives (8 features)
```python
RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    include_derivatives=True,  # de_d, de_q, dn
)
# Output: 5 + 3 derivative features → 8 features
```

### With Temporal Filtering (9 features)
```python
RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    include_ema_slow=True,   # EMA(e_d, e_q) with α=0.98
    include_ema_fast=True,   # EMA(e_d, e_q) with α=0.70
)
# Output: 5 + 2 slow EMA + 2 fast EMA → 9 features
```

### Full Configuration (13 features)
```python
RateSNNStateProcessor(
    include_currents=True,      # i_d, i_q
    include_errors=True,        # e_d, e_q
    include_speed=True,         # n
    include_references=True,    # i_d_ref, i_q_ref
    include_prev_action=True,   # u_d_prev, u_q_prev
    include_derivatives=True,   # de_d, de_q, dn
    include_ema_slow=True,      # EMA_slow(e_d, e_q)
    include_ema_fast=True,      # EMA_fast(e_d, e_q)
)
# Output: 2 + 2 + 1 + 2 + 2 + 3 + 2 + 2 = 16 features
```

**⚠️ Critical:** The total features must match your model's input layer size!

---

## 📈 Output Mode Configuration

### Absolute Mode (Default)
```python
RateSNNActionProcessor(incremental=False)
```
- Model outputs voltages directly: `v_d`, `v_q` ∈ [-1, 1]
- Scaled to physical range: `[-u_max, u_max]`
- Use this if your model was trained with absolute voltage targets

### Incremental Mode
```python
RateSNNActionProcessor(incremental=True, delta_max=0.2)
```
- Model outputs voltage changes: `Δv_d`, `Δv_q` ∈ [-1, 1]
- Accumulated over time: `v(t+1) = clip(v(t) + Δv * delta_max, [-u_max, u_max])`
- Use this if your model was trained with incremental/delta outputs

---

## 🔬 Baseline Comparison

Compare your SNN against a classical PI controller baseline:

```python
from embark.benchmark import PIControllerAgent, BenchmarkSuite

# Your SNN controller
snn_controller = TensorControllerAdapter(...)

# Classical baseline
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
pi_controller = PIControllerAgent.from_system_config(task.physics_engine.config)

# Run both
suite = BenchmarkSuite()
snn_results = suite.run(controller=snn_controller, name="MySNN")
pi_results = suite.run(controller=pi_controller, name="PI-Baseline")

# Compare
suite.print_summary(snn_results)
suite.print_summary(pi_results)
```

---

## 📝 Complete Working Example

Here's a minimal complete example you can copy and modify:

```python
"""
Minimal example: Benchmark a rate-encoding SNN on PMSM current control.
"""
import torch
from embark.benchmark import (
    BenchmarkSuite,
    TensorControllerAdapter,
    SNNControllerWrapper,
    RateSNNStateProcessor,
    RateSNNActionProcessor,
)

# 1. Load your trained model
my_snn = torch.load("my_snn_model.pt")

# 2. Wrap it
wrapped_snn = SNNControllerWrapper(model=my_snn, track_spikes=True)

# 3. Configure state processor (match your training features!)
state_proc = RateSNNStateProcessor(
    include_currents=True,   # Your model expects these 5 features
    include_errors=True,
    include_speed=True,
    error_gain=10.0,
    n_max=4000.0,
)

# 4. Configure action processor
action_proc = RateSNNActionProcessor(incremental=False)

# 5. Create controller
controller = TensorControllerAdapter(
    controller=wrapped_snn,
    state_processor=state_proc,
    action_processor=action_proc,
)

# 6. Run benchmark
suite = BenchmarkSuite()
summary = suite.run(controller=controller, name="MySNN-v1")

# 7. View and save results
suite.print_summary(summary)
suite.save_results(summary, "results.json")
```

**That's it!** You now have standardized benchmark results for your SNN controller.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     BenchmarkSuite                          │
│  (Orchestrates multiple scenarios)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  ClosedLoopHarness    │  (Single scenario runner)
           └───────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐
│     Task     │ │Controller│ │  Metrics   │
│              │ │          │ │            │
│ ┌──────────┐ │ │          │ │ • MAE      │
│ │ Physics  │ │ │          │ │ • ITAE     │
│ │ Engine   │ │ │          │ │ • Settling │
│ └──────────┘ │ │          │ │ • SyOps    │
│ ┌──────────┐ │ │          │ │            │
│ │Reference │ │ │          │ │            │
│ │Generator │ │ │          │ │            │
│ └──────────┘ │ │          │ │            │
└──────────────┘ └──────────┘ └────────────┘
```

For rate-encoding SNNs, the **Controller** is a `TensorControllerAdapter` that wraps:
- Your SNN model (via `SNNControllerWrapper`)
- State processor (dict → tensor conversion)
- Action processor (tensor → dict conversion)

---

## 📚 Documentation

- **[BENCHMARK_API.md](docs/BENCHMARK_API.md)** - Complete API reference
- **[BENCHMARK_SCENARIOS.md](docs/BENCHMARK_SCENARIOS.md)** - Scenario design, coverage, and interpretation guide
- **[RATE_SNN_BENCHMARK_INTERFACE.md](docs/RATE_SNN_BENCHMARK_INTERFACE.md)** - Rate-SNN specific guide
- **[Examples](examples/)** - Full working examples (coming soon)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Follow the existing processor patterns
2. Add tests for new features
3. Update documentation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Contact

Questions? Open an issue or contact the maintainers.

---

## 🎯 Summary: What You Need to Know

| Your Responsibility | EMBARK Provides |
|---------------------|-----------------|
| Trained SNN model (PyTorch) | Standardized test scenarios |
| Input feature configuration | Physics simulation (PMSM) |
| Output mode (absolute/incremental) | Performance metrics (MAE, ITAE, etc.) |
| | Neuromorphic metrics (SyOps, sparsity) |
| | Fair comparison framework |
| | Baseline PI controller |

**The goal:** Plug in your SNN model, configure features, get standardized benchmark results.
