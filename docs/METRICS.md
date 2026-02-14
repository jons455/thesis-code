# Metrics Reference

Single source of truth for what the benchmark currently measures.

---

## Core Metrics (Implemented)

All accumulators follow the `MetricAccumulator` protocol: O(1) `update()` per step, `compute()` once at episode end.

### Tracking Accuracy

| Metric | Output Key | Unit | Location |
|--------|-----------|------|----------|
| **MAE** | `mae_i_q`, `mae_i_d` | A | `tracking.py` |
| **ITAE** | `itae_i_q`, `itae_i_d` | A·s² | `tracking.py` |
| **Max Error** | `max_error_i_q`, `max_error_i_d` | A | `tracking.py` |

- **MAE** (Mean Absolute Error): `sum(|ref - meas|) / count` — intuitive, robust tracking quality measure.
- **ITAE** (Integral Time Absolute Error): `sum(t * |error| * dt)` — penalizes sustained errors, standard in control benchmarks.
- **Max Error**: `max(|ref - meas|)` — worst-case safety metric; a controller with low average error but high max-error may have dangerous spikes.

### Dynamics

| Metric | Output Key | Unit | Location |
|--------|-----------|------|----------|
| **Settling Time** | `settling_time` | s | `dynamics.py` |
| **Overshoot** | `overshoot` | % | `dynamics.py` |

- **Settling Time**: Time until error stays within threshold (default 0.05 A). Returns `inf` if never settles. Tracked on `i_q`.
- **Overshoot**: `max(0, (max_value - final_ref) / |final_ref| * 100)`. Tracked on `i_q`.

### Inference Latency

| Metric | Output Key | Unit | Location |
|--------|-----------|------|----------|
| **Mean Latency** | `mean_latency_ms` | ms | `latency.py` |
| **P95 Latency** | `p95_latency_ms` | ms | `latency.py` |
| **P99 Latency** | `p99_latency_ms` | ms | `latency.py` |
| **Max Latency** | `max_latency_ms` | ms | `latency.py` |
| **Jitter** | `jitter_ms` | ms | `latency.py` |
| **Total Inference Time** | `total_inference_time_s` | s | `latency.py` |
| **Chip Mean** | `chip_mean_us` | µs | `latency.py` |
| **Chip Median** | `chip_median_us` | µs | `latency.py` |
| **Chip P95** | `chip_p95_us` | µs | `latency.py` |
| **Chip P99** | `chip_p99_us` | µs | `latency.py` |
| **Chip Max** | `chip_max_us` | µs | `latency.py` |
| **Chip Min** | `chip_min_us` | µs | `latency.py` |

- **Round-trip latency** (ms): Read from `controller_info["inference_latency_s"]`. Measures full observation-to-action round trip.
- **On-chip latency** (µs): Read from `controller_info["chip_inference_time_s"]`. Measures hardware accelerator inference only. Present when controller reports chip timing (e.g. Akida).
- **Purpose**: Validate real-time feasibility. For typical PMSM current control at 10 kHz (100 µs period), mean latency should be < 50 µs and max latency < 100 µs.

### Neuromorphic / NeuroBench

When the controller exposes a `torch.nn.Module` as `.model`, the metric factory (`create_metrics()`) automatically adds NeuroBench adapters from `embark/benchmark/contrib/neurobench/metric_adapters.py`.

#### Static Metrics (model-only, computed once at episode end)

| Metric | Output Keys | Description |
|--------|------------|-------------|
| **Footprint** | `footprint`, `nb_footprint` | Model memory footprint |
| **Connection Sparsity** | `connection_sparsity`, `nb_connection_sparsity` | Fraction of zero-weight connections [0-1] |

#### Workload Metrics (updated per step)

| Metric | Output Keys | Description |
|--------|------------|-------------|
| **Synaptic Operations** | `total_syops`, `syops_per_step`, `effective_macs`, `effective_acs`, `dense`, `nb_synaptic_operations_total` | Computational cost proxy (energy indicator) |
| **Activation Sparsity** | `activation_sparsity`, `nb_activation_sparsity` | Fraction of non-spiking neurons [0-1] |

- **SyOps**: `effective_macs + effective_acs`; falls back to `dense` when both are zero. `syops_per_step = total_syops / steps`.
- **Activation Sparsity**: Higher is better — indicates more efficient spike-based computation.
- Additional NeuroBench metrics are discovered dynamically via `discover_neurobench_metric_classes()` and emitted with `nb_*` prefixed keys.

Controllers without a `.model` (e.g. PI) get only control + latency metrics; no neuromorphic metrics.

---

## Metric Factory

Location: `embark/benchmark/metrics/neurobench_factory.py`

`create_metrics(controller)` returns:

**Always included (control metrics):**
- `TrackingMAE(tracked_keys=["i_q", "i_d"])`
- `TrackingITAE(tracked_keys=["i_q", "i_d"])`
- `MaximumError(tracked_keys=["i_q", "i_d"])`
- `SettlingTime(tracked_key="i_q", threshold=0.05)`
- `Overshoot(tracked_key="i_q")`

**Conditionally included** (when `controller.model` is a `torch.nn.Module`):
- All NeuroBench static metric adapters (Footprint, ConnectionSparsity, ...)
- All NeuroBench workload metric adapters (SynapticOperations, ActivationSparsity, ...)

Note: `InferenceLatency` is always safe to add — it defaults to zero when the controller does not report timing.

---

## Benchmark Scenarios

Location: `embark/benchmark/harness/benchmark_suite.py`

### Standard Scenarios (6)

| Scenario | Speed | Reference | Purpose |
|----------|-------|-----------|---------|
| `step_low_load` | 1000 RPM | Step i_q=1A | Low torque response |
| `step_mid_load` | 1000 RPM | Step i_q=5A | Medium torque response |
| `step_high_load` | 1000 RPM | Step i_q=9A | Near-limit torque |
| `step_high_speed` | 2500 RPM | Step i_q=5A | High back-EMF operation |
| `sinusoidal_tracking` | 1000 RPM | sin(10Hz, 2A, offset 3A) | Dynamic tracking |
| `flux_weakening` | 2500 RPM | i_d=-3A, i_q=3A | Field-weakening region |

### Quick Scenarios (3)

Fast validation subset: `step_low_load`, `step_mid_load`, `sinusoidal_tracking`.

---

## Output Format

Three output levels: single-run (harness), multi-scenario (suite), and NeuroBench-style export.

### 1. Single run: `ClosedLoopHarness.run()`

Returns a **flat dict** with `steps` plus every key produced by each metric's `compute()`.

**Always present:**
- `steps`: `int` — number of control steps executed.

**Keys from default metrics** (only present if that metric is in the harness):

| Key | Type | Source |
|-----|------|--------|
| `mae_i_q`, `mae_i_d` | float | TrackingMAE |
| `itae_i_q`, `itae_i_d` | float | TrackingITAE |
| `max_error_i_q`, `max_error_i_d` | float | MaximumError |
| `settling_time` | float | SettlingTime (s or `inf`) |
| `overshoot` | float | Overshoot (%) |
| `total_syops`, `syops_per_step` | float | NeuroBench SynapticOperations adapter |
| `effective_macs`, `effective_acs`, `dense` | float | NeuroBench SynapticOperations adapter |
| `activation_sparsity`, `footprint`, `connection_sparsity` | float | NeuroBench adapters |
| `nb_*` | float | NeuroBench adapters (all raw keys) |
| `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`, `jitter_ms`, `total_inference_time_s` | float | InferenceLatency |
| `chip_mean_us`, `chip_median_us`, `chip_p95_us`, `chip_p99_us`, `chip_max_us`, `chip_min_us` | float | InferenceLatency (when chip timing present) |

**Example (excerpt):**

```python
{
  "steps": 2000,
  "mae_i_q": 0.012,
  "mae_i_d": 0.001,
  "max_error_i_q": 0.95,
  "settling_time": 0.05,
  "overshoot": 4.2,
  "total_syops": 12000.0,
  "syops_per_step": 6.0,
  "activation_sparsity": 0.85,
  "mean_latency_ms": 0.032,
  "max_latency_ms": 0.048,
}
```

### 2. Multi-scenario: `BenchmarkSummary.to_dict()`

Returned by `BenchmarkSuite.run(controller, name)` and serialized via `BenchmarkSuite.save_results(summary, path)` (JSON).

**Shape:**

```python
{
  "controller_name": str,
  "aggregate": {
    "mean_mae_iq": float,       # average MAE i_q across all scenarios
    "worst_max_error_iq": float, # worst-case max error across scenarios
    "num_safety_violations": int,
    "num_scenarios": int,
  },
  "scenarios": [
    {
      "name": str,
      "description": str,
      "metrics": { ... },        # same flat dict as harness.run()
      "safety_terminated": bool,
      "violation_reason": str | None,
    },
    ...
  ],
}
```

### 3. NeuroBench-style export: `ClosedLoopMetricExporter.to_neurobench_format()`

Takes the flat dict from one `harness.run()` and splits it into control vs workload for comparison tables.

**Shape:**

```python
{
  "benchmark": str,
  "model": str,
  "steps": int,
  "control_metrics": {
    # mae_i_q, mae_i_d, itae_i_q, itae_i_d,
    # max_error_i_q, max_error_i_d, settling_time, overshoot
  },
  "workload_metrics": {
    # total_syops, syops_per_step, and any key starting with "nb_"
  },
}
```

---

## NeuroBench Alignment

NeuroBench is built for **single-shot inference** (dataset -> model -> metrics via hooks). Our benchmark is **closed-loop** (controller <-> physics loop). We use **adapters** in `embark/benchmark/contrib/neurobench/metric_adapters.py` to bridge the gap:

| NeuroBench metric (concept) | Our usage |
|-----------------------------|-----------|
| **Footprint, ConnectionSparsity** | Static adapters; called in `compute()` with `controller.model`. |
| **ActivationSparsity, SynapticOperations** | Workload adapters; updated each step with `last_observation` / `last_action_tensor`, final result in `compute()`. |
| **Latency** | We use our own `InferenceLatency` accumulator (closed-loop round-trip and chip timing); not replaced by NeuroBench. |
| **MAE, ITAE, Max Error, Settling Time, Overshoot** | Control-only; implemented in `tracking.py` and `dynamics.py`. Not provided by NeuroBench. |

The controller-aware metric factory (`create_metrics(controller)`) adds NeuroBench adapters only when the controller has a `torch.nn.Module` as `.model`.

---

## CLI Usage

```bash
# Full 6-scenario suite
poetry run python -m evaluation.core.run_evaluation

# Quick 3-scenario check
poetry run python -m evaluation.core.run_evaluation --quick
```

---

## Removed / Deprecated

- **RMSE**: Removed; use MAE and ITAE for tracking.
- **Efficiency metrics**: ControlEffort, TotalVariation, EnergyConsumption removed.
- **LAC (Logarithmic Accuracy-Cost)**: Removed; no composite score in the pipeline.
- **`step_low_speed` scenario**: Removed (200 RPM too slow for practical PMSM benchmarking).
