# Metrics Reference

Single source of truth for what the benchmark currently measures.

---

## Default Metric Set (`create_metrics()`)

All accumulators follow the `MetricAccumulator` protocol: O(1) `update()` per step, `compute()` once at episode end.

The factory (`embark/benchmark/metrics/neurobench_factory.py`) always returns:

| Metric | Class | Output Key(s) | Unit |
|--------|-------|--------------|------|
| **MAE** | `TrackingMAE` | `mae_i_q`, `mae_i_d` | A |
| **ITAE** | `TrackingITAE` | `itae_i_q`, `itae_i_d` | A·s² |
| **Max Error** | `MaximumError` | `max_error_i_q`, `max_error_i_d` | A |
| **Settling Time** | `SettlingTime` | `settling_time_i_q` | s |
| **Overshoot** | `Overshoot` | `overshoot` | % |
| **Steady-State RMS** | `SteadyStateRMS` | `rms_i_q`, `rms_i_d` | A |
| **Inference Latency** | `InferenceLatency` | see below | ms / µs |

When the controller exposes `.model` (a `torch.nn.Module`), NeuroBench adapters are added automatically.

---

## Tracking Accuracy

### `TrackingMAE` — Mean Absolute Error (full episode)

**Formula:**

$$\text{MAE}_q = \frac{1}{N} \sum_{k=0}^{N} |i_{q,\text{ref}}[k] - i_q[k]|$$

- Computed over the **full episode** (transient + steady-state)
- Industry-standard metric for motor control validation; intuitive and robust
- Complements ITAE (transient focus) and RMS (steady-state focus)

**Output keys:** `mae_i_q`, `mae_i_d`
**Units:** A
**Location:** `metrics/accumulators/tracking.py`

---

### `TrackingITAE` — Integral Time Absolute Error (transient window)

**Formula:**

$$\text{ITAE}_q = \int_0^{50\text{ ms}} t \cdot |i_{q,\text{ref}}(t) - i_q(t)| \, dt$$

and likewise for $i_d$.

- Integration window: **first 50 ms** of each episode only (`window_s=0.05`)
- Time-weighting penalizes *sustained* transient errors more than brief spikes
- Standard in control benchmarking; closely tracks thermal stress buildup

**Output keys:** `itae_i_q`, `itae_i_d`
**Units:** A·s²
**Location:** `metrics/accumulators/tracking.py`

---

### `SteadyStateRMS` — RMS of steady-state error

**Formula:**

$$\text{RMS}(i_q) = \sqrt{\frac{1}{N} \sum_{k=T_{ss}}^{T} \left(i_{q,\text{ref}}[k] - i_q[k]\right)^2}$$

where $T_{ss}$ is the first step after `transient_s` (default 50 ms).

- Excludes the transient (first 50 ms) — captures only the steady-state region
- Measures torque ripple and residual bias after settling

**Output keys:** `rms_i_q`, `rms_i_d`
**Units:** A
**Location:** `metrics/accumulators/tracking.py`

---

### `MaximumError` — Worst-case absolute error

**Formula:** $e_{\max} = \max_k |i_{q,\text{ref}}[k] - i_q[k]|$ over the full episode.

- Safety metric: a controller with low average error but a large spike may still be dangerous

**Output keys:** `max_error_i_q`, `max_error_i_d`
**Units:** A
**Location:** `metrics/accumulators/tracking.py`

---

## Dynamics

### `SettlingTime` — Time to enter and remain within 2% band

**Threshold:** 2% of step size (`band_fraction=0.02`). For a 2 A step: ±0.04 A.

**Dwell requirement:** The signal must stay inside the band continuously for at least **1 ms** (`dwell_s=0.001`). This prevents a zero-crossing from being falsely counted as settling.

**Algorithm:**
1. Detect first non-zero reference as step target; compute `band = 0.02 × |step_target|`
2. Track the start of each in-band run (`_candidate_entry`)
3. If the signal leaves the band, reset the dwell clock
4. Once the signal has been in-band for ≥ 1 ms, record `settling_time = _candidate_entry`
5. Returns `inf` if never settled

**Output key:** `settling_time_i_q`
**Units:** seconds (s), or `inf`
**Location:** `metrics/accumulators/dynamics.py`

---

### `Overshoot` — Peak overshoot relative to step

**Formula:**

$$\text{overshoot} (\%) = \max\!\left(0,\ \frac{\text{peak} - i_{q,\text{ref}}}{|i_{q,\text{ref}}|} \times 100\right)$$

- `step_ref` is latched at the first non-zero reference value
- Returns 0.0 if no step fires or no overshoot occurs

**Output key:** `overshoot`
**Units:** %
**Location:** `metrics/accumulators/dynamics.py`

---

## Inference Latency

### `InferenceLatency`

Reads timing data from `controller_info` dict (populated by hardware-in-the-loop or profiling wrappers). Safe to use with any controller — defaults to 0.0 when no timing data is present.

#### Round-trip latency (milliseconds)

| Output Key | Description |
|-----------|-------------|
| `mean_latency_ms` | Mean round-trip time |
| `p95_latency_ms` | 95th percentile |
| `p99_latency_ms` | 99th percentile |
| `max_latency_ms` | Maximum |
| `jitter_ms` | Standard deviation |
| `total_inference_time_s` | Sum of all latencies |

#### On-chip latency (microseconds, when chip timing is available)

| Output Key | Description |
|-----------|-------------|
| `chip_mean_us` | Mean on-chip time |
| `chip_median_us` | Median |
| `chip_p95_us` | 95th percentile |
| `chip_p99_us` | 99th percentile |
| `chip_max_us` | Maximum |
| `chip_min_us` | Minimum |

**Real-time constraint:** At 10 kHz (100 µs period), mean latency should be < 50 µs, max < 100 µs.

**Location:** `metrics/accumulators/latency.py`

---

## NeuroBench Workload & Static Metrics

Added automatically when `controller.model` is a `torch.nn.Module`.

### Static (computed once at episode end)

| Metric | Output Keys | Description |
|--------|------------|-------------|
| **Footprint** | `footprint`, `nb_footprint` | Model memory footprint |
| **Connection Sparsity** | `connection_sparsity`, `nb_connection_sparsity` | Fraction of zero-weight connections [0–1] |

### Workload (updated per step)

| Metric | Output Keys | Description |
|--------|------------|-------------|
| **Synaptic Operations** | `total_syops`, `syops_per_step`, `effective_macs`, `effective_acs`, `dense` | Computational cost proxy (energy indicator) |
| **Activation Sparsity** | `activation_sparsity`, `nb_activation_sparsity` | Fraction of inactive neurons [0–1]; higher = better |

- **SyOps:** `effective_macs + effective_acs`; falls back to `dense` when both are zero. `syops_per_step = total_syops / steps`.
- Any additional NeuroBench metrics discovered via `discover_neurobench_metric_classes()` are emitted with `nb_*` prefixed keys.

**Location:** `embark/benchmark/contrib/neurobench/metric_adapters.py`

---

## Metric Factory

**Location:** `embark/benchmark/metrics/neurobench_factory.py`

```python
from embark.benchmark.metrics import create_metrics

metrics = create_metrics(controller)  # controller=None → control metrics only
```

**Always included:**
```python
TrackingMAE(tracked_keys=["i_q", "i_d"])
TrackingITAE(tracked_keys=["i_q", "i_d"], window_s=0.05)
MaximumError(tracked_keys=["i_q", "i_d"])
SettlingTime(tracked_key="i_q", band_fraction=0.02, dwell_s=0.001)
Overshoot(tracked_key="i_q")
SteadyStateRMS(tracked_keys=["i_q", "i_d"], transient_s=0.05)
InferenceLatency()
```

**Conditionally added** (when `controller.model` is a `torch.nn.Module`):
- NeuroBench static metric adapters (Footprint, ConnectionSparsity)
- NeuroBench workload metric adapters (SynapticOperations, ActivationSparsity)

---

## Output Format

### 1. Single run: `ClosedLoopHarness.run()`

Returns a flat dict.

**Always present:**

| Key | Type | Source |
|-----|------|--------|
| `steps` | int | Harness |
| `mae_i_q`, `mae_i_d` | float | TrackingMAE |
| `itae_i_q`, `itae_i_d` | float | TrackingITAE |
| `max_error_i_q`, `max_error_i_d` | float | MaximumError |
| `settling_time_i_q` | float | SettlingTime (s or `inf`) |
| `overshoot` | float | Overshoot (%) |
| `rms_i_q`, `rms_i_d` | float | SteadyStateRMS |
| `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`, `jitter_ms`, `total_inference_time_s` | float | InferenceLatency |
| `chip_mean_us`, `chip_median_us`, `chip_p95_us`, `chip_p99_us`, `chip_max_us`, `chip_min_us` | float | InferenceLatency |

**Conditionally present** (when `controller.model` is a `torch.nn.Module`):

| Key | Source |
|-----|--------|
| `total_syops`, `syops_per_step`, `effective_macs`, `effective_acs` | NeuroBench SynapticOperations |
| `activation_sparsity` | NeuroBench ActivationSparsity |
| `footprint`, `connection_sparsity` | NeuroBench static adapters |
| `nb_*` | All raw NeuroBench keys |

### 2. Multi-scenario: `BenchmarkSummary.to_dict()`

```python
{
  "controller_name": str,
  "aggregate": {
    "worst_max_error_iq": float,    # worst-case max error across scenarios
    "num_safety_violations": int,
    "num_scenarios": int,
  },
  "scenarios": [
    {
      "name": str,
      "description": str,
      "metrics": { ... },           # flat dict from harness.run()
      "safety_terminated": bool,
      "violation_reason": str | None,
    },
    ...
  ],
}
```

---

## NeuroBench Alignment

| NeuroBench concept | Our usage |
|--------------------|-----------|
| **Footprint, ConnectionSparsity** | Static adapters; evaluated in `compute()` on `controller.model` |
| **ActivationSparsity, SynapticOperations** | Workload adapters; updated each step via `last_observation` / `last_action_tensor` |
| **Latency** | Custom `InferenceLatency` accumulator; reads from `controller_info` (round-trip + chip timing) |
| **ITAE, MaxError, SettlingTime, Overshoot, RMS** | Control-only; implemented in `tracking.py` and `dynamics.py` |

---

## Removed / Deprecated

- **Mean MAE across scenarios** (`BenchmarkSummary.mean_mae_iq`): Removed — aggregating MAE over scenarios with different episode lengths and step sizes produces a meaningless number. Per-scenario `mae_i_q` values in `scenario_results` are used for comparison instead.
- **RMSE** (old): Replaced by `SteadyStateRMS`.
- **ControlEffort, TotalVariation, EnergyConsumption**: Removed.
- **LAC (Logarithmic Accuracy-Cost)**: Removed; no composite score in the pipeline.
- **`BenchmarkSummary.mean_mae_iq`**: Removed. Per-scenario `rms_i_q` values in `scenario_results` are used instead.
