# Benchmark Quick Reference

Cheat sheet for common operations in `embark.benchmark`.

## Import Patterns

```python
# Core harness
from embark.benchmark import ClosedLoopHarness, BenchmarkSuite

# Tasks
from embark.benchmark import PMSMCurrentControlTask, SafetyLimits

# Controllers
from embark.benchmark import PIControllerAgent
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark import TensorControllerAdapter

# Processors
from embark.benchmark.processors import RateSNNStateProcessor, RateSNNActionProcessor

# Metrics
from embark.benchmark import TrackingMAE, TrackingITAE, SettlingTime, Overshoot
from embark.benchmark.metrics import create_metrics

# Physics
from embark.benchmark import PMSMPhysicsEngine, PMSMConfig

# Reference generators
from embark.benchmark.tasks.reference_generators import (
    StepReference,
    SinusoidalReference,
    ConstantReference,
)
```

## Common Patterns

### Pattern 1: PI Controller Single Run

```python
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
controller = PIControllerAgent.from_system_config(task.physics_engine.config)
harness = ClosedLoopHarness(task=task, controller=controller)
results = harness.run()
task.physics_engine.close()
```

### Pattern 2: SNN Controller Single Run

```python
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
snn = SNNControllerAgent("model.pt", track_spikes=True)
state_proc = RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    i_max=20.0,
)
action_proc = RateSNNActionProcessor(incremental=False, u_max=48.0)
controller = TensorControllerAdapter(controller=snn, state_processor=state_proc, action_processor=action_proc)
controller.configure(task.physics_engine.config, task)
harness = ClosedLoopHarness(task=task, controller=controller)
results = harness.run()
task.physics_engine.close()
```

### Pattern 3: Multi-Scenario Suite

```python
suite = BenchmarkSuite()
summary = suite.run(controller=controller, name="MyController")
suite.print_summary(summary)
suite.save_results(summary, "results.json")
```

### Pattern 4: Custom Metrics

```python
metrics = [
    TrackingMAE(tracked_keys=["i_q", "i_d"]),
    TrackingITAE(tracked_keys=["i_q"]),
    MaximumError(tracked_keys=["i_q"]),
    SettlingTime(tracked_key="i_q", threshold=0.05),
    Overshoot(tracked_key="i_q"),
]
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
```

### Pattern 5: Auto Metrics (with NeuroBench)

```python
metrics = create_metrics(controller)  # Automatically includes NeuroBench if controller.model exists
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
```

## Standard Scenarios

```python
from embark.benchmark.harness import STANDARD_SCENARIOS, QUICK_SCENARIOS

# 6 scenarios: step_low_load, step_mid_load, step_high_load, 
#              step_high_speed, sinusoidal_tracking, flux_weakening
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)

# 3 scenarios: step_low_load, step_mid_load, sinusoidal_tracking
suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
```

## Processor Configurations

### MinMax Normalization

```python
MinMaxProcessor(
    input_keys=["i_d", "i_q"],
    reference_keys=["i_d_ref", "i_q_ref"],
    feature_range=(-1.0, 1.0),  # Default
)
```

### Standard Scaler

```python
StandardScalerProcessor(
    input_keys=["i_d", "i_q"],
    reference_keys=["i_d_ref", "i_q_ref"],
    mean={"i_d": 0.0, "i_q": 0.0},  # Optional
    std={"i_d": 1.0, "i_q": 1.0},   # Optional
)
```

### Linear Action Scaling

```python
LinearActionProcessor(
    output_keys=["v_d", "v_q"],
    bounds={"v_d": (-48.0, 48.0), "v_q": (-48.0, 48.0)},
    scale=1.0,  # Optional multiplier
)
```

### PWM Action (with dead-time)

```python
PWMActionProcessor(
    output_keys=["v_d", "v_q"],
    bounds={"v_d": (-48.0, 48.0), "v_q": (-48.0, 48.0)},
    dead_time_s=2e-6,  # Dead-time compensation
)
```

## Safety Limits

```python
SafetyLimits(
    max_current_a=20.0,    # Max |i_d| or |i_q| [A]
    max_voltage_v=60.0,    # Max |v_d| or |v_q| [V]
    max_speed_rpm=3000.0,  # Max omega [RPM]
)
```

## Reference Generators

### Step Reference

```python
StepReference(
    i_d_ref=0.0,
    i_q_ref=2.0,
    step_time=0.0,  # When step occurs [s]
)
```

### Sinusoidal Reference

```python
SinusoidalReference(
    i_d_ref=0.0,
    i_q_amp=2.0,        # Amplitude [A]
    i_q_offset=3.0,     # DC offset [A]
    frequency_hz=10.0,  # Frequency [Hz]
    phase=0.0,          # Phase [rad]
)
```

### Constant Reference

```python
ConstantReference(
    i_d_ref=0.0,
    i_q_ref=2.0,
)
```

## Result Keys

### Control Metrics

- `steps`: Episode length
- `mae_i_q`, `mae_i_d`: Mean Absolute Error [A]
- `itae_i_q`, `itae_i_d`: Integral Time Absolute Error [A·s²]
- `max_error_i_q`, `max_error_i_d`: Maximum error [A]
- `settling_time`: Settling time [s] (or `inf`)
- `overshoot`: Overshoot [%]

### Latency Metrics

- `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`
- `jitter_ms`, `total_inference_time_s`
- `chip_mean_us`, `chip_median_us`, `chip_p95_us`, `chip_p99_us`, `chip_max_us`, `chip_min_us`

### Neuromorphic Metrics (when controller.model exists)

- `total_syops`, `syops_per_step`: Synaptic operations
- `effective_macs`, `effective_acs`, `dense`: MAC/AC counts
- `activation_sparsity`: Activation sparsity [0-1]
- `footprint`: Model memory footprint
- `connection_sparsity`: Connection sparsity [0-1]
- `nb_*`: All NeuroBench raw keys

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `RuntimeError: TensorControllerAdapter.configure() must be called` | Call `adapter.configure(config, task)` |
| `KeyError: 'mae_i_q'` | Metric not included in harness - add `TrackingMAE(...)` |
| Safety violation | Check `task.terminated_by_safety` and `task.last_violation_reason` |
| NaN in results | Check for division by zero in metric `compute()` |
| Memory issues | Reduce `max_steps` or close physics engine |

## Type Hints

```python
from embark.benchmark.interfaces.types import (
    StateDict,      # dict[str, float]
    ActionDict,     # dict[str, float]
    ReferenceDict,  # dict[str, float]
    SystemConfig,   # PMSMConfig
)
```

## Protocol Interfaces

```python
from embark.benchmark.interfaces import (
    Controller,           # Unified interface
    TensorController,     # Neural controllers
    ClosedLoopTask,       # Task interface
    PhysicsEngine,        # Physics interface
    MetricAccumulator,    # Metric interface
    StateProcessor,       # State processor
    ActionProcessor,      # Action processor
)
```
