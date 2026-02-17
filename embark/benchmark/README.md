# PMSM Current Control Benchmark

NeuroBench-aligned closed-loop benchmark framework for neuromorphic PMSM current control.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and design principles
- **[API Reference](../docs/benchmark/BENCHMARK_API.md)** - Complete API documentation
- **[User Guide](../docs/benchmark/BENCHMARK_USER_GUIDE.md)** - Usage examples and tutorials
- **[Quick Reference](../docs/benchmark/BENCHMARK_QUICK_REFERENCE.md)** - Cheat sheet for common operations
- **[Metrics Reference](../docs/reference/METRICS.md)** - Metric definitions and output formats

## Architecture

```
benchmark/
├── agents.py              # PIControllerAgent (baseline), SNNControllerAgent
├── adapters/
│   └── tensor_adapter.py  # TensorControllerAdapter (wraps neural + processors)
├── harness/
│   └── closed_loop.py     # ClosedLoopHarness - unified control loop
├── physics/
│   ├── config.py          # PMSMConfig
│   └── pmsm.py            # PMSMPhysicsEngine (wraps GEM)
├── tasks/
│   ├── pmsm_current_control.py   # PMSMCurrentControlTask, SafetyLimits
│   └── reference_generators.py   # StepReference, SinusoidalReference
├── processors/            # For neural controllers (inside adapter)
│   ├── normalizers.py     # MinMaxProcessor, StandardScalerProcessor
│   ├── decoders.py        # LinearActionProcessor
│   └── identity.py        # Passthrough processors
├── metrics/accumulators/  # Real-time metric computation
│   ├── tracking.py        # TrackingRMSE, TrackingMAE
│   ├── dynamics.py        # SettlingTime, Overshoot
│   ├── efficiency.py      # ControlEffort
│   └── neuromorphic.py    # SyOpsAccumulator, SpikeCountAccumulator
├── interfaces/            # Protocol definitions
│   ├── controller.py      # Controller, TensorController
│   ├── physics.py         # PhysicsEngine
│   ├── task.py            # ClosedLoopTask
│   └── metrics.py         # MetricAccumulator
├── controllers/neural/    # Wrappers for trained models
│   └── snn_wrapper.py     # SNNControllerWrapper
├── controllers/remote/
│   └── akida_policy.py    # RemoteAkidaPolicy (hardware-in-the-loop)
└── contrib/neurobench/    # Optional NeuroBench interop (experimental)
    ├── model_wrapper.py   # NeuroBenchClosedLoopModel
    └── result_exporter.py # ClosedLoopMetricExporter
```

### Core vs. Contrib

**Core** (`embark/benchmark/`) runs closed-loop benchmarks with zero NeuroBench
dependencies. All protocols, the harness, physics, processors, and metrics live here.

**Contrib** (`embark/benchmark/contrib/neurobench/`) provides optional adapters for
NeuroBench interop: exposing controllers in the format NeuroBench tools expect
(`.net` property for hook registration) and reformatting results into NeuroBench-compatible
dicts. See [`NEUROBENCH_INTEGRATION_ROADMAP.md`](../../docs/reference/NEUROBENCH_INTEGRATION_ROADMAP.md)
for the phased integration plan.

## Quick Start

### Run PI Controller (Baseline)

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PIControllerAgent,
    PMSMCurrentControlTask,
    TrackingRMSE,
    SettlingTime,
)

# Create task
task = PMSMCurrentControlTask.from_config(
    n_rpm=1000,        # Motor speed [rpm]
    i_d_ref=0.0,       # d-axis current reference [A]
    i_q_ref=2.0,       # q-axis current reference [A]
    max_steps=1000,
)

# Create PI controller (auto-tuned using Technical Optimum)
controller = PIControllerAgent.from_system_config(task.physics_engine.config)

# Define metrics
metrics = [
    TrackingRMSE(tracked_keys=["i_q", "i_d"]),
    SettlingTime(tracked_key="i_q", threshold=0.02),
]

# Run benchmark
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()

print(f"RMSE i_q: {results['rmse_i_q']*1000:.2f} mA")
print(f"Settling time: {results['settling_time']*1000:.1f} ms")

task.physics_engine.close()
```

### Run SNN Controller

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PMSMCurrentControlTask,
    TensorControllerAdapter,
    TrackingRMSE,
    SyOpsAccumulator,
)
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark.processors import RateSNNStateProcessor, RateSNNActionProcessor

# Create task
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0, max_steps=1000)

# Load trained SNN (TensorController)
snn = SNNControllerAgent("path/to/checkpoint.pt", track_spikes=True)

# Processors convert dict<->tensor
state_proc = RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    i_max=20.0,
    n_max=4000.0,
)
action_proc = RateSNNActionProcessor(incremental=False, u_max=48.0)

# Wrap with TensorControllerAdapter for unified interface
controller = TensorControllerAdapter(
    controller=snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
controller.configure(task.physics_engine.config, task)

# Metrics
# Note: SyOpsAccumulator uses data passed via controller_info (from track_spikes=True)
# For NeuroBench WorkloadMetric, pass `controller.model` to the metric constructor.
metrics = [TrackingRMSE(tracked_keys=["i_q"]), SyOpsAccumulator()]

# Run - same harness interface as PI controller
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()

print(f"RMSE: {results['rmse_i_q']*1000:.2f} mA")
print(f"SyOps/step: {results['syops_per_step']:.1f}")

task.physics_engine.close()
```

## Controller Protocols

### Controller (Unified Interface)

All controllers (classical and neural) must conform to this unified interface:

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def __call__(self, state: dict, reference: dict) -> dict: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

- Classical controllers (PI) implement `Controller` directly
- Neural controllers must be wrapped with `TensorControllerAdapter`

### TensorController (Neural - Before Wrapping)

For SNN/ANN controllers that work with tensors:

```python
class TensorController(Protocol):
    def reset(self) -> None: ...
    def forward(self, observation: torch.Tensor) -> torch.Tensor: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

Use `TensorControllerAdapter` to wrap with processors and conform to `Controller`.

## PI Controller Details

The baseline `PIControllerAgent` in `agents.py` uses **Technical Optimum** tuning:

```
Kp_d = L_d / (2 * tau)
Ki_d = R_s / (2 * tau)
Kp_q = L_q / (2 * tau)
Ki_q = R_s / (2 * tau)
```

Features:
- Decoupling compensation (back-EMF)
- Anti-windup
- Voltage limiting

## Physics Engine

`PMSMPhysicsEngine` wraps GEM (gym-electric-motor) for realistic PMSM simulation:

- State: `{i_d, i_q, omega, epsilon, time}`
- Action: `{v_d, v_q}` or `{v_alpha, v_beta}` in Volts
- Configurable via `PMSMConfig`
