# Benchmark API Reference

Complete API reference for the `embark.benchmark` module.

## Table of Contents

- [Core Components](#core-components)
- [Harness](#harness)
- [Tasks](#tasks)
- [Controllers](#controllers)
- [Processors](#processors)
- [Metrics](#metrics)
- [Physics](#physics)
- [Interfaces](#interfaces)

---

## Core Components

### `ClosedLoopHarness`

Main orchestrator for running closed-loop control benchmarks.

**Location:** `embark.benchmark.harness.closed_loop`

**Constructor:**
```python
ClosedLoopHarness(
    task: ClosedLoopTask,
    controller: Controller,
    metrics: list[MetricAccumulator] | None = None,
)
```

**Methods:**

- `run(max_steps: int | None = None) -> dict[str, Any]`
  - Runs one episode of the benchmark
  - Returns dictionary with step count and all metric results
  - Keys include `steps` plus all keys from metric `compute()` methods

**Example:**
```python
from embark.benchmark import ClosedLoopHarness, PMSMCurrentControlTask, PIControllerAgent

task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
controller = PIControllerAgent.from_system_config(task.physics_engine.config)
harness = ClosedLoopHarness(task=task, controller=controller)
results = harness.run()
```

---

### `BenchmarkSuite`

Multi-scenario benchmark runner for standardized evaluation.

**Location:** `embark.benchmark.harness.benchmark_suite`

**Constructor:**
```python
BenchmarkSuite(
    scenarios: list[ScenarioDefinition] | None = None,
    metric_factory: Callable | None = None,
    verbose: bool = True,
)
```

**Methods:**

- `run(controller: Controller, name: str = "Controller") -> BenchmarkSummary`
  - Runs controller through all scenarios
  - Returns aggregated summary with per-scenario results

- `print_summary(summary: BenchmarkSummary) -> None` (static)
  - Prints formatted comparison table

- `save_results(summary: BenchmarkSummary, path: str | Path) -> None` (static)
  - Saves results to JSON file

**Standard Scenarios:**

- `STANDARD_SCENARIOS`: 6 scenarios (step responses, sinusoidal tracking, flux weakening)
- `QUICK_SCENARIOS`: 3 scenarios (subset for quick validation)

**Example:**
```python
from embark.benchmark import BenchmarkSuite

suite = BenchmarkSuite()
summary = suite.run(controller=my_controller, name="MySNN")
suite.print_summary(summary)
suite.save_results(summary, "results/benchmark.json")
```

---

## Tasks

### `PMSMCurrentControlTask`

Closed-loop PMSM current control task with physics engine and reference generation.

**Location:** `embark.benchmark.tasks.pmsm_current_control`

**Class Methods:**

- `from_config(n_rpm: float, i_d_ref: float = 0.0, i_q_ref: float = 0.0, max_steps: int = 2000, safety_limits: SafetyLimits | None = None) -> PMSMCurrentControlTask`
  - Convenience factory for step reference scenarios

**Constructor:**
```python
PMSMCurrentControlTask(
    physics_engine: PMSMPhysicsEngine,
    reference_generator: ReferenceGenerator,
    max_steps: int = 2000,
    safety_limits: SafetyLimits | None = None,
)
```

**Properties:**

- `physics_engine: PMSMPhysicsEngine` - The physics simulation engine
- `reference_keys: set[str]` - Keys in reference dict (e.g., `{"i_d_ref", "i_q_ref"}`)
- `max_steps: int | None` - Maximum episode length
- `terminated_by_safety: bool` - True if episode ended due to safety violation
- `last_violation_reason: str | None` - Details of safety violation if occurred

**Methods:**

- `reset(seed: int | None = None) -> tuple[StateDict, ReferenceDict]`
  - Resets task and physics engine
  - Returns initial state and reference

- `step(action: ActionDict) -> tuple[StateDict, ReferenceDict, bool]`
  - Advances simulation one timestep
  - Returns next state, reference, and done flag

**Example:**
```python
from embark.benchmark import PMSMCurrentControlTask, SafetyLimits

task = PMSMCurrentControlTask.from_config(
    n_rpm=1000,
    i_q_ref=2.0,
    max_steps=2000,
    safety_limits=SafetyLimits(max_current_a=15.0),
)
```

---

### `SafetyLimits`

Safety limits for early episode termination.

**Location:** `embark.benchmark.tasks.pmsm_current_control`

**Constructor:**
```python
SafetyLimits(
    max_current_a: float | None = 20.0,
    max_voltage_v: float | None = None,
    max_speed_rpm: float | None = None,
)
```

**Methods:**

- `check_action(action: dict[str, float]) -> str | None`
  - Checks action limits BEFORE physics step
  - Returns violation reason or None

- `check_state(state: StateDict) -> str | None`
  - Checks state limits AFTER physics step
  - Returns violation reason or None

---

### Reference Generators

**Location:** `embark.benchmark.tasks.reference_generators`

#### `StepReference`

Step reference signal generator.

```python
StepReference(
    i_d_ref: float = 0.0,
    i_q_ref: float = 0.0,
    step_time: float = 0.0,
)
```

#### `SinusoidalReference`

Sinusoidal reference signal generator.

```python
SinusoidalReference(
    i_d_ref: float = 0.0,
    i_q_amp: float = 0.0,
    i_q_offset: float = 0.0,
    frequency_hz: float = 1.0,
    phase: float = 0.0,
)
```

#### `ConstantReference`

Constant reference signal generator.

```python
ConstantReference(
    i_d_ref: float = 0.0,
    i_q_ref: float = 0.0,
)
```

---

## Controllers

### `PIControllerAgent`

Classical PI controller implementing `Controller` protocol directly.

**Location:** `embark.benchmark.agents`

**Class Methods:**

- `from_system_config(config: SystemConfig) -> PIControllerAgent`
  - Creates PI controller from physics engine config

**Constructor:**
```python
PIControllerAgent(
    params: PIParameters | None = None,
    decoupling: bool = True,
    anti_windup: bool = True,
    anti_windup_decay: float = 0.99,
    kp_d: float | None = None,
    ki_d: float | None = None,
    kp_q: float | None = None,
    ki_q: float | None = None,
)
```

**Features:**

- Technical Optimum tuning: `Kp = L / (2*τ)`, `Ki = R / (2*τ)`
- Decoupling compensation (back-EMF)
- Anti-windup protection
- Voltage limiting

**Methods:**

- `reset() -> None` - Reset integrator states
- `__call__(state: StateDict, reference: ReferenceDict) -> ActionDict` - Compute control action
- `get_state() -> dict[str, Any]` - Serialize internal state
- `set_state(state: dict[str, Any]) -> None` - Restore internal state

---

### `SNNControllerAgent`

Spiking Neural Network controller implementing `TensorController` protocol.

**Location:** `embark.benchmark.agents`

**Constructor:**
```python
SNNControllerAgent(
    model_path: str,
    device: str = "cpu",
    track_spikes: bool = False,
)
```

**Methods:**

- `reset() -> None` - Reset membrane potentials
- `forward(observation: torch.Tensor) -> torch.Tensor` - Forward pass
- `get_state() -> dict[str, Any]` - Serialize model state
- `set_state(state: dict[str, Any]) -> None` - Restore model state

**Properties:**

- `model: torch.nn.Module` - Underlying PyTorch model (for hook registration)
- `last_info: dict[str, Any] | None` - Spike statistics if `track_spikes=True`

---

### `TensorControllerAdapter`

Wraps `TensorController` + processors into unified `Controller` interface.

**Location:** `embark.benchmark.adapters.tensor_adapter`

**Constructor:**
```python
TensorControllerAdapter(
    controller: TensorController,
    state_processor: StateProcessor,
    action_processor: ActionProcessor,
)
```

**Methods:**

- `configure(physics_config: SystemConfig, task: ClosedLoopTask) -> None`
  - Configure processors with physics bounds (must be called before use)

- `reset() -> None` - Reset controller and clear intermediate values

- `__call__(state: StateDict, reference: ReferenceDict) -> ActionDict`
  - Process state, run controller, process action

**Properties:**

- `model: TensorController` - Direct access to underlying controller
- `last_observation: torch.Tensor | None` - Input tensor (normalized)
- `last_action_tensor: torch.Tensor | None` - Output tensor (before denormalization)
- `last_info: dict[str, Any] | None` - Spike stats from controller

**Example:**
```python
from embark.benchmark import TensorControllerAdapter
from embark.benchmark.processors import RateSNNStateProcessor, RateSNNActionProcessor

snn = SNNControllerAgent("path/to/model.pt", track_spikes=True)
state_proc = RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    i_max=20.0,
    n_max=4000.0,
)
action_proc = RateSNNActionProcessor(incremental=False, u_max=48.0)

adapter = TensorControllerAdapter(
    controller=snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
adapter.configure(task.physics_engine.config, task)
```

---

## Processors

**Location:** `embark.benchmark.processors`

### State Processors

Convert state dict → tensor for neural controllers.

#### `RateSNNStateProcessor`

Configurable state processor for rate-encoding SNNs with feature flags.

```python
RateSNNStateProcessor(
    include_currents: bool = True,
    include_errors: bool = True,
    include_speed: bool = True,
    include_references: bool = False,
    include_prev_action: bool = False,
    include_derivatives: bool = False,
    include_ema_slow: bool = False,
    include_ema_fast: bool = False,
    include_integral: bool = False,
    i_max: float = 20.0,
    n_max: float = 4000.0,
    error_gain: float = 10.0,
    clip_errors: bool = True,
)
```

**Methods:**
- `reset()` - Clear stateful features (derivatives, EMA, integrals, prev_action)
- `set_prev_action(u_d, u_q)` - Set previous action for incremental models
- `output_dim: int` - Number of features produced

#### `IdentityStateProcessor`

Passthrough processor (no transformation).

```python
IdentityStateProcessor()
```

---

### Action Processors

Convert tensor → action dict for neural controllers.

#### `RateSNNActionProcessor`

Action processor for rate-encoding SNNs with absolute and incremental modes.

```python
RateSNNActionProcessor(
    incremental: bool = False,
    u_max: float = 48.0,
    delta_max: float = 0.2,
)
```

**Methods:**
- `reset()` - Clear accumulated voltage (for incremental mode)
- `last_accumulated_voltage: tuple[float, float]` - Get (u_d, u_q) state
    scale: float = 1.0,
)
```

#### `PWMActionProcessor`

PWM conversion with dead-time compensation.

```python
PWMActionProcessor(
    output_keys: list[str],
    bounds: dict[str, tuple[float, float]],
    dead_time_s: float = 0.0,
)
```

#### `IdentityActionProcessor`

Passthrough processor (no transformation).

```python
IdentityActionProcessor()
```

---

## Metrics

**Location:** `embark.benchmark.metrics.accumulators`

All metrics implement `MetricAccumulator` protocol with O(1) `update()` and deferred `compute()`.

### Tracking Metrics

#### `TrackingMAE`

Mean Absolute Error for tracking performance.

```python
TrackingMAE(
    tracked_keys: list[str],
    name: str = "mae",
)
```

**Output keys:** `mae_{key}` for each tracked key

#### `TrackingITAE`

Integral Time Absolute Error (penalizes sustained errors).

```python
TrackingITAE(
    tracked_keys: list[str],
    name: str = "itae",
)
```

**Output keys:** `itae_{key}` for each tracked key

#### `MaximumError`

Maximum absolute error (worst-case safety metric).

```python
MaximumError(
    tracked_keys: list[str],
    name: str = "max_error",
)
```

**Output keys:** `max_error_{key}` for each tracked key

---

### Dynamics Metrics

#### `SettlingTime`

Time until error stays within threshold.

```python
SettlingTime(
    tracked_key: str,
    threshold: float = 0.05,
    name: str = "settling_time",
)
```

**Output key:** `settling_time` (seconds, or `inf` if never settles)

#### `Overshoot`

Percentage overshoot relative to final reference.

```python
Overshoot(
    tracked_key: str,
    name: str = "overshoot",
)
```

**Output key:** `overshoot` (percentage)

---

### Latency Metrics

#### `InferenceLatency`

Inference latency statistics.

**Location:** `embark.benchmark.metrics.accumulators.latency`

```python
InferenceLatency(name: str = "latency")
```

**Output keys:**
- `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `max_latency_ms`
- `jitter_ms`, `total_inference_time_s`
- `chip_mean_us`, `chip_median_us`, `chip_p95_us`, `chip_p99_us`, `chip_max_us`, `chip_min_us` (when chip timing available)

---

### Neuromorphic Metrics

NeuroBench adapters are automatically added by `create_metrics()` when controller has `.model` property.

**Location:** `embark.benchmark.metrics.neurobench_factory`

```python
create_metrics(controller: Controller | None = None) -> list[MetricAccumulator]
```

**Automatically includes:**
- Control metrics (MAE, ITAE, MaxError, SettlingTime, Overshoot)
- NeuroBench static metrics (Footprint, ConnectionSparsity) if `controller.model` exists
- NeuroBench workload metrics (SynapticOperations, ActivationSparsity) if `controller.model` exists

---

## Physics

### `PMSMPhysicsEngine`

Physics engine wrapper around GEM (gym-electric-motor).

**Location:** `embark.benchmark.physics.pmsm`

**Constructor:**
```python
PMSMPhysicsEngine(
    n_rpm: float = 1000.0,
    config: PMSMConfig = PMSMConfig(),
)
```

**Properties:**

- `config: PMSMConfig` - Motor configuration
- `state_keys: set[str]` - `{"i_d", "i_q", "omega", "epsilon", "time"}`
- `action_keys: set[str]` - `{"v_alpha", "v_beta"}`

**Methods:**

- `reset(seed: int | None = None) -> StateDict`
  - Reset to initial state

- `step(action: ActionDict) -> tuple[StateDict, dict[str, Any]]`
  - Execute one physics step
  - Action: `{"v_alpha": float, "v_beta": float}` or `{"v_d": float, "v_q": float}`

- `close() -> None`
  - Clean up resources

---

### `PMSMConfig`

PMSM motor configuration.

**Location:** `embark.benchmark.physics.config`

**Attributes:**

- `r_s: float` - Stator resistance (Ω)
- `l_d: float` - d-axis inductance (H)
- `l_q: float` - q-axis inductance (H)
- `psi_p: float` - Permanent magnet flux (Wb)
- `p: int` - Number of pole pairs
- `j_load: float` - Load inertia (kg·m²)
- `tau: float` - Sampling period (s)
- `i_max: float` - Maximum current (A)
- `u_max: float` - Maximum voltage (V)
- `use_dead_time: bool` - Enable PWM dead-time
- `dead_time: float` - Dead-time duration (s)

---

## Interfaces

Protocol definitions for type checking and extension.

**Location:** `embark.benchmark.interfaces`

### `Controller`

Unified controller interface (both classical and neural).

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
```

### `TensorController`

Neural network controller interface (before wrapping).

```python
class TensorController(Protocol):
    def reset(self) -> None: ...
    def forward(self, observation: torch.Tensor) -> torch.Tensor: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
```

### `ClosedLoopTask`

Task interface for control objectives.

```python
class ClosedLoopTask(Protocol):
    @property
    def physics_engine(self) -> PhysicsEngine: ...
    @property
    def reference_keys(self) -> set[str]: ...
    @property
    def max_steps(self) -> int | None: ...
    def reset(self, seed: int | None = None) -> tuple[StateDict, ReferenceDict]: ...
    def step(self, action: ActionDict) -> tuple[StateDict, ReferenceDict, bool]: ...
```

### `PhysicsEngine`

Physics simulation interface.

```python
class PhysicsEngine(Protocol):
    @property
    def config(self) -> SystemConfig: ...
    def reset(self, seed: int | None = None) -> StateDict: ...
    def step(self, action: ActionDict) -> tuple[StateDict, dict[str, Any]]: ...
    def close(self) -> None: ...
    @property
    def state_keys(self) -> set[str]: ...
    @property
    def action_keys(self) -> set[str]: ...
```

### `MetricAccumulator`

Metric computation interface.

```python
class MetricAccumulator(Protocol):
    @property
    def name(self) -> str: ...
    def reset(self) -> None: ...
    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,
        next_state: StateDict,
        controller_info: dict[str, Any] | None = None,
    ) -> None: ...
    def compute(self) -> float | dict[str, float]: ...
```

### `StateProcessor` / `ActionProcessor`

Processor interfaces for tensor conversion.

```python
class StateProcessor(Protocol):
    def configure(self, config: SystemConfig, task: ClosedLoopTask) -> None: ...
    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor: ...

class ActionProcessor(Protocol):
    def configure(self, config: SystemConfig) -> None: ...
    def __call__(self, action_tensor: torch.Tensor, config: SystemConfig) -> ActionDict: ...
```

---

## Type Definitions

**Location:** `embark.benchmark.interfaces.types`

- `StateDict: dict[str, float]` - System state (e.g., `{"i_d": 0.0, "i_q": 0.0, "omega": 104.7, ...}`)
- `ActionDict: dict[str, float]` - Control action (e.g., `{"v_d": 12.0, "v_q": -5.0}`)
- `ReferenceDict: dict[str, float]` - Reference signals (e.g., `{"i_d_ref": 0.0, "i_q_ref": 2.0}`)
- `SystemConfig` - Physics configuration dataclass
