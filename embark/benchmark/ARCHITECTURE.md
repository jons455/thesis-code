# Benchmark Architecture

Closed-loop benchmark framework for neuromorphic PMSM current control, **adapted from** the [NeuroBench](https://github.com/NeuroBench/neurobench) modular harness pattern.

> **Note on NeuroBench Alignment:** This framework is _adapted from_ NeuroBench, not strictly _aligned with_ it.
> We redefine `MetricAccumulator` (vs NeuroBench's `AccumulatedMetric`) to support closed-loop control.
> For true NeuroBench alignment, wrap official `neurobench.metrics.WorkloadMetric` inside our accumulators.

## Overview

This framework extends NeuroBench patterns for **closed-loop control** benchmarks with:
- Physics simulation in the loop
- Reference tracking
- Safety termination
- Two-phase safety checks (action limits → physics → state limits)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ClosedLoopHarness                            │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │                    Unified Control Loop                   │    │
│    │                                                          │    │
│    │   state, ref ──▶ controller(state, ref) ──▶ action      │    │
│    │                         │                                │    │
│    │   ┌─────────────────────┼─────────────────────┐         │    │
│    │   │                     ▼                     │         │    │
│    │   │  ┌─────────────────────────────────────┐  │         │    │
│    │   │  │  Classical: PIControllerAgent       │  │         │    │
│    │   │  │  (implements Controller directly)   │  │         │    │
│    │   │  └─────────────────────────────────────┘  │         │    │
│    │   │                    OR                     │         │    │
│    │   │  ┌─────────────────────────────────────┐  │         │    │
│    │   │  │  Neural: TensorControllerAdapter    │  │         │    │
│    │   │  │  ┌───────────┐  ┌───────────────┐   │  │         │    │
│    │   │  │  │ StateProc │─▶│ SNN.forward() │   │  │         │    │
│    │   │  │  └───────────┘  └───────┬───────┘   │  │         │    │
│    │   │  │                 ┌───────▼───────┐   │  │         │    │
│    │   │  │                 │  ActionProc   │   │  │         │    │
│    │   │  │                 └───────────────┘   │  │         │    │
│    │   │  └─────────────────────────────────────┘  │         │    │
│    │   └───────────────────────────────────────────┘         │    │
│    └──────────────────────────────────────────────────────────┘    │
│                                                                     │
│    ┌─────────────┐         ┌─────────────────────────────────┐     │
│    │    Task     │────────▶│     MetricAccumulators          │     │
│    │ (physics +  │         │  (RMSE, SyOps, SettlingTime)    │     │
│    │  reference  │         └─────────────────────────────────┘     │
│    │  + safety)  │                                                 │
│    └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. No If/Else in Harness

The harness uses a **unified Controller interface**. There is no conditional logic based on controller type:

```python
# Harness always does this - same for PI and SNN
while not done:
    action = controller(state, reference)
    state, ref, done = task.step(action)
```

**How this works:**
- Classical controllers (PI) implement `Controller` directly
- Neural controllers (SNN) are wrapped with `TensorControllerAdapter`

### 2. Dependency Injection

Tasks don't create their dependencies - they receive them:

```python
# Composable task - swap reference generator for different benchmarks
task = PMSMCurrentControlTask(
    physics_engine=engine,
    reference_generator=StepReference(i_q_ref=2.0),     # Step response
    # OR: reference_generator=SinusoidalReference(...),  # Tracking
)
```

### 3. Two-Phase Safety Termination

The **Task** (not physics engine) defines safety limits with two-phase checking:

```
Controller → action → [Check Action Limits] → Physics → state → [Check State Limits]
                              ↓                                        ↓
                    (voltage, NaN)                            (current, speed, NaN)
```

```python
task = PMSMCurrentControlTask(
    ...,
    safety_limits=SafetyLimits(
        max_voltage_v=60.0,  # Phase 1: BEFORE physics (prevents crazy commands)
        max_current_a=20.0,  # Phase 2: AFTER physics (detects instability)
    ),
)
```

If limits are exceeded:
- `done=True`
- `task.terminated_by_safety` is `True`
- `task.last_violation_reason` contains details (e.g., `"current_limit_exceeded:i_q=25.3A"`)

### 4. Metric Accumulator Contract

Accumulators have two distinct phases:

| Method | When Called | Returns | Performance |
|--------|-------------|---------|-------------|
| `update()` | Every timestep | `None` | O(1) - constant time |
| `compute()` | End of episode | `dict` | May iterate/sqrt/etc. |

**What O(1) means:**
- ✅ Arithmetic: `x + y`, `x * y`, `x ** 2` (squaring is fine!)
- ✅ Dict access: `state["i_q"]`
- ✅ Incrementing counters: `self.count += 1`
- ❌ Iterating over history: `for x in self.all_errors`
- ❌ Sorting: `sorted(self.values)`
- ❌ Growing unbounded lists: `self.history.append(x)` (memory, not time)

```python
class TrackingRMSE(MetricAccumulator):
    def update(self, state, ref, action, next_state, info):
        # O(1): squaring, addition, dict access
        error = ref["i_q_ref"] - next_state["i_q"]
        self._sum_sq_error += error ** 2  # Squaring is O(1), fine here
        self._count += 1

    def compute(self):
        # Called once at end - sqrt, division OK here
        return {"rmse_i_q": sqrt(self._sum_sq_error / self._count)}
```

## Core Components

### Protocols (Interfaces)

| Protocol | Purpose | Location |
|----------|---------|----------|
| `Controller` | Unified controller interface | `interfaces/controller.py` |
| `TensorController` | Neural controllers (before wrapping) | `interfaces/controller.py` |
| `PhysicsEngine` | Pure dynamics simulation | `interfaces/physics.py` |
| `ClosedLoopTask` | Control objective + reference + safety | `interfaces/task.py` |
| `StateProcessor` | State dict → tensor | `interfaces/processors.py` |
| `ActionProcessor` | Tensor → action dict | `interfaces/processors.py` |
| `MetricAccumulator` | Real-time metric computation | `interfaces/metrics.py` |

### Controller Protocol

```python
class Controller(Protocol):
    """Unified interface - both classical and neural controllers."""

    def reset(self) -> None: ...
    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

### TensorControllerAdapter

Wraps a `TensorController` + processors into the unified `Controller` interface.

**Important:** The adapter exposes intermediate values (not a "black hole"):

```python
@dataclass
class TensorControllerAdapter:
    controller: TensorController      # e.g., SNNControllerAgent
    state_processor: StateProcessor   # e.g., MinMaxProcessor
    action_processor: ActionProcessor # e.g., LinearActionProcessor

    def __call__(self, state, reference) -> ActionDict:
        self._last_observation = self.state_processor(state, reference)
        self._last_action_tensor = self.controller.forward(self._last_observation)
        return self.action_processor(self._last_action_tensor, config)

    # Exposed for metrics and external tools:
    @property
    def model(self) -> TensorController:
        """Direct access to underlying controller for hook registration."""
        return self.controller

    @property
    def last_observation(self) -> Tensor:
        """Input tensor (normalized) from last step."""

    @property
    def last_action_tensor(self) -> Tensor:
        """Output tensor (normalized, before denormalization)."""

    @property
    def last_info(self) -> dict | None:
        """Spike stats from controller (forwarded, not swallowed)."""
```

**Using with NeuroBench WorkloadMetric:**

```python
from neurobench.metrics import WorkloadMetric

adapter = TensorControllerAdapter(snn, state_proc, action_proc)

# Access underlying model for hook registration
workload = WorkloadMetric(adapter.model, ...)
```

### Physics Engine

```python
class PMSMPhysicsEngine:
    def reset(self, seed=None) -> StateDict:
        """Returns: {"i_d", "i_q", "omega", "epsilon", "time"}"""

    def step(self, action: ActionDict) -> tuple[StateDict, dict]:
        """action: {"v_d", "v_q"} in Volts"""

    def close(self) -> None: ...
```

### Task

```python
@dataclass
class PMSMCurrentControlTask:
    physics_engine: PMSMPhysicsEngine
    reference_generator: ReferenceGenerator  # Injected, not created
    safety_limits: SafetyLimits | None

    def reset(self) -> tuple[StateDict, ReferenceDict]: ...
    def step(self, action) -> tuple[StateDict, ReferenceDict, bool]: ...

    @property
    def terminated_by_safety(self) -> bool:
        """True if episode ended due to safety violation."""
```

### Safety Limits

```python
@dataclass
class SafetyLimits:
    max_current_a: float | None = 20.0   # Max |i_d| or |i_q|
    max_voltage_v: float | None = None   # Max |v_d| or |v_q|
    max_speed_rpm: float | None = None   # Max omega

    def check(self, state, action=None) -> bool:
        """Returns True if any limit violated."""
```

## Directory Structure

```
embark/benchmark/
├── ARCHITECTURE.md           # This file
├── __init__.py
├── agents.py                 # PIControllerAgent, SNNControllerAgent
├── adapters/
│   ├── __init__.py
│   └── tensor_adapter.py     # TensorControllerAdapter
├── harness/
│   └── closed_loop.py        # ClosedLoopHarness (unified, no if/else)
├── physics/
│   ├── config.py             # PMSMConfig
│   └── pmsm.py               # PMSMPhysicsEngine
├── tasks/
│   ├── pmsm_current_control.py  # PMSMCurrentControlTask, SafetyLimits
│   └── reference_generators.py  # StepReference, SinusoidalReference, etc.
├── processors/
│   ├── normalizers.py        # MinMaxProcessor, StandardScalerProcessor
│   ├── decoders.py           # LinearActionProcessor
│   └── identity.py           # IdentityStateProcessor
├── metrics/accumulators/
│   ├── tracking.py           # TrackingRMSE (update=O(1), compute=sqrt)
│   ├── dynamics.py           # SettlingTime, Overshoot
│   ├── efficiency.py         # ControlEffort
│   └── neuromorphic.py       # SyOpsAccumulator
├── interfaces/               # Protocol definitions
│   ├── controller.py         # Controller, TensorController
│   ├── physics.py
│   ├── task.py
│   ├── processors.py
│   ├── metrics.py
│   └── types.py
├── controllers/neural/
│   ├── snn_wrapper.py
│   └── ann_wrapper.py
└── contrib/neurobench/       # Optional NeuroBench interop (experimental)
    ├── model_wrapper.py      # NeuroBenchClosedLoopModel
    └── result_exporter.py    # ClosedLoopMetricExporter
```

## Usage Examples

### PI Controller Benchmark

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PIControllerAgent,
    PMSMCurrentControlTask,
    SafetyLimits,
    TrackingRMSE,
    SettlingTime,
)

# Create task with safety limits
task = PMSMCurrentControlTask.from_config(
    n_rpm=1000,
    i_q_ref=2.0,
    max_steps=1000,
    safety_limits=SafetyLimits(max_current_a=15.0),
)

# Classical controller - implements Controller directly
controller = PIControllerAgent.from_system_config(task.physics_engine.config)

# Metrics with proper update/compute contract
metrics = [
    TrackingRMSE(tracked_keys=["i_q", "i_d"]),
    SettlingTime(tracked_key="i_q"),
]

# Run benchmark
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()

print(f"RMSE: {results['rmse_i_q']*1000:.2f} mA")
print(f"Safety terminated: {task.terminated_by_safety}")

task.physics_engine.close()
```

### SNN Controller Benchmark

```python
from embark.benchmark import (
    PMSMCurrentControlTask,
    ClosedLoopHarness,
    TensorControllerAdapter,
    TrackingRMSE,
    SyOpsAccumulator,
)
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark.processors import MinMaxProcessor, LinearActionProcessor

# Create task
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0, max_steps=1000)

# Create neural controller (TensorController)
snn = SNNControllerAgent("path/to/checkpoint.pt", track_spikes=True)

# Create processors
state_proc = MinMaxProcessor(
    input_keys=["i_d", "i_q"],
    reference_keys=["i_d_ref", "i_q_ref"],
)
action_proc = LinearActionProcessor(
    output_keys=["v_d", "v_q"],
    bounds={"v_d": (-48, 48), "v_q": (-48, 48)},
)

# Wrap into unified Controller interface
controller = TensorControllerAdapter(
    controller=snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
controller.configure(task.physics_engine.config, task)

# Run benchmark - same harness, same loop, no special handling
harness = ClosedLoopHarness(
    task=task,
    controller=controller,
    metrics=[TrackingRMSE(tracked_keys=["i_q"]), SyOpsAccumulator()],
)
results = harness.run()

task.physics_engine.close()
```

## Extending the Framework

### Adding a New Controller

**Classical (dict-based):**
1. Implement `Controller` protocol directly
2. `__call__(state, reference) -> action` where all are dicts

**Neural (tensor-based):**
1. Implement `TensorController` protocol
2. Use `TensorControllerAdapter` to wrap with processors
3. Implement `forward(obs: Tensor) -> Tensor`

### Adding a New Reference Generator

1. Implement `ReferenceGenerator` protocol
2. Inject into task: `PMSMCurrentControlTask(reference_generator=MyGenerator())`

### Adding a New Metric

1. Implement `MetricAccumulator` protocol
2. `update()` must be O(1) - no expensive operations
3. `compute()` may be slow - called once at end

### Adding a New Physics Engine

1. Implement `PhysicsEngine` protocol
2. Define `state_keys`, `action_keys` properties
3. Implement `reset()`, `step()`, `close()`
4. Create corresponding `Config` dataclass

## Non-PyTorch Controllers (Akida, Keras, TensorFlow)

The architecture is **framework-agnostic**. While `TensorControllerAdapter` is a convenience for PyTorch, you can use any framework by implementing the `Controller` protocol directly.

**Example: Akida Agent**

```python
class AkidaControllerAgent(Controller):
    def __init__(self, model_path):
        self.model = akida.Model(model_path)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        # 1. Manual Preprocessing (Dict -> Numpy)
        # (Or use a custom NumpyProcessor)
        obs_numpy = my_numpy_normalizer(state, reference)

        # 2. Inference
        action_numpy = self.model.predict(obs_numpy)

        # 3. Manual Postprocessing (Numpy -> Dict)
        return {"v_d": float(action_numpy[0]), "v_q": float(action_numpy[1])}

    def reset(self):
        # Reset Akida states
        pass
```

You would then use it in the harness exactly like a PI controller:

```python
akida_agent = AkidaControllerAgent("model.fbz")
harness = ClosedLoopHarness(task=task, controller=akida_agent)
```

## Comparison with NeuroBench

| NeuroBench | This Framework | Notes |
|------------|----------------|-------|
| `NeuroBenchModel` | `Controller` / `TensorController` | Unified via adapter |
| `Preprocessor` | `StateProcessor` | Inside `TensorControllerAdapter` |
| `Postprocessor` | `ActionProcessor` | Inside `TensorControllerAdapter` |
| `AccumulatedMetric` | `MetricAccumulator` | Redefined for control |
| `BenchmarkHarness` | `ClosedLoopHarness` | Extended with physics loop |
| `WorkloadMetric` | (use via `adapter.model`) | Not wrapped, use directly |

**Key differences from NeuroBench:**

1. **Closed-loop:** Physics simulation in the loop, not single-shot inference
2. **Two-phase safety:** Action limits → physics → state limits
3. **Metric redefinition:** `MetricAccumulator` has `controller_info` parameter for spike stats
4. **Adapter pattern:** `TensorControllerAdapter` exposes `model` for NeuroBench tool compatibility

**For true NeuroBench metrics alignment:**

```python
from neurobench.metrics import WorkloadMetric

# 1. Wrap controller
adapter = TensorControllerAdapter(snn, ...)

# 2. Register NeuroBench hooks on the UNWRAPPED model
# adapter.model automatically unwraps SNNControllerAgent to get the nn.Module
workload = WorkloadMetric(adapter.model, ...)

# 3. Run harness
harness = ClosedLoopHarness(
    task=task,
    controller=adapter,
    metrics=[workload, ...] # WorkloadMetric works because hooks are registered
)
```
