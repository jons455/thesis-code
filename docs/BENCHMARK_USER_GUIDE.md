# Benchmark User Guide

Complete guide for using the `embark.benchmark` framework to evaluate controllers.

## Table of Contents

- [Quick Start](#quick-start)
- [Running Benchmarks](#running-benchmarks)
- [Controller Setup](#controller-setup)
- [Custom Scenarios](#custom-scenarios)
- [Custom Metrics](#custom-metrics)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Run PI Controller Baseline

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PIControllerAgent,
    PMSMCurrentControlTask,
    TrackingMAE,
    SettlingTime,
)

# Create task
task = PMSMCurrentControlTask.from_config(
    n_rpm=1000,        # Motor speed [rpm]
    i_q_ref=2.0,       # q-axis current reference [A]
    max_steps=1000,
)

# Create PI controller
controller = PIControllerAgent.from_system_config(task.physics_engine.config)

# Define metrics
metrics = [
    TrackingMAE(tracked_keys=["i_q", "i_d"]),
    SettlingTime(tracked_key="i_q", threshold=0.02),
]

# Run benchmark
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()

print(f"MAE i_q: {results['mae_i_q']*1000:.2f} mA")
print(f"Settling time: {results['settling_time']*1000:.1f} ms")

task.physics_engine.close()
```

### 2. Run SNN Controller

```python
from embark.benchmark import (
    ClosedLoopHarness,
    PMSMCurrentControlTask,
    TensorControllerAdapter,
    TrackingMAE,
)
from embark.benchmark.agents import SNNControllerAgent
from embark.benchmark.processors import MinMaxProcessor, LinearActionProcessor

# Create task
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0, max_steps=1000)

# Load trained SNN
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

# Wrap with adapter
controller = TensorControllerAdapter(
    controller=snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
controller.configure(task.physics_engine.config, task)

# Run benchmark
harness = ClosedLoopHarness(
    task=task,
    controller=controller,
    metrics=[TrackingMAE(tracked_keys=["i_q"])],
)
results = harness.run()

print(f"MAE: {results['mae_i_q']*1000:.2f} mA")

task.physics_engine.close()
```

### 3. Run Multi-Scenario Suite

```python
from embark.benchmark import BenchmarkSuite

# Create controller (PI or SNN)
controller = PIControllerAgent.from_system_config(...)

# Run suite
suite = BenchmarkSuite()
summary = suite.run(controller=controller, name="MyController")

# Print results
suite.print_summary(summary)

# Save to file
suite.save_results(summary, "results/benchmark.json")
```

---

## Running Benchmarks

### Single Scenario

Use `ClosedLoopHarness` for a single scenario:

```python
harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run(max_steps=2000)  # Optional override
```

**Results format:**
```python
{
    "steps": 1000,
    "mae_i_q": 0.012,
    "mae_i_d": 0.001,
    "max_error_i_q": 0.95,
    "settling_time": 0.05,
    "overshoot": 4.2,
    # ... other metric keys
}
```

### Multi-Scenario Suite

Use `BenchmarkSuite` for standardized evaluation:

```python
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)  # or QUICK_SCENARIOS
summary = suite.run(controller=controller, name="MySNN")
```

**Summary structure:**
- `summary.controller_name`: Display name
- `summary.scenario_results`: List of per-scenario results
- `summary.mean_mae_iq`: Aggregate metric
- `summary.worst_max_error_iq`: Worst-case metric
- `summary.num_safety_violations`: Safety failure count

### Custom Metric Factory

Provide custom metrics per scenario:

```python
def my_metric_factory(controller):
    return [
        TrackingMAE(tracked_keys=["i_q"]),
        CustomMetric(...),  # Your custom metric
    ]

suite = BenchmarkSuite(metric_factory=my_metric_factory)
```

---

## Controller Setup

### PI Controller

**Direct usage (no adapter needed):**
```python
controller = PIControllerAgent.from_system_config(config)
```

**Custom parameters:**
```python
from embark.benchmark.agents import PIParameters

params = PIParameters(
    L_d=0.001,
    L_q=0.001,
    R_s=0.1,
    # ... other parameters
)
controller = PIControllerAgent(params=params)
```

### SNN Controller

**Step 1: Load model**
```python
from embark.benchmark.agents import SNNControllerAgent

snn = SNNControllerAgent(
    model_path="path/to/model.pt",
    device="cuda",  # or "cpu"
    track_spikes=True,  # Enable spike statistics
)
```

**Step 2: Create processors**
```python
from embark.benchmark.processors import MinMaxProcessor, LinearActionProcessor

# State processor: normalize inputs
state_proc = MinMaxProcessor(
    input_keys=["i_d", "i_q"],
    reference_keys=["i_d_ref", "i_q_ref"],
    feature_range=(-1.0, 1.0),
)

# Action processor: scale outputs
action_proc = LinearActionProcessor(
    output_keys=["v_d", "v_q"],
    bounds={
        "v_d": (-48.0, 48.0),
        "v_q": (-48.0, 48.0),
    },
)
```

**Step 3: Wrap with adapter**
```python
from embark.benchmark import TensorControllerAdapter

controller = TensorControllerAdapter(
    controller=snn,
    state_processor=state_proc,
    action_processor=action_proc,
)
controller.configure(task.physics_engine.config, task)
```

### ANN Controller

Same as SNN, but use `ANNControllerWrapper`:

```python
from embark.benchmark.controllers import ANNControllerWrapper

ann = ANNControllerWrapper(model_path="path/to/ann.pt")
# ... same processor setup as SNN
```

### Remote Controllers (Akida)

For hardware-accelerated inference:

```python
from embark.benchmark.controllers.remote import AkidaPolicy

controller = AkidaPolicy(
    server_url="http://raspberry-pi:8000",
    model_path="path/to/model.fbz",
)
# No adapter needed - implements Controller directly
```

---

## Custom Scenarios

### Define Custom Scenario

```python
from embark.benchmark.harness import ScenarioDefinition
from embark.benchmark.tasks.reference_generators import SinusoidalReference

my_scenario = ScenarioDefinition(
    name="custom_tracking",
    description="Custom sinusoidal tracking at 5Hz",
    n_rpm=1500.0,
    reference_generator=SinusoidalReference(
        i_d_ref=0.0,
        i_q_amp=3.0,
        i_q_offset=2.0,
        frequency_hz=5.0,
    ),
    max_steps=3000,
    safety_limits=SafetyLimits(max_current_a=15.0),
)

# Use in suite
suite = BenchmarkSuite(scenarios=[my_scenario])
```

### Custom Reference Generator

```python
from embark.benchmark.tasks.reference_generators import ReferenceGenerator

class MyReferenceGenerator(ReferenceGenerator):
    def __init__(self, ...):
        # Your initialization

    def __call__(self, t: float) -> dict[str, float]:
        # Return {"i_d_ref": ..., "i_q_ref": ...}
        return {"i_d_ref": 0.0, "i_q_ref": 2.0 * np.sin(2 * np.pi * t)}

# Use in task
task = PMSMCurrentControlTask(
    physics_engine=engine,
    reference_generator=MyReferenceGenerator(...),
)
```

---

## Custom Metrics

### Implement MetricAccumulator

```python
from embark.benchmark.interfaces import MetricAccumulator
from embark.benchmark.interfaces.types import StateDict, ReferenceDict, ActionDict

class MyCustomMetric(MetricAccumulator):
    def __init__(self):
        self._sum_error = 0.0
        self._count = 0

    @property
    def name(self) -> str:
        return "my_metric"

    def reset(self) -> None:
        self._sum_error = 0.0
        self._count = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,
        next_state: StateDict,
        controller_info: dict | None = None,
    ) -> None:
        # O(1) operations only!
        error = abs(reference["i_q_ref"] - next_state["i_q"])
        self._sum_error += error
        self._count += 1

    def compute(self) -> dict[str, float]:
        # Expensive operations OK here
        mean_error = self._sum_error / max(self._count, 1)
        return {"my_metric": mean_error}
```

**Important:** `update()` must be O(1). No iteration, sorting, or growing lists.

### Use Custom Metric

```python
metrics = [
    TrackingMAE(tracked_keys=["i_q"]),
    MyCustomMetric(),
]

harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
results = harness.run()
print(f"My metric: {results['my_metric']}")
```

---

## Extending the Framework

### Add New Controller Type

**Option 1: Implement Controller directly (classical)**
```python
from embark.benchmark.interfaces import Controller

class MyClassicalController(Controller):
    def reset(self) -> None:
        # Reset state
        pass

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        # Compute action
        return {"v_d": ..., "v_q": ...}

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass
```

**Option 2: Implement TensorController (neural)**
```python
from embark.benchmark.interfaces import TensorController
import torch

class MyNeuralController(TensorController):
    def reset(self) -> None:
        # Reset model state
        pass

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Forward pass
        return self.model(observation)

    def get_state(self) -> dict:
        return {"model_state": self.model.state_dict()}

    def set_state(self, state: dict) -> None:
        self.model.load_state_dict(state["model_state"])
```

### Add New Physics Engine

```python
from embark.benchmark.interfaces import PhysicsEngine, SystemConfig

class MyPhysicsEngine(PhysicsEngine):
    def __init__(self, ...):
        self._config = SystemConfig(...)

    @property
    def config(self) -> SystemConfig:
        return self._config

    def reset(self, seed: int | None = None) -> StateDict:
        # Reset to initial state
        return {"x": 0.0, "y": 0.0, ...}

    def step(self, action: ActionDict) -> tuple[StateDict, dict]:
        # Advance physics
        next_state = {...}
        info = {}
        return next_state, info

    def close(self) -> None:
        # Cleanup
        pass

    @property
    def state_keys(self) -> set[str]:
        return {"x", "y", ...}

    @property
    def action_keys(self) -> set[str]:
        return {"u1", "u2", ...}
```

### Add New Task Type

```python
from embark.benchmark.interfaces import ClosedLoopTask, PhysicsEngine

class MyTask(ClosedLoopTask):
    def __init__(self, physics_engine: PhysicsEngine, ...):
        self._physics = physics_engine
        self._max_steps = 1000

    @property
    def physics_engine(self) -> PhysicsEngine:
        return self._physics

    @property
    def reference_keys(self) -> set[str]:
        return {"ref_x", "ref_y"}

    @property
    def max_steps(self) -> int | None:
        return self._max_steps

    def reset(self, seed: int | None = None) -> tuple[StateDict, ReferenceDict]:
        state = self._physics.reset(seed)
        ref = {"ref_x": 0.0, "ref_y": 0.0}
        return state, ref

    def step(self, action: ActionDict) -> tuple[StateDict, ReferenceDict, bool]:
        next_state, info = self._physics.step(action)
        ref = self._generate_reference(...)
        done = self._check_termination(next_state)
        return next_state, ref, done
```

---

## Troubleshooting

### Controller Not Configured

**Error:** `RuntimeError: TensorControllerAdapter.configure() must be called before use.`

**Solution:**
```python
adapter.configure(task.physics_engine.config, task)
```

### Safety Violation

**Check termination reason:**
```python
if task.terminated_by_safety:
    print(f"Violation: {task.last_violation_reason}")
```

**Adjust safety limits:**
```python
task = PMSMCurrentControlTask.from_config(
    ...,
    safety_limits=SafetyLimits(
        max_current_a=25.0,  # Increase limit
        max_voltage_v=60.0,
    ),
)
```

### Metric Returns NaN

**Check for division by zero:**
```python
def compute(self) -> dict[str, float]:
    if self._count == 0:
        return {"my_metric": 0.0}
    return {"my_metric": self._sum / self._count}
```

### Processors Not Configured

**Error:** Processors need physics config for bounds.

**Solution:**
```python
state_proc.configure(task.physics_engine.config, task)
action_proc.configure(task.physics_engine.config)
```

### Memory Issues

**Reduce episode length:**
```python
task = PMSMCurrentControlTask.from_config(..., max_steps=500)
```

**Close physics engine:**
```python
task.physics_engine.close()
```

### Performance Issues

**Check metric update() complexity:**
- Must be O(1) per call
- No iteration over history
- No sorting
- Defer expensive operations to `compute()`

**Profile:**
```python
import cProfile
cProfile.run("harness.run()")
```

---

## Best Practices

1. **Always close physics engines:**
   ```python
   try:
       results = harness.run()
   finally:
       task.physics_engine.close()
   ```

2. **Use context managers for resources:**
   ```python
   # Implement if needed
   with PMSMPhysicsEngine(n_rpm=1000) as engine:
       task = PMSMCurrentControlTask(...)
       ...
   ```

3. **Validate controller before benchmarking:**
   ```python
   # Quick sanity check
   state, ref = task.reset()
   action = controller(state, ref)
   assert "v_d" in action and "v_q" in action
   ```

4. **Use quick scenarios for development:**
   ```python
   suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
   ```

5. **Save intermediate results:**
   ```python
   import json
   with open("results.json", "w") as f:
       json.dump(results, f, indent=2)
   ```

6. **Monitor safety violations:**
   ```python
   if summary.num_safety_violations > 0:
       print("WARNING: Safety violations detected!")
   ```
