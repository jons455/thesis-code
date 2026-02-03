# NeuroBench-Aligned Architectural Refactoring

## Overview

This document outlines the refactoring plan to align the `embark` benchmarking framework with NeuroBench's modular harness architecture. The goal is to create a clean, extensible framework for benchmarking neuromorphic controllers on closed-loop control tasks.

---

## I. Architectural Philosophy

### From Gym to NeuroBench

The original design followed a Gym-style "monolithic environment" pattern where `PMSMEnv` handled physics, normalization, reference generation, and reward computation. The new design follows NeuroBench's "modular harness" pattern where each component has a single responsibility.

| Old Pattern (Gym) | New Pattern (NeuroBench) |
|-------------------|--------------------------|
| `PMSMEnv` does everything | Components are independent |
| Wrappers hide processing | Processors are first-class |
| Metrics computed post-hoc | Accumulators observe in real-time |
| RL-focused (reward-centric) | Benchmarking-focused (metric-centric) |

### Component Mapping

| NeuroBench Component | Our Equivalent | Responsibility |
|---------------------|----------------|----------------|
| **Model** | `ControllerPolicy` | SNN, ANN, or classical controller (PI/PID) |
| **Benchmark (Task)** | `ClosedLoopTask` | Reference trajectory + Physics composition |
| **Processors** | `StateProcessor` / `ActionProcessor` | Unit conversion, normalization, spike encoding |
| **Accumulators** | `MetricAccumulator` | Stateful metric computation (RMSE, SyOps) |
| **Harness** | `ClosedLoopHarness` | Orchestrates the control loop |

---

## II. Core Design Decisions

### Decision 1: Observation Format
**Verdict: Domain-Specific Keys (Option B)**

- Use semantic keys (`i_q`, `theta_gimbal`) in Physics/Task
- Use `MetricRegistry` to map these to generic concepts for cross-system comparison
- Example: `Registry.register(metric="tracking_error", key="i_q_error")`

### Decision 2: Baseline Auto-Tuning
**Verdict: Required with Escape Hatch**

- Protocol enforces `tune(config)` or `from_system_config()` factory
- Provide `ManualTuner` class for explicit parameter override
- Default: Technical Optimum for PI controllers

### Decision 3: Backward Compatibility
**Verdict: Full Migration (Clean Break)**

- No `LegacyPMSMEnv` wrapper
- One-time `migrate_checkpoint.py` script for old SNN weights
- Clean runtime code without legacy support

### Decision 4: Metrics Normalization
**Verdict: Store Both Raw and Normalized**

- Raw totals for debugging specific episodes
- Time-normalized (per-second) for cross-system comparison
- Storage is cheap; losing data is permanent

### Decision 5: Action Space Units
**Verdict: Physical Units Always**

- `PhysicsEngine` accepts physical units (Volts, Newtons)
- `ActionProcessor` handles normalization ([-1, 1] → [-24V, +24V])
- Keeps physics engine "pure" and realistic

### Decision 6: Coordinate Transforms
**Verdict: Keep in Adapter or Controller**

- Generic interface uses "native" coordinates of the system
- PMSM adapter accepts `v_alpha`, `v_beta` (Clarke frame)
- If controller works in d-q frame, it handles Park transform internally

### Decision 7: Stateful Controllers
**Verdict: Strictly Require `get_state()` / `set_state()`**

- Essential for reproducible benchmarks
- Enables replay of specific failure scenarios
- Required for checkpoint-based evaluation

### Decision 8: Neuromorphic Metrics Location
**Verdict: Controller-Specific, Not Physics-Specific**

- SyOps is a cost of the agent's compute, not physical motion
- Physics engine doesn't know about spikes
- Metrics accumulator observes controller internals if needed

---

## II.a NeuroBench Alignment Strategy

### Goal
Align the refactor with the NeuroBench architecture (models, processors, metrics, hooks),
while extending it for closed-loop control tasks.

### What We Reuse from NeuroBench
- `NeuroBenchModel` wrappers for SNN/ANN controllers
- Hook-based metrics (e.g., `SynapticOperations`, `ActivationSparsity`)
- Processor manager pattern (pre-/post-processing)

### What We Add (Closed-Loop Extensions)
- `ClosedLoopBenchmark` (NeuroBench-style `Benchmark` but with physics stepping)
- `ClosedLoopTask` + `PhysicsEngine` interfaces (control-specific)
- Control metrics accumulators (tracking, dynamics, safety, efficiency)

### Mapping to NeuroBench Modules
| NeuroBench Package | Our Extension |
|--------------------|---------------|
| `neurobench.models` | `embark/benchmark/models` |
| `neurobench.benchmarks` | `embark/benchmark/benchmarks/closed_loop.py` |
| `neurobench.preprocessing` | `embark/benchmark/preprocessing` |
| `neurobench.postprocessing` | `embark/benchmark/postprocessing` |
| `neurobench.metrics` | `embark/benchmark/metrics/control` |

### Expected Outcome
- SNN controllers automatically expose SyOps/sparsity through NeuroBench hooks
- Closed-loop evaluation matches NeuroBench metric semantics
- Architecture remains aligned with upstream NeuroBench (2025_GC branch)

## III. Protocol Definitions

### 1. PhysicsEngine Protocol

The physics engine represents pure dynamics. No rewards, no references, no normalization.

```python
from typing import Protocol, Any

class PhysicsEngine(Protocol):
    """Abstract interface for physical dynamical systems."""

    @property
    def config(self) -> "SystemConfig":
        """Immutable physical properties (R, L, J, friction, limits)."""
        ...

    def reset(self, seed: int | None = None) -> dict[str, float]:
        """Reset to initial state. Returns initial state dict."""
        ...

    def step(self, action: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        """
        Execute one physics step.

        Args:
            action: Physical units (e.g., {"v_alpha": 12.0, "v_beta": -5.0} in Volts)

        Returns:
            (next_state, debug_info)
            - next_state: Physical state dict (e.g., {"i_d": 1.2, "i_q": 5.0, ...})
            - debug_info: Optional diagnostics (e.g., {"solver_steps": 3})
        """
        ...

    def close(self) -> None:
        """Clean up resources (simulator handles, etc.)."""
        ...

    @property
    def state_keys(self) -> set[str]:
        """Keys present in state dict."""
        ...

    @property
    def action_keys(self) -> set[str]:
        """Keys expected in action dict."""
        ...
```

### 2. ClosedLoopTask Protocol

The task defines the goal. It composes a physics engine and generates references.

```python
class ClosedLoopTask(Protocol):
    """Defines the control objective. Owns a PhysicsEngine."""

    @property
    def physics_engine(self) -> PhysicsEngine:
        """The underlying dynamical system."""
        ...

    @property
    def reference_keys(self) -> set[str]:
        """Keys provided in reference dict (e.g., {'i_q_ref', 'i_d_ref'})."""
        ...

    @property
    def max_steps(self) -> int | None:
        """Maximum episode length (None for infinite)."""
        ...

    def reset(self, seed: int | None = None) -> tuple[dict[str, float], dict[str, float]]:
        """
        Reset task and physics.

        Returns:
            (initial_state, initial_reference)
        """
        ...

    def step(self, action: dict[str, float]) -> tuple[dict[str, float], dict[str, float], bool]:
        """
        Step physics and update reference.

        Returns:
            (next_state, next_reference, done)
        """
        ...
```

### 3. ControllerPolicy Protocol

Controllers can be tensor-based (neural) or dict-based (classical).

```python
import torch

class TensorController(Protocol):
    """Neural network controllers (SNN, ANN)."""

    def reset(self) -> None:
        """Reset internal state (membrane potentials, hidden states)."""
        ...

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute action from observation tensor."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        ...


class DictController(Protocol):
    """Classical controllers (PI, PID, MPC)."""

    def reset(self) -> None:
        """Reset internal state (integrator windup, etc.)."""
        ...

    def __call__(
        self,
        state: dict[str, float],
        reference: dict[str, float]
    ) -> dict[str, float]:
        """Compute action from state and reference."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state."""
        ...

    @classmethod
    def from_system_config(
        cls,
        config: "SystemConfig",
        tuning: str = "technical_optimum"
    ) -> "DictController":
        """Factory method for auto-tuning."""
        ...
```

### 4. Processor Protocols

Processors handle the conversion between physics (dicts) and controllers (tensors).

```python
class StateProcessor(Protocol):
    """Converts physics state dict → controller observation tensor."""

    def configure(self, physics_config: "SystemConfig", task: ClosedLoopTask) -> None:
        """Called once at harness setup to learn normalization bounds."""
        ...

    def __call__(
        self,
        state: dict[str, float],
        reference: dict[str, float]
    ) -> torch.Tensor:
        """Process state and reference into observation tensor."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of output tensor."""
        ...


class ActionProcessor(Protocol):
    """Converts controller action tensor → physics action dict."""

    def configure(self, physics_config: "SystemConfig") -> None:
        """Called once at harness setup to learn action bounds."""
        ...

    def __call__(
        self,
        action: torch.Tensor,
        physics_config: "SystemConfig"
    ) -> dict[str, float]:
        """Convert action tensor to physical units."""
        ...
```

### 5. MetricAccumulator Protocol

Metrics observe the raw control loop data and compute statistics.

```python
class MetricAccumulator(Protocol):
    """Stateful metric that observes the control loop."""

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    def reset(self) -> None:
        """Reset accumulated state."""
        ...

    def update(
        self,
        state: dict[str, float],
        reference: dict[str, float],
        action: dict[str, float],
        next_state: dict[str, float],
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Update metric with one timestep of data.

        Args:
            state: Current physical state
            reference: Current reference
            action: Action taken (physical units)
            next_state: Resulting state
            controller_info: Optional controller internals (spikes, etc.)
        """
        ...

    def compute(self) -> float | dict[str, float]:
        """Compute final metric value(s)."""
        ...
```

### 6. ClosedLoopHarness

The main orchestrator that runs the benchmark.

```python
class ClosedLoopHarness:
    """NeuroBench-style harness for closed-loop control benchmarks."""

    def __init__(
        self,
        task: ClosedLoopTask,
        controller: TensorController | DictController,
        state_processor: StateProcessor | None = None,
        action_processor: ActionProcessor | None = None,
        metrics: list[MetricAccumulator] | None = None,
    ):
        self.task = task
        self.controller = controller
        self.state_proc = state_processor
        self.action_proc = action_processor
        self.metrics = metrics or []

        # Detect controller type
        self._uses_tensors = hasattr(controller, 'forward')

        # Configure processors if present
        if self.state_proc:
            self.state_proc.configure(task.physics_engine.config, task)
        if self.action_proc:
            self.action_proc.configure(task.physics_engine.config)

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """
        Run one episode of the benchmark.

        Returns:
            Dictionary of metric results
        """
        state, reference = self.task.reset()
        self.controller.reset()
        for m in self.metrics:
            m.reset()

        effective_max = max_steps or self.task.max_steps or float('inf')
        step = 0
        done = False

        while not done and step < effective_max:
            # Compute action
            if self._uses_tensors:
                obs = self.state_proc(state, reference)
                action_tensor = self.controller.forward(obs)
                action = self.action_proc(action_tensor, self.task.physics_engine.config)
                controller_info = getattr(self.controller, 'last_info', None)
            else:
                action = self.controller(state, reference)
                controller_info = None

            # Step physics
            next_state, next_ref, done = self.task.step(action)

            # Update metrics (observe raw truth)
            for m in self.metrics:
                m.update(state, reference, action, next_state, controller_info)

            state, reference = next_state, next_ref
            step += 1

        # Compute final metrics
        results = {"steps": step}
        for m in self.metrics:
            result = m.compute()
            if isinstance(result, dict):
                results.update(result)
            else:
                results[m.name] = result

        return results
```

---

## IV. Directory Structure

```
embark/
├── benchmark/
│   ├── __init__.py
│   │
│   ├── interfaces/                    # Protocol definitions
│   │   ├── __init__.py
│   │   ├── physics.py                 # PhysicsEngine protocol
│   │   ├── task.py                    # ClosedLoopTask protocol
│   │   ├── controller.py              # TensorController, DictController protocols
│   │   ├── processors.py              # StateProcessor, ActionProcessor protocols
│   │   ├── metrics.py                 # MetricAccumulator protocol
│   │   └── types.py                   # SystemConfig, type aliases
│   │
│   ├── harness/                       # The central orchestrator
│   │   ├── __init__.py
│   │   ├── closed_loop.py             # ClosedLoopHarness
│   │   └── hooks.py                   # Logging, visualization hooks (optional)
│   │
│   ├── physics/                       # Physics engine implementations
│   │   ├── __init__.py
│   │   ├── pmsm.py                    # PMSMPhysicsEngine (wraps GEM)
│   │   └── config.py                  # PMSMConfig dataclass
│   │
│   ├── tasks/                         # Task implementations
│   │   ├── __init__.py
│   │   ├── pmsm_current_control.py    # PMSMCurrentControlTask
│   │   └── reference_generators.py   # StepReference, SinusoidalReference, etc.
│   │
│   ├── processors/                    # Processor implementations
│   │   ├── __init__.py
│   │   ├── normalizers.py             # StandardScalerProcessor, MinMaxProcessor
│   │   ├── encoders.py                # RateEncoder, LatencyEncoder (for SNN)
│   │   ├── decoders.py                # PopulationDecoder, etc.
│   │   └── identity.py                # IdentityProcessor (passthrough)
│   │
│   ├── controllers/                   # Controller implementations
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── pi.py                  # PIController
│   │   │   ├── pid.py                 # PIDController
│   │   │   └── tuning.py              # TechnicalOptimum, ZieglerNichols, ManualTuner
│   │   └── neural/
│   │       ├── __init__.py
│   │       ├── snn_wrapper.py         # Wraps SNN models as TensorController
│   │       └── ann_wrapper.py         # Wraps ANN models as TensorController
│   │
│   ├── metrics/                       # Metric accumulators
│   │   ├── __init__.py
│   │   ├── accumulators/
│   │   │   ├── __init__.py
│   │   │   ├── tracking.py            # TrackingRMSE, TrackingMAE, ITAE
│   │   │   ├── dynamics.py            # SettlingTime, Overshoot, TotalVariation
│   │   │   ├── efficiency.py          # ControlEffort, EnergyConsumption
│   │   │   └── neuromorphic.py        # SyOps, SpikeCount, Sparsity
│   │   └── registry.py                # MetricRegistry for task-specific mappings
│   │
│   ├── agents.py                      # DEPRECATED: Keep SNN agent wrappers temporarily
│   ├── pmsm_env.py                    # DEPRECATED: Remove after migration
│   └── run_benchmark.py               # CLI entry point (uses harness)
│
├── utils/
│   ├── __init__.py
│   ├── config.py                      # Global configuration
│   ├── paths.py                       # Path utilities
│   ├── reproducibility.py             # Seeding, determinism
│   └── validation.py                  # Protocol compliance validation
│
└── scripts/
    ├── migrate_checkpoint.py          # One-time migration for old SNN weights
    └── generate_training_data.py      # Uses new harness for data generation
```

---

## V. Migration Plan

### Phase 0: NeuroBench Alignment (Day 0)
- [ ] Add NeuroBench as dependency (pin to 2025_GC branch if required)
- [ ] Create `embark/benchmark/models/` with `NeuroBenchModel` wrappers
- [ ] Verify NeuroBench hooks work with SNN controller for SyOps/sparsity

### Phase 1: Interface Definition (Day 1)
- [ ] Create `embark/benchmark/interfaces/` package
- [ ] Define all protocol classes with full type hints
- [ ] Define `SystemConfig` and type aliases
- [ ] Write protocol compliance tests

### Phase 2: Harness Implementation (Day 1-2)
- [ ] Implement `ClosedLoopBenchmark` (NeuroBench-style run loop)
- [ ] Implement `IdentityProcessor` (passthrough for dict controllers)
- [ ] Write harness unit tests

### Phase 3: PMSM Migration (Day 2-3)
- [ ] Create `PMSMPhysicsEngine` (extract from `pmsm_env.py`)
- [ ] Create `PMSMCurrentControlTask` (extract reference logic)
- [ ] Create `PMSMConfig` dataclass
- [ ] Migrate `PIController` to new structure
- [ ] Verify existing benchmark results match

### Phase 4: Processors (Day 3-4)
- [ ] Implement `StandardScalerProcessor`
- [ ] Implement `MinMaxProcessor`
- [ ] Implement basic spike encoders (if needed for SNN)
- [ ] Write processor tests

### Phase 5: Metrics Migration (Day 4-5)
- [ ] Refactor existing metrics to accumulator pattern
- [ ] Create `TrackingRMSEAccumulator`
- [ ] Create `SyOpsAccumulator`
- [ ] Implement `MetricRegistry`
- [ ] Verify metric values match pre-refactor

### Phase 6: SNN Integration (Day 5-6)
- [ ] Create `SNNControllerWrapper` (NeuroBenchModel wrapper)
- [ ] Ensure spike statistics flow to neuromorphic metrics
- [ ] Test full SNN benchmark pipeline

### Phase 7: Documentation & Cleanup (Day 6-7)
- [ ] Write "How to Add a New Physics Engine" guide
- [ ] Write "How to Implement a Controller" guide
- [ ] Remove deprecated code (`pmsm_env.py`, old `agents.py`)
- [ ] Update `ARCHITECTURE.md`

---

## VI. Open Questions & Considerations

### 1. GEM Simulator Integration
**Status: Needs Investigation**

Current `pmsm_env.py` uses GEM's `ElectricMotorEnvironment`. Questions:
- Does GEM expose raw physics stepping without Gym wrapper?
- If not, we may need a thin adapter that wraps GEM's Gym env but extracts raw state

### 2. Reference Generator Flexibility
**Status: Design Decision Needed**

Should reference generators be:
- **Embedded in Task**: `PMSMCurrentControlTask` always uses step response
- **Composable**: `PMSMCurrentControlTask(reference_gen=StepReference(...))`

**Recommendation**: Composable, for flexibility in experiments.

### 3. Multi-Objective Metrics
**Status: Design Decision Needed**

Some metrics are naturally multi-valued (e.g., RMSE per axis). Options:
- Return `dict[str, float]` from `compute()`
- Create separate accumulators per axis
- Use hierarchical naming (`rmse.i_q`, `rmse.i_d`)

**Recommendation**: Return dict, use dot notation for hierarchy.

### 4. Real-Time Hooks
**Status: Optional Enhancement**

For debugging and visualization, consider:
- `HarnessHook` protocol with `on_step()`, `on_reset()`, `on_done()`
- Logging hook, plotting hook, etc.

**Recommendation**: Implement after core refactoring.

### 5. Batched Evaluation
**Status: Future Consideration**

For neural controllers, batched inference is faster. Consider:
- `BatchedHarness` that runs N episodes in parallel
- Requires vectorized physics (may not be feasible with GEM)

**Recommendation**: Defer to future work.

### 6. Controller Internal Observability
**Status: Needs Design**

For neuromorphic metrics (SyOps, spike counts), we need access to controller internals. Options:
- Controller exposes `last_info` dict after each `forward()`
- Metrics hook directly into controller (tight coupling)
- Controller has optional `get_inference_stats()` method

**Recommendation**: Use `last_info` pattern for loose coupling.

---

## VII. Example Usage

### Running a Benchmark with PI Controller

```python
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.tasks import PMSMCurrentControlTask
from embark.benchmark.controllers.classical import PIController
from embark.benchmark.metrics.accumulators import TrackingRMSE, SettlingTime

# Create components
task = PMSMCurrentControlTask.from_config("configs/pmsm_default.yaml")
controller = PIController.from_system_config(task.physics_engine.config)

# Define metrics
metrics = [
    TrackingRMSE(tracked_keys=["i_q", "i_d"]),
    SettlingTime(tracked_key="i_q", threshold=0.02),
]

# Run benchmark
harness = ClosedLoopHarness(
    task=task,
    controller=controller,
    metrics=metrics,
)

results = harness.run()
print(results)
# {'steps': 10000, 'rmse_i_q': 0.23, 'rmse_i_d': 0.15, 'settling_time_i_q': 0.0042}
```

### Running a Benchmark with SNN Controller

```python
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.tasks import PMSMCurrentControlTask
from embark.benchmark.controllers.neural import SNNControllerWrapper
from embark.benchmark.processors import StandardScalerProcessor, LinearActionProcessor
from embark.benchmark.metrics.accumulators import TrackingRMSE, SyOpsAccumulator

# Create components
task = PMSMCurrentControlTask.from_config("configs/pmsm_default.yaml")

# Load trained SNN
snn_model = load_snn_checkpoint("checkpoints/snn_best.pt")
controller = SNNControllerWrapper(snn_model)

# Processors for neural controller
state_proc = StandardScalerProcessor(
    input_keys=["i_d", "i_q", "e_d", "e_q"],  # state keys to use
    reference_keys=["i_d_ref", "i_q_ref"],     # reference keys to append
)
action_proc = LinearActionProcessor(
    output_keys=["v_alpha", "v_beta"],
    bounds={"v_alpha": (-24, 24), "v_beta": (-24, 24)},
)

# Metrics including neuromorphic
metrics = [
    TrackingRMSE(tracked_keys=["i_q", "i_d"]),
    SyOpsAccumulator(),  # reads from controller.last_info
]

# Run benchmark
harness = ClosedLoopHarness(
    task=task,
    controller=controller,
    state_processor=state_proc,
    action_processor=action_proc,
    metrics=metrics,
)

results = harness.run()
print(results)
# {'steps': 10000, 'rmse_i_q': 0.31, 'rmse_i_d': 0.22, 'total_syops': 1234567, 'syops_per_step': 123.4}
```

---

## VIII. File Change Summary

| Category | New Files | Modified Files | Deleted Files |
|----------|-----------|----------------|---------------|
| Interfaces | 6 | 0 | 0 |
| Harness | 2 | 0 | 0 |
| Physics | 2 | 0 | 0 |
| Tasks | 2 | 0 | 0 |
| Processors | 5 | 0 | 0 |
| Controllers | 5 | 0 | 0 |
| Metrics | 6 | 0 | 0 |
| Utils | 1 | 0 | 0 |
| Scripts | 1 | 1 | 0 |
| Deprecated | 0 | 0 | 2 (pmsm_env.py, old agents.py) |
| **TOTAL** | **30** | **1** | **2** |

---

## IX. Success Criteria

The refactoring is complete when:

1. **Functional Parity**: Same benchmark results as pre-refactor (within numerical tolerance)
2. **Protocol Compliance**: All implementations pass protocol validation tests
3. **SNN Pipeline Works**: Full SNN training → benchmark → metrics pipeline functional
4. **Documentation Complete**: Extension guides written and reviewed
5. **Tests Pass**: All existing tests pass + new protocol tests

---

## X. Next Steps

1. **Approve this plan** and resolve open questions
2. **Create interface package** with protocol definitions
3. **Implement harness** with basic integration test
4. **Migrate PMSM** incrementally with comparison tests
5. **Migrate metrics** to accumulator pattern
6. **Integrate SNN** and verify neuromorphic metrics
7. **Clean up** deprecated code and update documentation
