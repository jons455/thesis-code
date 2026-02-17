# Benchmark Documentation Index

Complete documentation for the `embark.benchmark` module.

## Overview

The `embark.benchmark` module provides a modular closed-loop benchmark framework for evaluating neuromorphic controllers (SNNs) against classical controllers (PI) for PMSM current control. The framework is adapted from NeuroBench patterns but extended for closed-loop control scenarios.

## Documentation Structure

### Getting Started

1. **[README.md](../../embark/benchmark/README.md)** - Quick start guide with basic examples
2. **[ARCHITECTURE.md](../../embark/benchmark/ARCHITECTURE.md)** - Architecture overview and design principles
3. **[Quick Reference](BENCHMARK_QUICK_REFERENCE.md)** - Cheat sheet for common operations

### Detailed Guides

4. **[User Guide](BENCHMARK_USER_GUIDE.md)** - Comprehensive usage guide with:
   - Step-by-step tutorials
   - Controller setup (PI, SNN, ANN, Remote)
   - Custom scenarios and metrics
   - Extension patterns
   - Troubleshooting

5. **[API Reference](BENCHMARK_API.md)** - Complete API documentation:
   - All classes and methods
   - Type signatures
   - Protocol definitions
   - Examples for each component

### Specialized Documentation

6. **[Metrics Reference](../reference/METRICS.md)** - Metric definitions:
   - Control metrics (MAE, ITAE, SettlingTime, etc.)
   - Latency metrics
   - Neuromorphic metrics (SyOps, Sparsity, etc.)
   - Output formats

7. **[SNN Controller Comparison](../analysis/SNN_CONTROLLER_COMPARISON.md)** - Comparison of SNN coding schemes:
   - Population Analog Readout (v5)
   - Pulse-Based Switching (v9)
   - Input features and preprocessing requirements
   - Implementation guide for temporal state processor

8. **[Normalization Analysis](../analysis/NORMALIZATION_ANALYSIS.md)** - Analysis of normalization strategies

9. **[PWM Analysis Summary](../analysis/PWM_ANALYSIS_SUMMARY.md)** - PWM conversion and dead-time compensation

## Quick Navigation

### By Task

**I want to...**

- **Run a benchmark** → [User Guide: Quick Start](BENCHMARK_USER_GUIDE.md#quick-start)
- **Set up an SNN controller** → [User Guide: Controller Setup](BENCHMARK_USER_GUIDE.md#controller-setup)
- **Create custom metrics** → [User Guide: Custom Metrics](BENCHMARK_USER_GUIDE.md#custom-metrics)
- **Understand the architecture** → [ARCHITECTURE.md](../../embark/benchmark/ARCHITECTURE.md)
- **Look up an API** → [API Reference](BENCHMARK_API.md)
- **Find a code example** → [Quick Reference](BENCHMARK_QUICK_REFERENCE.md)

### By Component

- **Harness** → [API: ClosedLoopHarness](BENCHMARK_API.md#closedloopharness), [API: BenchmarkSuite](BENCHMARK_API.md#benchmarksuite)
- **Tasks** → [API: Tasks](BENCHMARK_API.md#tasks)
- **Controllers** → [API: Controllers](BENCHMARK_API.md#controllers)
- **Processors** → [API: Processors](BENCHMARK_API.md#processors)
- **Metrics** → [API: Metrics](BENCHMARK_API.md#metrics), [Metrics Reference](../reference/METRICS.md)
- **Physics** → [API: Physics](BENCHMARK_API.md#physics)

## Key Concepts

### Unified Controller Interface

All controllers (classical and neural) use the same `Controller` protocol:

```python
controller(state: StateDict, reference: ReferenceDict) -> ActionDict
```

- **Classical controllers** (PI) implement this directly
- **Neural controllers** (SNN/ANN) are wrapped with `TensorControllerAdapter`

### Two-Phase Safety

Safety limits are checked in two phases:
1. **Action limits** - Before physics step (prevents invalid commands)
2. **State limits** - After physics step (detects instability)

### O(1) Metric Contract

All metrics follow the `MetricAccumulator` protocol:
- `update()` - Must be O(1) per call (no iteration, sorting, or growing lists)
- `compute()` - Called once at episode end (expensive operations OK)

### Processor Chain

Neural controllers use a processor chain:
1. **State Processor** - Converts `StateDict` → `torch.Tensor` (normalization)
2. **Controller** - Forward pass (`Tensor` → `Tensor`)
3. **Action Processor** - Converts `Tensor` → `ActionDict` (denormalization, scaling)

## Common Patterns

### Pattern 1: Single Scenario

```python
task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
controller = PIControllerAgent.from_system_config(task.physics_engine.config)
harness = ClosedLoopHarness(task=task, controller=controller)
results = harness.run()
```

### Pattern 2: Multi-Scenario Suite

```python
suite = BenchmarkSuite()
summary = suite.run(controller=controller, name="MyController")
suite.print_summary(summary)
```

### Pattern 3: SNN with Processors

```python
snn = SNNControllerAgent("model.pt")
state_proc = RateSNNStateProcessor(
    include_currents=True,
    include_errors=True,
    include_speed=True,
    i_max=20.0,
)
action_proc = RateSNNActionProcessor(incremental=False, u_max=48.0)
controller = TensorControllerAdapter(snn, state_proc, action_proc)
controller.configure(task.physics_engine.config, task)
```

## Standard Scenarios

The framework includes 6 predefined scenarios (see `STANDARD_SCENARIOS`):

- `step_low_speed_500rpm_2A` - Low speed (500 RPM) step response
- `step_mid_speed_1500rpm_2A` ⭐ - **Primary reference** scenario (1500 RPM)
- `step_high_speed_2500rpm_2A` - High speed (2500 RPM) step response
- `multi_step_bidirectional_1500rpm` - Multi-step bidirectional tracking
- `four_quadrant_transition_1500rpm` - Four-quadrant transition (regenerative braking)
- `field_weakening_2500rpm` - Field-weakening operation

For detailed scenario specifications, see [BENCHMARK_SCENARIOS.md](BENCHMARK_SCENARIOS.md).  
See [User Guide: Custom Scenarios](BENCHMARK_USER_GUIDE.md#custom-scenarios) for creating custom scenarios.

## Standard Metrics

Default metrics include:

- **Tracking**: MAE, ITAE, Maximum Error
- **Dynamics**: Settling Time, Overshoot
- **Latency**: Mean, P95, P99, Max latency
- **Neuromorphic**: SyOps, Activation Sparsity, Footprint (when controller.model exists)

See [Metrics Reference](../reference/METRICS.md) for complete list.

## Extension Points

The framework is designed for extension:

- **New Controllers** - Implement `Controller` or `TensorController` protocol
- **New Tasks** - Implement `ClosedLoopTask` protocol
- **New Metrics** - Implement `MetricAccumulator` protocol
- **New Processors** - Implement `StateProcessor` or `ActionProcessor` protocol
- **New Physics Engines** - Implement `PhysicsEngine` protocol

See [User Guide: Extending the Framework](BENCHMARK_USER_GUIDE.md#extending-the-framework) for details.

## Related Documentation

> **Note:** The following documentation paths are planned but not yet available:
> - Technical Architecture documentation
> - Training Guide
> - Evaluation Pipeline documentation
> 
> For now, see the [User Guide](BENCHMARK_USER_GUIDE.md) and [API Reference](BENCHMARK_API.md) for implementation details.

## Examples

See the following for complete examples:

- **[README.md](../../embark/benchmark/README.md)** - Basic examples
- **[User Guide](BENCHMARK_USER_GUIDE.md)** - Comprehensive examples
- **[Quick Reference](BENCHMARK_QUICK_REFERENCE.md)** - Code snippets

## Support

For issues or questions:

1. Check [Troubleshooting](BENCHMARK_USER_GUIDE.md#troubleshooting)
2. Review [API Reference](BENCHMARK_API.md) for method signatures
3. Check [ARCHITECTURE.md](../../embark/benchmark/ARCHITECTURE.md) for design rationale
