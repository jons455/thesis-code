# Benchmark Suite Quick Reference

This is a quick reference guide for the EMBARK benchmark suite. For comprehensive details, see [BENCHMARK_SCENARIOS.md](BENCHMARK_SCENARIOS.md).

## The 6 Standard Scenarios

| # | Scenario ID | Speed | Current Profile | Duration | Primary Purpose |
|---|-------------|-------|-----------------|----------|-----------------|
| 1 | `step_low_speed_500rpm_2A` | 500 RPM | 0→2A i_q | 0.3s | Low-speed sensitivity |
| 2 | `step_mid_speed_1500rpm_2A` ⭐ | 1500 RPM | 0→2A i_q | 0.3s | **Primary reference** |
| 3 | `step_high_speed_2500rpm_2A` | 2500 RPM | 0→2A i_q | 0.3s | High-speed + voltage limits |
| 4 | `multi_step_bidirectional_1500rpm` | 1500 RPM | ±2A i_q (4 steps) | 1.0s | Dynamic tracking |
| 5 | `four_quadrant_transition_1500rpm` | 1500 RPM | +2→-2→0A i_q | 0.9s | Regenerative braking |
| 6 | `field_weakening_2500rpm` | 2500 RPM | i_d=-2A, i_q=2A | 0.6s | Multivariable control |

⭐ **Scenario 2 is your primary reference** - use this for headline performance numbers.

## Quick Usage

### Run Full Benchmark Suite

```python
from embark.benchmark.harness import BenchmarkSuite

suite = BenchmarkSuite()  # Uses STANDARD_SCENARIOS by default
summary = suite.run(controller=my_controller, name="MySNN-v1")
suite.print_summary(summary)
suite.save_results(summary, "results/benchmark.json")
```

### Run Quick Validation (2 scenarios)

```python
from embark.benchmark.harness import BenchmarkSuite, QUICK_SCENARIOS

suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
summary = suite.run(controller=my_controller)
```

### Run Specific Scenarios

```python
from embark.benchmark.harness import BenchmarkSuite, STANDARD_SCENARIOS

# Run only speed-range characterization (scenarios 1-3)
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS[:3])
summary = suite.run(controller=my_controller)
```

## Key Metrics Per Scenario

### Single-Step Scenarios (1-3)
- **MAE i_q**: Mean absolute tracking error
- **ITAE i_q**: Time-weighted integral error
- **Settling time**: Time to reach ±5% of reference
- **Overshoot**: Peak overshoot percentage
- **Rise time**: 10%-90% transition time

### Multi-Step Scenario (4)
- **Worst-case settling time** across 4 steps
- **Worst-case overshoot** across 4 steps
- **Consistency**: Standard deviation of settling times
- **Overall MAE i_q**

### Four-Quadrant Scenario (5)
- **Torque reversal metrics** (+2 → -2A transition)
- **Zero-crossing behavior**
- **Braking transient MAE**

### Field-Weakening Scenario (6)
- **Both i_d and i_q metrics**
- **Cross-coupling error** (i_d disturbance when i_q steps)
- **d-q decoupling quality**

## Performance Targets

### Good SNN Performance
- ✅ MAE i_q within **10%** of PI baseline
- ✅ Settling time within **20%** of PI
- ✅ Overshoot < **10%**
- ✅ **Zero** safety violations

### Acceptable Tradeoffs
- Overshoot up to **15%** if MAE is better
- Settling up to **30%** longer if steady-state tracking is better

### Red Flags
- ❌ Any safety violations
- ❌ MAE > **2x** PI baseline
- ❌ Settling > **2x** PI baseline
- ❌ Failure on Scenario 2 (nominal conditions)

## Coverage Summary

✅ **Speed range**: 500, 1500, 2500 RPM  
✅ **Transients**: Single-step, multi-step, reversal  
✅ **Quadrants**: Motoring, generating, zero-crossing  
✅ **Advanced**: Field-weakening, d-q coupling  
✅ **Runtime**: ~5-10 seconds total

## Custom Scenarios

### Create a Custom Scenario

```python
from embark.benchmark.harness import ScenarioDefinition
from embark.benchmark.tasks.reference_generators import MultiStepReference

custom = ScenarioDefinition(
    name="my_custom_scenario",
    description="Custom test for my specific use case",
    n_rpm=1800.0,
    reference_generator=MultiStepReference(
        steps=[
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 3.0),
            (0.3, -1.0, 3.0),
        ]
    ),
    max_steps=5000,
)

# Run with custom + standard scenarios
suite = BenchmarkSuite(scenarios=[custom] + STANDARD_SCENARIOS)
```

### Available Reference Generators

```python
from embark.benchmark.tasks.reference_generators import (
    StepReference,           # Single step
    ConstantReference,       # Constant value
    SinusoidalReference,     # Sinusoidal tracking
    MultiStepReference,      # Multiple step transitions
)

# Step reference (0 → 2A at t=0)
StepReference(i_d_ref=0.0, i_q_ref=2.0)

# Constant reference
ConstantReference(i_d_ref=0.0, i_q_ref=3.0)

# Sinusoidal (1A amplitude, 2Hz, offset at 2A)
SinusoidalReference(i_d_ref=0.0, i_q_amp=1.0, i_q_offset=2.0, frequency_hz=2.0)

# Multi-step (list of (time, i_d, i_q) tuples)
MultiStepReference(steps=[
    (0.0, 0.0, 0.0),
    (0.1, 0.0, 2.0),
    (0.3, -1.0, 2.0),
])
```

## Reporting Checklist

When publishing or comparing controllers:

- [ ] Report **Scenario 2** (primary reference) prominently
- [ ] Include **all 6 scenarios** for comprehensive evaluation
- [ ] Compare with **PI baseline** on same scenarios
- [ ] Report **safety violation count**
- [ ] Include **aggregate metrics** (mean MAE, worst-case error)
- [ ] If stochastic: report **mean ± std** across multiple runs
- [ ] Save **raw results JSON** for reproducibility

## Development Workflow

### 1. Initial Development
```python
# Use quick scenarios for fast iteration
suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
```

### 2. Tuning
```python
# Focus on Scenario 2 (primary reference)
suite = BenchmarkSuite(scenarios=[STANDARD_SCENARIOS[1]])
```

### 3. Validation
```python
# Run full suite
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)
```

### 4. Publication
```python
# Run multiple seeds (if stochastic)
results = []
for seed in range(5):
    set_random_seed(seed)
    summary = suite.run(controller=my_controller)
    results.append(summary)
    
# Compute mean ± std
```

## Interpretation Tips

### Scenario-Specific Insights

**Scenario 1 worse than 2?** → Low-speed sensitivity issues  
**Scenario 3 worse than 2?** → Voltage saturation or back-EMF issues  
**Scenario 4 inconsistent?** → Memory effects or state accumulation  
**Scenario 5 zero-crossing spike?** → Deadtime compensation needed  
**Scenario 6 i_d affects i_q?** → Poor d-q decoupling  

### Common Failure Patterns

| Symptom | Likely Cause |
|---------|--------------|
| High overshoot (>15%) | Aggressive controller gains |
| Slow settling (>50ms) | Conservative gains or saturation |
| High MAE in Scenario 4 | Poor dynamic tracking |
| Failure in Scenario 5 | Zero-crossing handling issues |
| Failure in Scenario 6 | Lack of d-q decoupling |

## See Also

- **[BENCHMARK_SCENARIOS.md](BENCHMARK_SCENARIOS.md)** - Comprehensive scenario guide
- **[BENCHMARK_API.md](BENCHMARK_API.md)** - Complete API reference
- **[RATE_SNN_BENCHMARK_INTERFACE.md](RATE_SNN_BENCHMARK_INTERFACE.md)** - SNN-specific guide
