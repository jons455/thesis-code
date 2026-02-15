# Benchmark Scenarios Guide

This document provides detailed information about the **6 optimal benchmark scenarios** for PMSM current control evaluation, designed based on motor control benchmarking best practices and research.

## Overview

The EMBARK benchmark suite uses 6 carefully selected scenarios that provide **minimum necessary coverage** to comprehensively evaluate controller performance across the full operating envelope. These scenarios answer three key questions:

1. **Does the controller match PI baseline at steady state?**
2. **Can it handle dynamics and transients?**
3. **Does it work across the full operating range?**

---

## Design Philosophy

### Why These 6 Scenarios?

Based on analysis of motor control benchmarking literature [[1]](https://arxiv.org/html/2512.06603v1) [[2]](https://arxiv.org/html/2402.01782v1) and PMSM control best practices [[3]](https://www.nature.com/articles/s41598-025-02396-y), this suite was optimized for:

- **Comprehensive coverage** with minimal redundancy
- **Fixed-speed operation** (constant RPM per scenario)
- **Current control focus** (FOC inner loop, d-q currents)
- **Rate-based SNN evaluation** vs PI baseline
- **Manageable runtime** for iterative development

### What Was Dropped and Why

Compared to exhaustive test suites, we intentionally **excluded**:

- **Zero-crossing only test**: Redundant with Scenario 5 (four-quadrant), which includes zero-crossing at higher magnitude
- **Sinusoidal tracking test**: Lower priority for initial benchmarking; step response establishes fundamental performance
- **Multiple load levels at same speed**: Speed variation (500/1500/2500 RPM) provides better coverage than load variation at fixed speed

---

## Scenario Specifications

### Scenario 1: Single-Step Low Speed

**ID:** `step_low_speed_500rpm_2A`

**Configuration:**
- Speed: 500 RPM
- Reference: 0 → 2A i_q at t=0
- i_d: 0A (constant)
- Duration: 0.3s (3000 steps @ 100µs)

**Purpose:**
Tests fundamental step response at **challenging low-speed conditions** where:
- Controller sensitivity to parameter variations is highest
- Delays and quantization effects are most visible
- Low back-EMF makes voltage saturation less likely, isolating controller dynamics

**Key Metrics:**
- Settling time (primary)
- Overshoot (primary)
- Rise time
- MAE and ITAE during transient

**What This Reveals:**
- Baseline transient performance
- Low-speed robustness
- Parameter sensitivity

---

### Scenario 2: Single-Step Mid Speed ⭐

**ID:** `step_mid_speed_1500rpm_2A`

**Configuration:**
- Speed: 1500 RPM
- Reference: 0 → 2A i_q at t=0
- i_d: 0A (constant)
- Duration: 0.3s (3000 steps @ 100µs)

**Purpose:**
This is the **PRIMARY REFERENCE SCENARIO** for all detailed comparisons [[1]](https://arxiv.org/html/2512.06603v1). Mid-range operation (1500 RPM) represents:
- Most common operating point in applications
- Nominal conditions for direct SNN vs PI comparison
- "Ground truth" for settling time and overshoot benchmarks

**Key Metrics:**
- Settling time (reference value)
- Overshoot (reference value)
- MAE i_q (steady-state performance)
- ITAE i_q (transient performance)

**What This Reveals:**
- Nominal transient response
- Direct apples-to-apples comparison with PI baseline
- Controller tuning quality

**Usage:**
Use this scenario's results as the **baseline** when reporting controller performance. All other scenarios test edge cases and operating range.

---

### Scenario 3: Single-Step High Speed

**ID:** `step_high_speed_2500rpm_2A`

**Configuration:**
- Speed: 2500 RPM
- Reference: 0 → 2A i_q at t=0
- i_d: 0A (constant)
- Duration: 0.3s (3000 steps @ 100µs)

**Purpose:**
Tests performance at **high speed** where:
- Voltage limits become active (high back-EMF)
- Speed-dependent dynamics emerge
- Controller must handle faster electrical dynamics

**Key Metrics:**
- Settling time
- Overshoot
- Maximum voltage utilization
- Voltage saturation events

**What This Reveals:**
- High-speed performance
- Voltage limit handling
- Speed-dependent controller behavior

**Coverage:**
Together with Scenarios 1 and 2, provides **complete speed-range characterization** (500/1500/2500 RPM).

---

### Scenario 4: Multi-Step Bidirectional

**ID:** `multi_step_bidirectional_1500rpm`

**Configuration:**
- Speed: 1500 RPM (constant)
- Reference sequence:
  - t=0.0s: 0A
  - t=0.1s: +2A (motoring)
  - t=0.35s: -2A (generating)
  - t=0.6s: +2A (motoring)
  - t=0.85s: -2A (generating)
- Duration: 1.0s (10000 steps @ 100µs)

**Purpose:**
Tests **dynamic tracking and consistency** across multiple transients [[2]](https://arxiv.org/html/2402.01782v1):
- Covers both motoring (+) and generating (-) quadrants
- Reveals if SNN exhibits memory effects or performance degradation
- Most representative of real-world varying torque demands

**Key Metrics:**
- **Worst-case** settling time across 4 steps
- **Worst-case** overshoot across 4 steps
- Settling time consistency (std dev across steps)
- MAE i_q (overall tracking)

**What This Reveals:**
- Dynamic tracking capability
- Consistency across consecutive transients
- Memory effects in stateful controllers (SNNs)
- Robustness to sign changes

---

### Scenario 5: Four-Quadrant Transition

**ID:** `four_quadrant_transition_1500rpm`

**Configuration:**
- Speed: 1500 RPM (constant)
- Reference sequence:
  - t=0.0s: 0A
  - t=0.1s: +2A (motoring)
  - t=0.4s: -2A (regenerative braking)
  - t=0.7s: 0A (zero crossing)
- Duration: 0.9s (9000 steps @ 100µs)

**Purpose:**
**Critical for regenerative braking validation** [[3]](https://www.nature.com/articles/s41598-025-02396-y):
- Tests full torque reversal (hardest transient: +2A → -2A)
- Zero-crossing behavior reveals deadtime effects
- Essential if application includes braking/energy recovery

**Key Metrics:**
- Settling time for torque reversal (+2 → -2)
- Overshoot during reversal
- Zero-crossing behavior (deadtime effects)
- MAE during braking transient

**What This Reveals:**
- Torque reversal capability
- Zero-crossing handling
- Regenerative braking performance
- Low-signal controller behavior (near zero)

---

### Scenario 6: Field-Weakening

**ID:** `field_weakening_2500rpm`

**Configuration:**
- Speed: 2500 RPM (constant)
- Reference sequence:
  - t=0.0s: i_d=0A, i_q=0A
  - t=0.1s: i_d=-2A, i_q=0A (apply field weakening)
  - t=0.35s: i_d=-2A, i_q=2A (torque with field weakening)
- Duration: 0.6s (6000 steps @ 100µs)

**Purpose:**
Only scenario testing **d-q coupling/decoupling** [[4]](https://www.nature.com/articles/s41598-025-19384-x):
- Validates multivariable control capability (both axes active)
- High-speed + non-zero i_d is most challenging operating condition
- Separates advanced controllers from basic ones

**Key Metrics:**
- i_d settling time (field-weakening activation)
- i_q settling time (with active i_d)
- Cross-coupling error (i_d disturbance when i_q steps)
- MAE for both i_d and i_q

**What This Reveals:**
- Multivariable control capability
- d-q decoupling effectiveness
- Field-weakening operation (extends speed range)
- Voltage saturation handling at high speed

**Note:**
Many basic controllers fail this scenario due to poor d-q decoupling. Success here indicates **advanced control capability**.

---

## Coverage Matrix

| Property | Scenario(s) | Explanation |
|----------|-------------|-------------|
| **Low-speed performance** | 1 | 500 RPM tests low back-EMF, high sensitivity |
| **Nominal performance** | 2 ⭐ | 1500 RPM is reference operating point |
| **High-speed performance** | 3, 6 | 2500 RPM tests voltage limits, back-EMF |
| **Transient response** | 1-3 | Single steps characterize rise/settling/overshoot |
| **Dynamic tracking** | 4 | Multiple consecutive steps test consistency |
| **Motoring & generating** | 4, 5 | Both positive and negative torque |
| **Torque reversal** | 5 | +2A → -2A is hardest transient |
| **Zero-crossing** | 5 | Tests deadtime effects, low-signal behavior |
| **d-q decoupling** | 6 | Only scenario with active i_d ≠ 0 |
| **Voltage saturation** | 3, 6 | High speed + high current/field-weakening |

---

## Implementation Details

### Sampling Time

All scenarios use **100 µs sampling time** (10 kHz control frequency), which is:
- Standard for PMSM benchmarking [[1]](https://arxiv.org/html/2512.06603v1)
- Typical for real-time motor control implementations
- Sufficient for current loop bandwidth (≤1 kHz)

### Scenario Selection

**Full benchmark run** (6 scenarios):
```python
from embark.benchmark.harness import BenchmarkSuite, STANDARD_SCENARIOS

suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)
summary = suite.run(controller=my_controller)
```

**Quick validation** (2 scenarios):
```python
from embark.benchmark.harness import QUICK_SCENARIOS

suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
summary = suite.run(controller=my_controller)
```

**Custom subset**:
```python
# Run only speed-range characterization (scenarios 1-3)
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS[:3])
```

### Metrics Computation

Each scenario computes:

**For all scenarios:**
- MAE i_q: Mean absolute error in q-axis current
- ITAE i_q: Integral of time-weighted absolute error
- MaxError i_q: Worst-case instantaneous error

**For step scenarios (1-3, and transitions in 4-6):**
- Settling time (5% threshold)
- Overshoot (% above reference)
- Rise time (10%-90%)

**For multi-step scenarios (4):**
- Worst-case metrics across all steps
- Consistency metrics (std dev of settling times)

**For field-weakening (6):**
- All metrics computed for **both** i_d and i_q

---

## Interpretation Guide

### Comparing SNN vs PI Baseline

**Good SNN performance:**
- MAE i_q within 10% of PI baseline (all scenarios)
- Settling time within 20% of PI (scenarios 1-3)
- Overshoot < 10% (scenarios 1-3)
- No safety violations (all scenarios)

**Acceptable tradeoffs:**
- Slightly higher overshoot (up to 15%) acceptable if MAE is lower
- Longer settling (up to 30%) acceptable for better steady-state tracking

**Red flags:**
- Safety violations (indicates instability)
- MAE > 2x PI baseline
- Settling time > 2x PI baseline
- Failure on Scenario 2 (nominal conditions)

### Scenario-Specific Insights

**Scenario 1 (Low Speed):**
- If performance is **worse** here than Scenario 2: controller has low-speed sensitivity issues
- If performance is **better** here: voltage saturation may be limiting high-speed performance

**Scenario 2 (Mid Speed) ⭐:**
- This is your **headline number**. Report this prominently.
- Direct comparison point with PI and other controllers

**Scenario 3 (High Speed):**
- If performance degrades significantly vs Scenario 2: voltage saturation or back-EMF issues
- Check maximum voltage utilization metric

**Scenario 4 (Multi-Step):**
- If consistency is poor (high std dev in settling times): memory effects or state-dependent performance
- If later steps are worse: controller fatigue or state accumulation issues

**Scenario 5 (Four-Quadrant):**
- If zero-crossing has large error spike: deadtime compensation needed
- If -2A performance differs from +2A: asymmetric controller behavior

**Scenario 6 (Field-Weakening):**
- If i_d affects i_q significantly: poor d-q decoupling (common in basic controllers)
- If both axes settle well: advanced multivariable control capability

---

## Extending the Suite

### Adding Custom Scenarios

```python
from embark.benchmark.harness import ScenarioDefinition
from embark.benchmark.tasks.reference_generators import MultiStepReference

# Create custom scenario
my_scenario = ScenarioDefinition(
    name="custom_aggressive_transient",
    description="Very fast transient for stress testing",
    n_rpm=2000.0,
    reference_generator=MultiStepReference(
        steps=[
            (0.0, 0.0, 0.0),
            (0.05, 0.0, 5.0),  # Fast 5A step
            (0.15, 0.0, -5.0), # Fast reversal
        ]
    ),
    max_steps=5000,
)

# Run with custom scenarios
suite = BenchmarkSuite(scenarios=[my_scenario] + STANDARD_SCENARIOS)
```

### When to Add Scenarios

Add custom scenarios if:
- Your application has specific operating conditions not covered
- You need domain-specific validation (e.g., automotive vs industrial)
- You're investigating specific failure modes

**Don't add scenarios for:**
- Redundant coverage (e.g., multiple speeds with same dynamics)
- Exhaustive parameter sweeps (use sensitivity analysis instead)
- Overfitting to your specific controller

---

## Best Practices

### Reporting Results

When publishing or comparing controllers, **always include**:
1. **Scenario 2** (primary reference) results prominently
2. **All 6 scenarios** for comprehensive evaluation
3. Comparison with **PI baseline** on same scenarios
4. Safety violation count
5. Aggregate metrics (mean MAE, worst-case error)

### Development Workflow

**During development:**
1. Start with `QUICK_SCENARIOS` (fast iteration)
2. Use Scenario 2 as primary tuning target
3. Fix failures before optimizing performance

**For validation:**
1. Run full `STANDARD_SCENARIOS` suite
2. Compare against PI baseline
3. Investigate any outlier scenarios

**For publication:**
1. Run multiple seeds (if stochastic controller)
2. Report mean ± std dev across runs
3. Include PI baseline results
4. Save raw results JSON for reproducibility

---

## References

1. [Motor Control Benchmarking Best Practices](https://arxiv.org/html/2512.06603v1) - ArXiv, 2024
2. [Transient Performance Evaluation in Motor Control](https://arxiv.org/html/2402.01782v1) - ArXiv, 2024
3. [Regenerative Braking in PMSM Systems](https://www.nature.com/articles/s41598-025-02396-y) - Nature Scientific Reports
4. [Field-Oriented Control and d-q Decoupling](https://www.nature.com/articles/s41598-025-19384-x) - Nature Scientific Reports

---

## Summary

The 6-scenario suite provides:

✅ **Complete speed-range coverage** (500/1500/2500 RPM)  
✅ **Transient characterization** (settling time, overshoot, rise time)  
✅ **Dynamic tracking** (multi-step, consistency)  
✅ **Quadrant coverage** (motoring, generating, torque reversal)  
✅ **Advanced operation** (field-weakening, d-q coupling)  
✅ **Manageable runtime** (~5-10 seconds total)

This is the **minimum necessary coverage** to claim comprehensive benchmarking while keeping analysis manageable for iterative development.
