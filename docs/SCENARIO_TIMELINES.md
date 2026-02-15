# Scenario Timeline Visualization

This document provides visual timelines for each of the 6 standard benchmark scenarios.

## Scenario 1: Single-Step Low Speed (500 RPM)

```
Duration: 0.3s (3000 steps @ 100µs)
Speed: 500 RPM (constant)

i_q [A]
  2 ┤    ┌─────────────────────
    │    │
  1 ┤    │
    │    │
  0 ┼────┘
    └────┬─────────────────────> time [s]
       0.0                    0.3

Tests: Low-speed sensitivity, baseline transient response
```

---

## Scenario 2: Single-Step Mid Speed (1500 RPM) ⭐

```
Duration: 0.3s (3000 steps @ 100µs)
Speed: 1500 RPM (constant)

i_q [A]
  2 ┤    ┌─────────────────────
    │    │
  1 ┤    │
    │    │
  0 ┼────┘
    └────┬─────────────────────> time [s]
       0.0                    0.3

Tests: PRIMARY REFERENCE - nominal performance, settling time, overshoot
```

---

## Scenario 3: Single-Step High Speed (2500 RPM)

```
Duration: 0.3s (3000 steps @ 100µs)
Speed: 2500 RPM (constant)

i_q [A]
  2 ┤    ┌─────────────────────
    │    │
  1 ┤    │
    │    │
  0 ┼────┘
    └────┬─────────────────────> time [s]
       0.0                    0.3

Tests: High-speed performance, voltage limits, back-EMF
```

---

## Scenario 4: Multi-Step Bidirectional (1500 RPM)

```
Duration: 1.0s (10000 steps @ 100µs)
Speed: 1500 RPM (constant)

i_q [A]
  2 ┤    ┌──┐     ┌──┐
    │    │  │     │  │
  0 ┼────┘  │     │  │
    │       │     │  │
 -2 ┤       └─────┘  └─────────
    └───┬───┬───┬───┬───┬──────> time [s]
      0.0 0.1 0.35 0.6 0.85   1.0

Step 1 (t=0.1s):  0 → +2A (motoring)
Step 2 (t=0.35s): +2 → -2A (generating)
Step 3 (t=0.6s):  -2 → +2A (motoring)
Step 4 (t=0.85s): +2 → -2A (generating)

Tests: Dynamic tracking, consistency, memory effects
```

---

## Scenario 5: Four-Quadrant Transition (1500 RPM)

```
Duration: 0.9s (9000 steps @ 100µs)
Speed: 1500 RPM (constant)

i_q [A]
  2 ┤    ┌──┐
    │    │  │
  0 ┼────┘  │        ┌─────────
    │       │        │
 -2 ┤       └────────┘
    └───┬───┬────┬───┬─────────> time [s]
      0.0 0.1  0.4 0.7       0.9

Transition 1 (t=0.1s): 0 → +2A (motoring)
Transition 2 (t=0.4s): +2 → -2A (TORQUE REVERSAL - hardest transient)
Transition 3 (t=0.7s): -2 → 0A (zero-crossing)

Tests: Regenerative braking, torque reversal, zero-crossing, deadtime effects
```

---

## Scenario 6: Field-Weakening (2500 RPM)

```
Duration: 0.6s (6000 steps @ 100µs)
Speed: 2500 RPM (constant)

i_d [A]
  0 ┼────┐
    │    │
 -2 ┤    └──────────────────────
    └────┬───────┬──────────────> time [s]
       0.0    0.1            0.6

i_q [A]
  2 ┤            ┌──────────────
    │            │
  0 ┼────────────┘
    └────┬───────┬──────────────> time [s]
       0.0    0.1   0.35     0.6

Transition 1 (t=0.1s):  i_d: 0 → -2A (field weakening activation)
Transition 2 (t=0.35s): i_q: 0 → +2A (torque with active field weakening)

Tests: d-q coupling/decoupling, multivariable control, voltage saturation
```

---

## Scenario Comparison Matrix

| Scenario | Speed [RPM] | i_q Transitions | i_d Active | Duration [s] | Total Steps |
|----------|-------------|-----------------|------------|--------------|-------------|
| 1. Low Speed | 500 | 1 (0→2A) | No | 0.3 | 3000 |
| 2. Mid Speed ⭐ | 1500 | 1 (0→2A) | No | 0.3 | 3000 |
| 3. High Speed | 2500 | 1 (0→2A) | No | 0.3 | 3000 |
| 4. Multi-Step | 1500 | 4 (±2A) | No | 1.0 | 10000 |
| 5. Four-Quadrant | 1500 | 3 (+2→-2→0) | No | 0.9 | 9000 |
| 6. Field-Weak. | 2500 | 2 (0→2A) | Yes (-2A) | 0.6 | 6000 |

**Total benchmark runtime**: ~3.1 seconds of simulated time (~31,000 control steps)

---

## Operating Point Coverage

### Speed Coverage
```
RPM
2500 ┤ ●─────────────● Scenarios 3, 6 (high-speed, voltage limits)
     │
1500 ┤ ●─●─●───────── Scenarios 2, 4, 5 (nominal, dynamic)
     │
 500 ┤ ●───────────── Scenario 1 (low-speed sensitivity)
     └──────────────────────────> Scenario type
        Step  Multi  Advanced
```

### Current Coverage (i_q axis)
```
i_q [A]
  2 ┤ ████████████████ All scenarios reach +2A
  1 ┤
  0 ┤ ████████████████ Scenarios 4, 5 include zero-crossing
 -1 ┤
 -2 ┤     ████████████ Scenarios 4, 5 include -2A (generating)
    └──────────────────────────> Scenario
       1  2  3  4  5  6
```

### d-axis Coverage
```
i_d [A]
  0 ┤ ████████████████ Scenarios 1-5 (i_d = 0)
    │
 -2 ┤              ███ Scenario 6 only (field-weakening)
    └──────────────────────────> Scenario
       1  2  3  4  5  6
```

---

## Transient Difficulty Ranking

From easiest to hardest:

1. **Scenario 1** - Low speed, single step (easiest)
2. **Scenario 2** - Nominal speed, single step
3. **Scenario 3** - High speed, single step (voltage limits)
4. **Scenario 4** - Multi-step tracking (consistency challenge)
5. **Scenario 5** - Four-quadrant with torque reversal (hardest unidirectional)
6. **Scenario 6** - Field-weakening with d-q coupling (hardest multivariable)

**Expected failure order**: Most controllers will fail on Scenario 6 first (advanced multivariable control), then Scenario 5 (torque reversal), then Scenario 4 (consistency).

---

## Design Rationale

### Why 0→2A Instead of Higher Currents?

- **2A is moderate load** (~40% of typical 5A continuous rating)
- Avoids current saturation effects that mask controller performance
- Focuses on controller dynamics rather than system limits
- Still sufficient to reveal transient behavior and settling characteristics

### Why These Specific Speeds?

- **500 RPM**: Low-speed operation where delays and quantization are most visible
- **1500 RPM**: Nominal mid-range operation (reference point)
- **2500 RPM**: High-speed where voltage limits and back-EMF matter
- **Coverage**: 5:1 speed ratio provides comprehensive range characterization

### Why Multi-Step in Scenario 4?

- **Consistency check**: Reveals if performance degrades over time
- **Memory effects**: Tests stateful controllers (SNNs) for state accumulation
- **Real-world relevance**: Varying torque demands are most common in applications
- **Both quadrants**: Covers motoring (+) and generating (-) in one scenario

### Why Four-Quadrant in Scenario 5?

- **Torque reversal** (+2→-2A) is the hardest transient in motor control
- **Regenerative braking** is critical for energy-efficient applications
- **Zero-crossing** reveals deadtime effects and low-signal behavior
- **Essential validation** for automotive and industrial applications

### Why Field-Weakening?

- **Only multivariable scenario**: Tests d-q coupling (i_d ≠ 0)
- **Separates basic from advanced controllers**: Most basic controllers fail here
- **Extends operating range**: Field-weakening enables higher speeds beyond base speed
- **Real-world necessity**: Required for high-speed operation in EVs and traction drives

---

## Timing Details

All scenarios use **100 µs sampling period** (10 kHz control frequency):
- Standard for PMSM benchmarking
- Typical for real-time motor control
- Sufficient for current loop bandwidth (≤1 kHz)

Step timings chosen to allow:
- **~100ms settling time** between steps (typical for well-tuned PI)
- **Clear separation** of transients for analysis
- **Reasonable duration** for visualization and debugging

---

## See Also

- **[BENCHMARK_SCENARIOS.md](BENCHMARK_SCENARIOS.md)** - Comprehensive scenario guide
- **[BENCHMARK_SUITE_QUICK_REFERENCE.md](BENCHMARK_SUITE_QUICK_REFERENCE.md)** - Quick reference guide
- **[README.md](../README.md)** - Main documentation
