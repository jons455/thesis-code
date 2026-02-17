# PWM Current State (Single Source of Truth)

This document is the canonical reference for PWM and dead-time behavior in this repository.

## Short Answer

Yes, the simulation pipeline is now in a clean state:
- GEM handles inverter dead-time internally.
- Simulation controller output decoding is linear by default.
- PWM post-processing is opt-in only.

## Current Ground Truth

## 1) Physics/GEM

File: `embark/benchmark/physics/pmsm.py`

```python
converter = ContB6BridgeConverter(
    tau=self.config.tau,
    interlocking_time=self.config.dead_time,
)
```

- GEM dead-time is enabled in this repo via `interlocking_time=self.config.dead_time`.
- Therefore, GEM is not in the idealized `interlocking_time=0.0` default mode.

## 2) Training pipeline

File: `evaluation/generate_training_data.py`

- PI controller outputs `v_d`/`v_q`.
- Those voltages are passed directly to `task.step(...)`.
- No custom PWM stage is injected in training loop.
- Dead-time/non-idealities come from GEM converter behavior.

## 3) Evaluation pipeline

Primary file: `evaluation/core/run_evaluation.py`

- Default decoder: `LinearActionProcessor`.
- Optional flag: `--pwm` to enable PWM output processing explicitly.
- This avoids double dead-time in normal simulation runs.

Supporting updates:
- `evaluation/analysis/compare_models.py`: linear decode for tensor controllers.
- `evaluation/analysis/compare_akida.py`: linear decode + Akida instantiated with `enable_pwm=False` in simulation comparisons.
- `evaluation/akida/run_benchmark.py`: added `--pwm` opt-in, default off.

## FAQ

### Do we need dead-time in GEM simulation?
Yes. Keep dead-time enabled in GEM for simulation benchmarks in this repo.

### Should we flatten/disable dead-time in GEM?
No for default simulation benchmarking. Disabling GEM dead-time would make simulation less representative of configured inverter behavior and break current alignment assumptions.

### Should we enable PWMActionProcessor in simulation by default?
No. Default should remain linear decode because GEM already applies dead-time through the converter.

### When should PWMActionProcessor/PWMConverter be used?
Use it only for explicit hardware-oriented experiments where PWM/duty-cycle behavior is part of the evaluated pipeline, and ensure training/evaluation/deployment are consistent.

### Do we need to regenerate training data now?
No, not for this specific pipeline fix. Training already follows PI -> task.step() with GEM handling converter effects.

### Why did oscillation happen before?
Custom PWM dead-time plus GEM dead-time effectively stacked in evaluation paths, creating a mismatch versus training behavior.

## Recommended Operating Modes

- **Simulation benchmark (default)**
  - GEM dead-time: ON
  - Controller output decoder: Linear
  - PWM post-processing: OFF

- **Hardware-oriented experiment (opt-in)**
  - PWM post-processing: ON (explicit)
  - Keep the full pipeline consistent end-to-end
  - Document run settings with each result
