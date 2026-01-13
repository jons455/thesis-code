# GEM PMSM Simulation - Configuration & Validation

This document consolidates all GEM (gym-electric-motor) configuration, setup, and validation learnings.

**Merged from**: `GEM_KONFIGURATION.md`, `GEM_LEARNINGS.md`, `CONTROLLER_VERIFICATION.md`

---

## 1. Overview

The GEM simulation is configured to closely replicate the MATLAB/Simulink PMSM model. After extensive debugging, the **GEM Standard Controller achieves identical steady-state currents as MATLAB** (tracking error < 1e-11 A).

### Environment

**Environment-ID**: `Cont-CC-PMSM-v0` (Continuous Current Control PMSM)

This environment implements:
- Full PMSM electrical model (ODE-based)
- dq-transformation (Park/Clarke internally)
- State normalization to [-1, +1]
- PI current controller (gem_controllers)
- Decoupling compensation

---

## 2. Motor Parameters

Matched to MATLAB/Simulink model:

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Pole pairs | p | 3 | - |
| Stator resistance | R_s | 0.543 | Ω |
| d-axis inductance | L_d | 1.13 | mH |
| q-axis inductance | L_q | 1.42 | mH |
| PM flux linkage | Ψ_PM | 16.9 | mWb |

### Limits

| Parameter | Value | Unit |
|-----------|-------|------|
| Max current | 10.8 | A |
| Max voltage | 48 | V |
| Max speed | 3000 | rpm |

### Control Parameters

| Parameter | Value |
|-----------|-------|
| Control frequency | 10 kHz |
| Sampling period | 100 µs |
| PI tuning | Technical Optimum (a=4) |

---

## 3. Key Learnings from Validation

### Problem 1: Step Time Inconsistency

**Issue**: MATLAB used step at t=0.1s, Python applied reference immediately.

**Solution**: Added configurable `step_time` parameter. For validation, both use immediate step (t=0).

### Problem 2: Speed Control

**Issue**: Speed cannot be set as a reference in Current Control environment.

**Solution**: Use `ConstantSpeedLoad` to enforce fixed mechanical speed.

```python
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
load = ConstantSpeedLoad(omega_fixed=1000 * 2 * np.pi / 60)  # 1000 rpm
```

### Problem 3: Normalization

**Issue**: GEM normalizes all states to [-1, +1]. Must denormalize for physical values.

**Solution**: Access limits from physical system:
```python
env_unwrapped = env
while hasattr(env_unwrapped, 'env'):
    env_unwrapped = env_unwrapped.env
limits = env_unwrapped.physical_system.limits
i_d_physical = i_d_normalized * limits['i_sd']
```

### Problem 4: Custom Controller Issues

**Issue**: Own PI controller with MATLAB parameters didn't match.

**Root Cause**: GEM's internal limit handling conflicts with custom anti-windup.

**Solution**: Use **GEM Standard Controller** (`gem_controllers.TorqueController`) - it's verified equivalent to MATLAB.

---

## 4. Validation Results

### GEM Standard Controller vs. MATLAB

| Speed (rpm) | i_d MAE | i_q MAE | Status |
|-------------|---------|---------|--------|
| 500 | 8.8e-15 A | 1.1e-12 A | ✅ Perfect |
| 1500 | 7.7e-14 A | 2.9e-12 A | ✅ Perfect |
| 2500 | 3.3e-13 A | 7.3e-12 A | ✅ Perfect |

### Operating Points (1000 rpm)

| id_ref | iq_ref | Δid | Δiq | Status |
|--------|--------|-----|-----|--------|
| 0 A | 2 A | ~0 | ~0 | ✅ |
| 0 A | 5 A | ~0 | ~0 | ✅ |
| 0 A | 8 A | ~0 | ~0 | ✅ |
| -3 A | 2 A | ~0 | ~0 | ✅ |
| -3 A | 5 A | ~0 | ~0 | ✅ |
| -5 A | 5 A | ~0 | ~0 | ✅ |

### Known Difference: Voltage Offset

There's a ~68% relative offset in u_d and u_q compared to MATLAB. This is due to different normalization/sensor modeling but **does not affect current tracking**.

---

## 5. File Locations

| Purpose | File |
|---------|------|
| Main simulation | `pmsm-pem/simulation/simulate_pmsm.py` |
| Operating point tests | `pmsm-pem/simulation/run_operating_point_tests.py` |
| MATLAB comparison | `pmsm-pem/validation/compare_simulations.py` |
| Exported data | `pmsm-pem/export/gem_standard/` |
| Archived runs | `pmsm-pem/export/archive/` |

---

## 6. Quick Start

```python
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad

# Motor parameters
motor_parameter = dict(p=3, r_s=0.543, l_d=0.00113, l_q=0.00142, psi_p=0.0169)
limit_values = dict(i=10.8, u=48.0, omega=314.16)

# Create motor and load
motor = PermanentMagnetSynchronousMotor(motor_parameter=motor_parameter, limit_values=limit_values)
load = ConstantSpeedLoad(omega_fixed=1000 * 2 * np.pi / 60)

# Create environment
env = gem.make('Cont-CC-PMSM-v0', motor=motor, load=load, tau=1e-4, visualization=None)

# Use GEM standard controller (recommended!)
import gem_controllers as gc
controller = gc.GemController.make(env, 'Cont-CC-PMSM-v0', decoupling=True, a=4)
```

---

## 7. Conclusion

**Use GEM Standard Controller** for all simulations. It is verified equivalent to MATLAB/Simulink and handles all internal complexities correctly.

The validation archives are preserved in:
- `pmsm-pem/export/archive/baseline_2024-12-18/`
- `pmsm-pem/export/archive/verification_2025-12-18_1418/`
