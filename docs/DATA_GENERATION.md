# Data Generation Documentation

## MATLAB Simulation Setup

### Model Overview

The FOC controller is implemented in Simulink (`foc_pmsm.slx`) and configured via `pmsm_init.m`.

### Simulation Parameters

**Motor Parameters:**
- Nominal current: 4.2 A
- Maximum current: 10.8 A
- DC bus voltage: 48 V
- Maximum voltage (with SVPWM): ~27.7 V
- Nominal speed: 3000 RPM
- Sampling time: 100 μs (10 kHz control frequency)

**Electrical Parameters:**
- L_d = 1.13 mH (direct-axis inductance)
- L_q = 1.42 mH (quadrature-axis inductance)
- R_s = 0.543 Ω (stator resistance)
- Pole pairs = 3
- Ψ_PM = 16.9 mWb (permanent magnet flux linkage)

### Data Generation Process

1. **Randomized Operating Points:**
   Each simulation run uses randomly generated setpoints:
   ```matlab
   id_ref = rand * I_nenn;  % 0 to 4.2 A
   iq_ref = rand * I_nenn;  % 0 to 4.2 A
   n_ref  = rand * n_nenn;  % 0 to 3000 RPM
   ```

2. **Simulation Execution:**
   - 1000 independent simulation runs
   - Each run: 200 ms duration
   - Sampling: 10 kHz (2001 samples per run)
   - Total: 2,001,000 data points

3. **Output Format:**
   - CSV files: `sim_0001.csv` to `sim_1000.csv`
   - Columns: `time`, `i_d`, `i_q`, `n`, `u_d`, `u_q`
   - Time range: 0.0 to 0.2 seconds per run

### Data Characteristics

- **Diversity:** 1000 unique operating conditions ensure broad coverage of motor operating envelope
- **Temporal Resolution:** 100 μs sampling captures fast controller dynamics
- **Completeness:** All runs include transient and steady-state behavior

## Data Processing

See `data-preparation/README.md` for details on data merging and preprocessing.

