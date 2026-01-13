# Documentation

**Project**: Neuromorphic PMSM Controller Benchmark  
**Goal**: Systematic evaluation pipeline comparing PI vs SNN controllers for PMSM current control.

---

## Active Documents

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, components, data flow |
| [BENCHMARK_METRICS.md](BENCHMARK_METRICS.md) | Control + neuromorphic metrics documentation |
| [SIMULATION.md](SIMULATION.md) | GEM configuration, validation, learnings |
| [WORK_PROGRESS.md](WORK_PROGRESS.md) | Progress log with dates |

---

## Project Structure

```
thesis-code/
├── benchmark/              # NeuroBench integration (standalone)
│   ├── pmsm_env.py        # Gymnasium wrapper for GEM
│   ├── agents.py          # PI baseline, SNN controllers
│   └── run_benchmark.py   # Main benchmark script
│
├── pmsm-pem/              # GEM PMSM simulation
│   ├── simulation/        # Simulation scripts
│   ├── metrics/           # Metrics framework (~1100 lines)
│   ├── validation/        # MATLAB comparison
│   └── export/            # Generated data
│
├── pmsm-matlab/           # MATLAB/Simulink reference
│
└── docs/                  # This folder
    └── archive/           # Old/superseded docs
```

---

## Quick Links

- **Run benchmark**: `cd benchmark && python run_benchmark.py`
- **GEM simulation**: `cd pmsm-pem && python simulation/simulate_pmsm.py`
- **Compare to MATLAB**: `cd pmsm-pem && python validation/compare_simulations.py`

---

## Status (2026-01-13)

| Work Package | Status |
|--------------|--------|
| WP1: Simulation & Baseline | ✅ Complete |
| WP2: NeuroBench Integration | ✅ Complete |
| WP3: SNN Implementation | Pending |
| WP4: Evaluation | Pending |

---

## Archive

The `archive/` folder contains superseded documentation:
- Edge Impulse guides (alternative approach, not used)
- Old training guides (to be updated for snnTorch)
- Original GEM docs (consolidated into SIMULATION.md)

