# PMSM Neuromorphic Controller Benchmark

[![CI](https://github.com/jonas/pmsm-neuromorphic-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/jonas/pmsm-neuromorphic-benchmark/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Benchmark framework for evaluating neuromorphic (SNN) controllers against conventional PI controllers for PMSM (Permanent Magnet Synchronous Motor) current control.

Part of a Master's thesis on neuromorphic computing for motor control applications.

## Features

- **GEM Integration**: Validated PMSM simulation using gym-electric-motor
- **NeuroBench Compatible**: Follows NeuroBench framework for standardized evaluation
- **Comprehensive Metrics**: Control performance (ITAE, settling time), neuromorphic metrics (SyOps, sparsity)
- **Reproducible**: Seed management and experiment tracking utilities

## Installation

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- (Optional) CUDA-capable GPU for faster SNN training

### Quick Install

```bash
# Clone the repository
git clone https://github.com/jonas/pmsm-neuromorphic-benchmark.git
cd pmsm-neuromorphic-benchmark

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Development Setup

```bash
# Install with all dependency groups (dev, docs, jupyter)
poetry install --with dev,docs,jupyter

# Install pre-commit hooks
poetry run pre-commit install

# Verify installation
poetry run pytest --collect-only
```

## Project Structure

The framework is designed as a **modular pipeline** where each component can be developed, tested, and documented independently. Each module has its own README with usage examples and API documentation.

```
thesis-code/
├── benchmark/                # NeuroBench integration        → README.md
│   ├── pmsm_env.py          # Gymnasium wrapper for GEM
│   ├── agents.py            # PI controller, SNN placeholder
│   ├── processors.py        # Spike encoding utilities
│   └── tests/               # Unit tests
│
├── metrics/                  # Benchmark metrics framework   → METRICS_DOCUMENTATION.md
│   ├── benchmark_metrics.py # ~1100 lines of metrics
│   └── tests/               # Metric tests
│
├── utils/                    # Utility modules
│   └── reproducibility.py   # Seed management, experiment tracking
│
├── pmsm-pem/                 # GEM PMSM simulation           → README.md
│   ├── simulation/          # Simulation scripts
│   ├── validation/          # MATLAB comparison
│   └── export/              # Results (CSV, plots)
│
├── pmsm-matlab/              # MATLAB/Simulink reference     → README.md
│   ├── foc_pmsm.slx         # Simulink FOC model
│   └── pmsm_init.m          # Motor parameters
│
├── data-preperation/         # Data processing (legacy)      → README.md
│   └── data_exploration.ipynb
│
│   ├── ARCHITECTURE.md      # System architecture
│   ├── BENCHMARK_METRICS.md # Metrics documentation
│   └── SIMULATION.md        # GEM configuration
│
├── tests/                    # Integration & regression tests
└── pyproject.toml           # Project & tool configuration (Poetry)
```

This modularity allows swapping components (e.g., different controllers, encoding schemes) without changing the overall pipeline.

## Motor Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Pole pairs (p) | 3 | - |
| Stator resistance (R_s) | 0.543 | Ω |
| d-axis inductance (L_d) | 1.13 | mH |
| q-axis inductance (L_q) | 1.42 | mH |
| PM flux linkage (Ψ_PM) | 16.9 | mWb |
| Maximum current (I_max) | 10.8 | A |
| DC-link voltage (V_DC) | 48 | V |
| Maximum speed (n_max) | 3000 | RPM |

## Usage

### Running the Benchmark

```python
from benchmark import PMSMEnv, PIControllerAgent
from metrics import run_benchmark
import pandas as pd

# Create environment and agent
env = PMSMEnv(max_steps=500)
agent = PIControllerAgent()

# Run simulation
state, _ = env.reset()
for _ in range(500):
    action = agent(state)
    state, reward, done, truncated, info = env.step(action)
    if done:
        break

# Evaluate with metrics
df = pd.DataFrame(env.episode_data)
result = run_benchmark(df, controller_name="PI Baseline")
print(result.summary())
```

### Reproducible Experiments

```python
from utils import set_seed, ExperimentConfig

# Set all random seeds
set_seed(42)

# Create experiment configuration
config = ExperimentConfig(
    seed=42,
    model_name="snn_lif_64",
    hyperparameters={"hidden_size": 64, "beta": 0.9}
)
config.save("experiments/exp_001.yaml")
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=benchmark --cov=metrics --cov=utils --cov-report=html

# Run specific test categories
poetry run pytest -m "not slow"           # Skip slow tests
poetry run pytest -m integration          # Integration tests only
poetry run pytest tests/test_regression.py  # Regression tests
```

## Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .
poetry run ruff check --fix .  # Auto-fix issues

# Type check
poetry run mypy benchmark metrics utils
```

## Validation Results

The PI controller baseline achieves precise tracking across all operating points:

| Test | i_d [A] | i_q [A] | Tracking Error |
|------|---------|---------|----------------|
| Baseline | 0 | 2 | 0.0000 A |
| Medium Load | 0 | 5 | 0.0000 A |
| High Load | 0 | 8 | 0.0000 A |
| Field Weakening | -3 | 2 | 0.0000 A |
| FW + Load | -3 | 5 | 0.0000 A |
| Strong FW | -5 | 5 | 0.0000 A |

## Simulation Parameters

- **Sampling rate**: 10 kHz (Ts = 100 µs)
- **Episode duration**: 0.2 s per run
- **Steps per episode**: 2000


## Acknowledgments

- [gym-electric-motor (GEM)](https://github.com/upb-lea/gym-electric-motor) - Electric motor simulation
- [NeuroBench](https://neurobench.readthedocs.io/) - Neuromorphic computing benchmarks
- [snnTorch](https://snntorch.readthedocs.io/) - Spiking neural network framework
