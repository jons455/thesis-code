# Evaluation Module

Benchmark evaluation framework for SNN controllers trained via imitation learning from PI controller trajectories.

## Purpose

This module provides:
- **SNNBenchmarkController**: A benchmark-compatible wrapper for trained SNN models
- **run_evaluation.py**: Script for comparing SNN controllers against PI baseline
- Example of how external users can integrate controllers with the benchmark framework

## Quick Start

```bash
# Run the benchmark evaluation (from project root)
poetry run python evaluation/core/run_evaluation.py

# With custom operating point
poetry run python evaluation/core/run_evaluation.py --speed 1500 --iq-ref 3.0

# Save results to JSON
poetry run python evaluation/core/run_evaluation.py --save-results data/results/run_001.json
```

## Files

| File | Purpose |
|------|---------|
| `run_evaluation.py` | Evaluation script comparing SNN vs PI controllers |
| `models/` | Directory containing trained model checkpoints |
| `models/best/best_model.pt` | Pre-trained SNN (imitation learning from PI) |

## Using the Controller

### Basic Usage

```python
from evaluation.core import SNNBenchmarkController
from embark.benchmark import PMSMEnv
from embark.benchmark.controller_interface import run_benchmark

# Load trained SNN controller
controller = SNNBenchmarkController("models/best/best_model.pt")

# Create environment
env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0)

# Run benchmark
results = run_benchmark(controller, env)
print(results.summary())
```

### Manual Episode Loop

```python
from evaluation.core import SNNBenchmarkController
from embark.benchmark import PMSMEnv
from embark.benchmark.processors import get_default_processors

controller = SNNBenchmarkController(
    "models/best/best_model.pt",
    device="cpu",
    track_spikes=True,
)

env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0)
processors = get_default_processors(controller, env.config.i_max, env.config.u_max)
state_raw, _ = env.reset()
state = processors.state_preprocessor(state_raw)
controller.reset()  # Reset neuron states

for step in range(2000):
    action = controller(state)  # normalized
    action_env = processors.action_postprocessor(action)
    state_raw, reward, done, truncated, info = env.step(action_env)
    state = processors.state_preprocessor(state_raw)
    if done:
        break

# Get neuromorphic metrics
metrics = controller.get_neuromorphic_metrics()
print(f"Total spikes: {metrics['total_spikes']:,}")
print(f"Sparsity: {metrics['sparsity']*100:.1f}%")
```

## Controller Interface

The `SNNBenchmarkController` implements the benchmark's `ControllerInterface` protocol:

```python
class SNNBenchmarkController:
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute control action from state.

        Args:
            state: Normalized state [i_d, i_q, e_d, e_q] in [-1, 1]

        Returns:
            Normalized action [u_d, u_q] in [-1, 1]
        """
        ...

    def reset(self) -> None:
        """Reset controller state for new episode."""
        ...

    def get_info(self) -> dict:
        """Return controller metadata."""
        ...

    def get_neuromorphic_metrics(self) -> dict | None:
        """Return neuromorphic metrics (spikes, sparsity, etc.)."""
        ...
```

## Configuration Options

### SNNBenchmarkController Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str | Required | Path to trained model (.pt file) |
| `device` | str | "cpu" | Device for inference ("cpu" or "cuda") |
| `track_spikes` | bool | True | Track spike activity for metrics |
| `num_inference_steps` | int | 1 | SNN timesteps per control step |

### run_evaluation.py Options

```
usage: run_evaluation.py [-h] [--model MODEL] [--device {cpu,cuda}]
                         [--speed SPEED] [--id-ref ID_REF] [--iq-ref IQ_REF]
                         [--max-steps MAX_STEPS] [--inference-steps STEPS]
                         [--save-results FILE]

Options:
  --model MODEL         Path to SNN checkpoint (default: evaluation/models/best_model.pt)
  --device {cpu,cuda}   Inference device (default: cpu)
  --speed SPEED         Motor speed in RPM (default: 1000)
  --id-ref ID_REF       d-axis current reference in A (default: 0.0)
  --iq-ref IQ_REF       q-axis current reference in A (default: 5.0)
  --max-steps STEPS     Steps per episode (default: 2000)
  --save-results FILE   Save results to JSON file
```

## Metrics Reported

### Accuracy Metrics
- **RMSE i_d/i_q**: Root mean square tracking error [mA]
- **MAE i_d/i_q**: Mean absolute tracking error [mA]
- **Final error**: Steady-state tracking error [mA]

### Dynamics Metrics
- **Settling time**: Time to reach ±2% of setpoint [ms]
- **Rise time**: 10% to 90% response time [ms]
- **Overshoot**: Maximum overshoot [%]

### Neuromorphic Metrics (SNN only)
- **Total spikes**: Spike count over episode
- **Spikes per step**: Average spikes per control step
- **Sparsity**: Fraction of non-spiking neurons (higher = more efficient)
- **Neurons**: Total neuron count
- **Synapses**: Total synapse count
- **Inference latency**: Time per control step [µs]

## Model Details

The included `best_model.pt` is a Spiking Neural Network trained via imitation learning:

| Property | Value |
|----------|-------|
| Architecture | 2-layer LIF SNN |
| Input | 4 (i_d, i_q, e_d, e_q) |
| Hidden | 64 neurons per layer |
| Output | 2 (u_d, u_q) |
| Beta (hidden) | 0.9 |
| Beta (output) | 0.995 |
| Training | Imitation learning from PI trajectories |

**Note**: This model was trained for pipeline testing and may not achieve optimal control performance.

## Adding New Controllers

To benchmark your own controller:

1. Create a class implementing the `ControllerInterface`:

```python
from benchmark.controller_interface import ControllerAgent
import numpy as np

class MyController(ControllerAgent):
    def __init__(self, ...):
        # Load your model
        self.model = ...

    def __call__(self, state: np.ndarray) -> np.ndarray:
        # Compute control action
        action = self.model.predict(state)
        return np.clip(action, -1.0, 1.0)

    def reset(self) -> None:
        # Reset any internal state
        pass

    def get_info(self) -> dict:
        return {"name": "MyController", "type": "custom"}
```

2. Run benchmark:

```python
from benchmark import PMSMEnv
from benchmark.controller_interface import run_benchmark

controller = MyController(...)
env = PMSMEnv(n_rpm=1000, i_q_ref=5.0)
results = run_benchmark(controller, env)
print(results.summary())
```

## See Also

- `benchmark/README.md` — Benchmark framework documentation
- `benchmark/controller_interface.py` — Controller interface specification
- `snn/README.md` — SNN architecture and training
- `docs/BENCHMARK_METRICS.md` — Detailed metrics documentation
