# SNN Controller Architecture & Implementation Plan

This document outlines the SNN controller approaches for PMSM current control, with implementation priority on the **Pure SNN** approach.

---

## Overview: Two Approaches

| Approach | Description | Pipeline Complexity | Akida Deployment |
|----------|-------------|---------------------|------------------|
| **Pure SNN** (Primary) | SNN handles integration internally via slow-leak output neurons | Simple | Fully on-chip ✅ |
| **Hybrid SNN** (Optional) | SNN outputs "kicks", external integrator accumulates | More complex | SNN on chip, integrator on host |

We implement **Pure SNN first** because:
1. Simpler pipeline (no pre/post processors needed)
2. Full neuromorphic deployment possible
3. Same training data works for both approaches
4. Can add Hybrid later with minimal changes

---

## Approach 1: Pure SNN (Primary)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PURE SNN CONTROLLER                          │
│                                                                     │
│  Input              Hidden Layers           Output Layer            │
│  [i_d, i_q,    ┌──────────────────────┐   ┌─────────────────┐      │
│   e_d, e_q] ──▶│ Dense → LIF (β=0.9)  │──▶│ Dense → LIF     │      │
│  (normalized)  │ Dense → LIF (β=0.9)  │   │ (β=0.995)       │      │
│                └──────────────────────┘   │ ↓               │      │
│                       ↓ spikes            │ Membrane = u_d,u_q     │
│                                           └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input encoding | Direct (no deltas) | Simplest; hidden layers handle dynamics |
| Hidden neurons | LIF with β=0.9 | Fast response to input changes |
| Output neurons | LIF with β=0.995 | Slow leak acts as integrator |
| Output readout | Membrane potential | Continuous value, not spikes |
| Reset mechanism | None for output | Accumulate without reset |
| Training target | Absolute voltage [u_d, u_q] | Direct imitation of PI controller |

### How It Solves the Steady-State Problem

At steady state (constant reference, zero error):
1. Input [i_d, i_q, e_d, e_q] is constant
2. Hidden LIF neurons fire at a stable rate (determined by input level)
3. Output neurons receive constant spike input → membrane stabilizes
4. Slow leak (β=0.995) means membrane holds value with minimal decay
5. Result: Output voltage is maintained without external integrator

### Data Flow

```
Environment State          SNN                    Action
[i_d, i_q, e_d, e_q] ──▶ SimpleSNNController ──▶ [u_d, u_q]
   (normalized)           (membrane readout)     (normalized)
```

### Pipeline Components

```
snn/
├── __init__.py
├── models.py         # SimpleSNNController class
├── dataset.py        # PMSMDataset: loads PI trajectories
└── train.py          # Training script

benchmark/
├── agents.py         # Add SNNControllerAgent (wraps trained model)
└── run_benchmark.py  # Add SNN evaluation option
```

### Training Details

| Aspect | Specification |
|--------|---------------|
| Framework | snnTorch (PyTorch-based) |
| Input | [i_d, i_q, e_d, e_q] normalized to [-1, 1] |
| Target | [u_d, u_q] normalized to [-1, 1] |
| Loss | MSE between membrane output and target voltage |
| Sequence handling | BPTT through time (trajectory windows) |
| Batch size | 32-64 trajectories |
| Learning rate | 1e-3 (Adam) |

---

## Approach 2: Hybrid SNN-Integrator (Future Option)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HYBRID SNN CONTROLLER                          │
│                                                                     │
│  Preprocessor           SNN               Postprocessor             │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐         │
│  │ DeltaEncoder │──▶│ LIF Network  │──▶│ Integrator      │         │
│  │ [i,e]→[i,Δe] │   │ (all β=0.9)  │   │ u += kick       │         │
│  └──────────────┘   └──────────────┘   └─────────────────┘         │
│                            ↓                    ↓                   │
│                      [kick_d, kick_q]     [u_d, u_q]                │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Pure SNN

| Aspect | Pure SNN | Hybrid SNN |
|--------|----------|------------|
| Input | [i_d, i_q, e_d, e_q] | [i_d, i_q, Δe_d, Δe_q] |
| SNN output | Membrane → voltage | Spikes → decoded kicks |
| Integration | Inside SNN (slow-leak) | External postprocessor |
| Training target | u_d, u_q (absolute) | Δu = u[t] - u[t-1] |
| Steady-state sparsity | Hidden sparse, output tonic | All layers can be sparse |

### Additional Components Needed

```
benchmark/
├── processors.py     # Add:
│   ├── DeltaEncodingPreprocessor
│   └── IntegratorPostprocessor
├── config.py         # ProcessorConfig dataclass
└── runner.py         # EpisodeRunner with processor chain
```

### When to Use Hybrid

Consider Hybrid approach if:
- Pure SNN shows voltage drift at steady state
- Need maximum sparsity (silent at steady state)
- Want to compare approaches in thesis

---

## Implementation Plan

### Phase 1: Pure SNN Foundation (Days 1-2)

#### Day 1: Model & Dataset

**Task 1.1: Create folder structure**
```
snn/
├── __init__.py
├── models.py
├── dataset.py
└── train.py
```

**Task 1.2: Implement SimpleSNNController**
- 4 input neurons (i_d, i_q, e_d, e_q)
- 2 hidden layers (64 neurons each, LIF β=0.9)
- 2 output neurons (slow-leak LIF β=0.995)
- Forward pass returns membrane potential

**Task 1.3: Implement PMSMDataset**
- Load CSV files from `pmsm-pem/export/train/`
- Extract columns: i_sd, i_sq, u_sd, u_sq
- Compute errors: e_d = i_d_ref - i_d, e_q = i_q_ref - i_q
- Normalize by limits (i_max=10.8A, u_max=48V)
- Return (input_sequence, target_sequence) pairs

#### Day 2: Training Script

**Task 2.1: Implement train.py**
- DataLoader with trajectory batching
- Training loop with BPTT
- Validation split (80/20)
- Save best model checkpoint
- Learning curves (loss vs epoch)

**Task 2.2: Train initial model**
- Run training for 50-100 epochs
- Verify loss decreases
- Check for obvious issues

### Phase 2: Closed-Loop Integration (Days 2-3)

**Task 2.3: Add SNNControllerAgent to benchmark/agents.py**
- Load trained .pt file
- Maintain SNN state across timesteps
- Reset state on episode reset
- `__call__(state) -> action` interface

**Task 2.4: Test in closed loop**
- Use existing PMSMEnv
- Run simple episode
- Check for stability (no NaN, no explosion)
- Visualize step response

### Phase 3: Evaluation (Days 3-4)

**Task 3.1: Benchmark metrics**
- Use existing metrics framework
- Compare SNN vs PI:
  - RMSE, ITAE (tracking accuracy)
  - Settling time, overshoot (dynamics)
  - Control smoothness (voltage chattering)

**Task 3.2: Neuromorphic metrics**
- Activation sparsity (% of silent neurons)
- SyOps estimation (synaptic operations)
- Energy estimate (pJ/inference)

**Task 3.3: Generate plots**
- Step response comparison
- Tracking error over time
- Sparsity visualization

### Phase 4 (Optional): Add Hybrid Approach (Days 4-5)

Only if Pure SNN works and time permits:

**Task 4.1: Implement preprocessors**
- DeltaEncodingPreprocessor
- IntegratorPostprocessor

**Task 4.2: Implement HybridSNN model**
- Same architecture but standard output layer
- Training target: Δu instead of u

**Task 4.3: Compare approaches**
- Run same benchmarks on both
- Document differences in thesis

---

## File Specifications

### snn/models.py

```python
"""
SNN Controller Models for PMSM Current Control
==============================================

Models:
- SimpleSNNController: Pure SNN with slow-leak output (primary)
- HybridSNNController: SNN with external integrator (future)
"""

import torch
import torch.nn as nn
import snntorch as snn


class SimpleSNNController(nn.Module):
    """
    Pure SNN controller with built-in integration.
    
    The output layer uses slow-leak LIF neurons (high beta)
    whose membrane potential directly encodes the voltage command.
    
    Parameters
    ----------
    hidden_size : int
        Number of neurons in hidden layers (default: 64)
    beta_hidden : float
        Decay rate for hidden layers (default: 0.9)
    beta_output : float
        Decay rate for output layer (default: 0.995)
    """
    # Implementation here
    pass


# Future: HybridSNNController for comparison
```

### snn/dataset.py

```python
"""
Dataset for PMSM SNN Training
=============================

Loads PI controller trajectories and prepares them for SNN training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class PMSMDataset(Dataset):
    """
    Dataset of PI controller trajectories for imitation learning.
    
    Each sample is a trajectory window:
    - Input: [i_d, i_q, e_d, e_q] sequence
    - Target: [u_d, u_q] sequence
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files
    window_size : int
        Number of timesteps per training window
    stride : int
        Step between windows (for augmentation)
    """
    # Implementation here
    pass
```

### snn/train.py

```python
"""
Training Script for PMSM SNN Controller
=======================================

Usage:
    python -m snn.train --epochs 100 --batch_size 32
"""

import argparse
import torch
from torch.utils.data import DataLoader
from snn.models import SimpleSNNController
from snn.dataset import PMSMDataset


def train():
    # Training loop
    pass


if __name__ == "__main__":
    train()
```

### benchmark/agents.py (Addition)

```python
# Add to existing agents.py:

class SNNControllerAgent:
    """
    SNN controller agent for benchmark evaluation.
    
    Wraps a trained SimpleSNNController for closed-loop control.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = SimpleSNNController.load(model_path)
        self.model.eval()
        self.device = device
        self.state = None
    
    def reset(self):
        """Reset SNN membrane states."""
        self.state = None
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Compute control action from observation."""
        with torch.no_grad():
            x = torch.tensor(observation, dtype=torch.float32)
            voltage, self.state = self.model(x.unsqueeze(0), self.state)
            return voltage.squeeze(0).numpy()
```

---

## Success Criteria

### Minimum Viable Product (MVP)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Training converges | Loss < 0.01 | Training curve |
| Closed-loop stable | No NaN/explosion | Episode completion |
| Tracks reference | RMSE < 1.0 A | Benchmark metrics |
| Demonstrates sparsity | > 50% silent neurons | Activation logging |

### Good Result

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Competitive tracking | RMSE < 0.5 A | Within 5× of PI |
| Fast response | Settling time < 10ms | Step response |
| High sparsity | > 80% silent | NeuroBench metrics |
| Smooth control | TV < 2× PI | Control smoothness |

### Excellent Result (Thesis Win)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Near-PI accuracy | RMSE < 0.2 A | Within 2× of PI |
| Energy advantage | < 1 mJ/inference | SyOps estimation |
| Akida-ready | Quantized model | 4-bit weights |

---

## Dependencies

```toml
# Add to pyproject.toml
[tool.poetry.dependencies]
snntorch = "^0.9"
```

Or:
```bash
pip install snntorch
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Training doesn't converge | Increase hidden size, tune beta values |
| Closed-loop unstable | Add gradient clipping, reduce learning rate |
| Poor steady-state | Increase output beta (0.999), or switch to Hybrid |
| Voltage drift | Add small output regularization, or add Hybrid integrator |

---

## Future Extensions

After Pure SNN works:

1. **Hybrid Approach**: Add external integrator for comparison
2. **Quantization**: Add QAT for Akida deployment
3. **NIR Export**: Export to neuromorphic intermediate representation
4. **Operating Points**: Test across speed/load conditions
5. **Disturbance Rejection**: Evaluate robustness

---

*Last Updated: 2026-01-20*
