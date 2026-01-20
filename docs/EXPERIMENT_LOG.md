# Experiment Log

This document tracks training experiments, hyperparameter searches, and benchmark runs. Each experiment should be reproducible from the recorded configuration.

Use this for the thesis Results chapter.

---

## Experiment Index

| ID | Date | Description | Status | Key Result |
|----|------|-------------|--------|------------|
| E001 | 2026-01-20 | Quick test (3 epochs, 5 files) | ✅ Done | Loss: 0.04 |
| E001b | 2026-01-20 | Closed-loop verification (E001 model) | ✅ Done | Stable but untrained |
| E002 | 2026-01-20 | Full training baseline | 🔄 Running | — |
| E003 | — | Hyperparameter: hidden_size | 🔲 TODO | — |
| E004 | — | Hyperparameter: beta_output | 🔲 TODO | — |
| E005 | — | Closed-loop validation (E002 model) | 🔲 TODO | — |

---

## E001: Quick Test (Pipeline Validation)

**Date**: 2026-01-20  
**Goal**: Verify training pipeline works end-to-end  
**Status**: ✅ Complete

### Configuration

```yaml
# Model
hidden_size: 64
num_hidden_layers: 2
beta_hidden: 0.9
beta_output: 0.995

# Training
epochs: 3
batch_size: 32
learning_rate: 0.001
window_size: 100
stride: 50

# Data
max_files: 5  # Limited for quick test
val_split: 0.2
```

### Command
```bash
poetry run python -m snn.train --epochs 3 --max_files 5
```

### Results

| Epoch | Train Loss | Val Loss | MAE | Best? |
|-------|------------|----------|-----|-------|
| 1 | 0.149 | 0.071 | 0.216 | ✅ |
| 2 | 0.059 | 0.041 | 0.166 | ✅ |
| 3 | 0.046 | 0.040 | 0.165 | ✅ |

### Observations
- Loss decreases consistently → training works
- MAE of 0.165 on normalized voltage → ~8V error (needs improvement)
- Only 5 files used → expect better with full dataset
- Model saved to `snn/checkpoints/best_model.pt`

### Next Steps
- Run full training with all 580+ files
- Increase epochs to 100

---

## E001b: Closed-Loop Verification

**Date**: 2026-01-20  
**Goal**: Verify SNN can run in closed-loop with GEM simulation  
**Status**: ✅ Complete

### Configuration

Uses model from E001 (3 epochs, 5 files training)

```yaml
environment: PMSMEnv
n_rpm: 1000
i_d_ref: 0.0
i_q_ref: 2.0
max_steps: 500
```

### Command
```bash
python -m benchmark.run_benchmark
```

### Results

| Controller | Final Error | RMSE | Time in Target | Status |
|------------|-------------|------|----------------|--------|
| PI Baseline | 0.00 mA | ~0 A | 453/500 | PASS |
| SNN (E001) | 262,985 mA | 215.9 A | 0/500 | Needs training |

| Sparsity Metric | Value |
|-----------------|-------|
| Hidden Layer 0 | 80.9% |
| Hidden Layer 1 | 90.3% |

### Observations
- **Pipeline WORKS**: SNN runs in closed-loop without crashes or NaN
- **Currents exploded**: Expected with minimal training (only 3 epochs, 5 files)
- **Sparsity is excellent**: 80-90% means efficient neuromorphic execution
- **Next**: Full training should dramatically improve tracking

### Key Implementation
- `SNNControllerAgent` added to `benchmark/agents.py`
- Maintains stateful membrane potentials across timesteps
- Properly resets neuron states for new episodes

---

## E002: Full Training Baseline

**Date**: 2026-01-20  
**Goal**: Train on complete dataset to establish baseline performance  
**Status**: 🔄 Running

### Configuration

```yaml
# Model
hidden_size: 64
num_hidden_layers: 2
beta_hidden: 0.9
beta_output: 0.995

# Training
epochs: 100
batch_size: 32
learning_rate: 0.001
window_size: 100
stride: 50

# Data
max_files: null  # All files
val_split: 0.2
```

### Command
```bash
poetry run python -m snn.train --epochs 100
```

### Results

| Metric | Value |
|--------|-------|
| Final train loss | — |
| Final val loss | — |
| Best val loss | — |
| MAE (normalized) | — |
| MAE (Amps) | — |
| Training time | — |

### Observations
- [To be filled after experiment]

---

## E003: Hyperparameter Study — Hidden Size

**Date**: —  
**Goal**: Find optimal hidden layer size  
**Status**: 🔲 TODO

### Configurations Tested

| Config | hidden_size | Parameters | Val Loss | MAE |
|--------|-------------|------------|----------|-----|
| A | 32 | ~2,500 | — | — |
| B | 64 | ~4,600 | — | — |
| C | 128 | ~17,000 | — | — |
| D | 256 | ~66,000 | — | — |

### Command Template
```bash
poetry run python -m snn.train --epochs 50 --hidden_size {SIZE}
```

### Observations
- [To be filled]

---

## E004: Hyperparameter Study — Output Beta

**Date**: —  
**Goal**: Find optimal slow-leak rate for output neurons  
**Status**: 🔲 TODO

### Configurations Tested

| Config | beta_output | Time Constant | Val Loss | Closed-Loop Stable? |
|--------|-------------|---------------|----------|---------------------|
| A | 0.99 | ~10 steps | — | — |
| B | 0.995 | ~200 steps | — | — |
| C | 0.999 | ~1000 steps | — | — |
| D | 0.9999 | ~10000 steps | — | — |

### Observations
- [To be filled]
- Higher beta = better steady-state hold, slower response
- Lower beta = faster response, more drift

---

## E005: Closed-Loop Validation

**Date**: —  
**Goal**: Test trained SNN in closed-loop with PMSMEnv  
**Status**: 🔲 TODO

### Configuration
```yaml
model: snn/checkpoints/best_model.pt
environment: PMSMEnv
max_steps: 500
reference:
  i_d_ref: 0.0
  i_q_ref: 2.0
```

### Results

| Metric | PI Baseline | SNN |
|--------|-------------|-----|
| Final i_d error [A] | 0.0000 | — |
| Final i_q error [A] | 0.0000 | — |
| RMSE [A] | 0.0000 | — |
| ITAE | — | — |
| Settling time [ms] | — | — |
| Overshoot [%] | — | — |
| Steps in target | 453/500 | — |

### Step Response Plot
[To be added: comparison plot]

### Observations
- [Stability assessment]
- [Comparison to PI]
- [Issues observed]

---

## E006: Operating Point Sweep

**Date**: —  
**Goal**: Test SNN across multiple operating points  
**Status**: 🔲 TODO

### Operating Points

| Point | i_d [A] | i_q [A] | Speed [rpm] | SNN RMSE | PI RMSE |
|-------|---------|---------|-------------|----------|---------|
| Baseline | 0 | 2 | 1000 | — | 0.00 |
| Medium load | 0 | 5 | 1000 | — | 0.00 |
| High load | 0 | 8 | 1000 | — | 0.00 |
| Field weakening | -3 | 2 | 1000 | — | 0.00 |
| Low speed | 0 | 2 | 500 | — | 0.00 |
| High speed | 0 | 2 | 2500 | — | 0.00 |

---

## Experiment Template

```markdown
## EXXX: [Experiment Title]

**Date**: YYYY-MM-DD  
**Goal**: [What are we trying to learn?]  
**Status**: 🔲 TODO / 🔄 Running / ✅ Complete

### Configuration
```yaml
# Model
hidden_size: 
beta_output: 

# Training
epochs: 
batch_size: 
```

### Command
```bash
[Command to reproduce]
```

### Results
| Metric | Value |
|--------|-------|

### Observations
- [Key findings]
- [Unexpected results]
- [Next steps]
```

---

## Tips for Reproducibility

1. **Always record the random seed** — Set in train.py
2. **Save config with checkpoint** — Already done in model.save()
3. **Log git commit hash** — Know exact code version
4. **Save training curves** — history.json in checkpoints/

---

*Last Updated: 2026-01-20*
