## E007: TTFS Training (Dead Neuron Fix)

**Date**: 2026-01-27
**Goal**: Verify TTFS model training with boosted weight initialization
**Status**: 🔄 Running

### Issue
Initial training showed constant loss (~1.04) and no learning.
**Diagnosis**: "Dead Neuron" problem. Sparse spiking (1 spike/neuron/window) caused gradients to vanish.

### Fix
Implemented **4x gain boost** on Kaiming initialization in `ttfs.py` to ensure signal propagation.

### Configuration
```yaml
model_type: ttfs
hidden_size: 64
ttfs_time_window: 10  # Reduced from 20 for speed
weight_gain: 4.0      # Custom boost
epochs: 20
```

### Command
```bash
poetry run python evaluation/snn/utils/train.py --model_type ttfs --epochs 20 --device cuda --ttfs_time_window 10
```

### Initial Results (Epoch 1-17)
| Epoch | Train Loss | Val Loss | MAE |
|-------|------------|----------|-----|
| 1 | 0.0104 | 0.0044 | 0.0467 |
| 2 | 0.0039 | 0.0027 | 0.0403 |
| ... | ... | ... | ... |
| 7 | 0.0034 | 0.0034 | 0.0397 |
| 17 | 0.0033 | 0.0033 | 0.0393 |

### Observations
- **Fix Verified**: Loss is decreasing significantly (order of magnitude improvement).
- **Convergence**: Major gains happen in first 5-7 epochs. From epoch 7 to 17, improvement is negligible (<1%).
- **Decision**: **Reduce training to 10 epochs** for all future SNN runs to save compute time.
- **Performance**: Training is slow (~30 mins/epoch) due to nested time loop (1000 steps/sample).

---

## E004: Full Training with Membrane SNN (Planned)
