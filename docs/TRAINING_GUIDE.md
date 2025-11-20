# Neural Network Training Guide

## Overview

This guide describes the process of training neural networks to replicate the FOC controller behavior.

## Data Preparation

### Step 1: Merge Simulation Data

```bash
cd data-preparation
python main.py data
```

This generates:
- `data/merged/merged_panel.csv` - Panel format with run IDs
- `data/merged/merged_stacked.csv` - Continuous time series

### Step 2: Prepare Training Data

```bash
python prepare_edge_impulse.py
```

Options:
- **Factor 1:** 10 kHz (original, for ANN training)
- **Factor 10:** 1 kHz (for SNN training)
- **Factor 100:** 100 Hz (for analysis)

Output files:
- `edge_impulse_data/basic_*_edge_impulse_train.csv`
- `edge_impulse_data/basic_*_edge_impulse_validation.csv`
- `edge_impulse_data/basic_*_edge_impulse_test.csv`

## ANN Training

### Model Architecture

**Recommended Architecture:**
```
Input: [i_d, i_q, n] (3 features)
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Output: [u_d, u_q] (2 outputs, Linear)
```

**Training Parameters:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 256
- Epochs: 50-100
- Validation split: 15% (by run_id)

### Expected Performance

- **Target MAE:** < 1.0 V for both u_d and u_q
- **Model size:** 10-50 KB
- **Inference time:** 1-2 ms on standard MCU

## SNN Conversion

### Approach

1. Train ANN with ReLU activations (see above)
2. Convert ANN to SNN using rate coding
3. Replace ReLU neurons with Leaky Integrate-and-Fire (LIF) neurons

### Conversion Tools

Recommended tools:
- **SNN Toolbox:** Automated ANN-to-SNN conversion
- **Norse:** PyTorch-based SNN implementation
- **snnTorch:** Rate-coded SNN conversion

### Timing Considerations

**Challenge:** Original data sampled at 10 kHz (100 μs period)

**Solutions:**
- Downsample to 1 kHz for SNN training (factor=10)
- Use neuromorphic hardware for real-time 10 kHz operation
- Hybrid approach: SNN at lower rate, ANN fallback

### Expected Performance

- **Accuracy loss:** 1-5% compared to ANN
- **Inference time:** 5-10 ms (50 simulation steps)
- **Energy efficiency:** 10-100× better on neuromorphic hardware

## Evaluation Metrics

### Primary Metrics
- Mean Absolute Error (MAE) for u_d and u_q
- Maximum error
- R² score

### Secondary Metrics
- Inference latency
- Model size
- Energy consumption (if neuromorphic hardware available)

## Deployment

### Target Platforms

**ANN:**
- Standard microcontrollers (STM32, ESP32, nRF52)
- Edge AI accelerators (Coral TPU, Intel Neural Compute Stick)

**SNN:**
- Neuromorphic hardware (Intel Loihi, BrainScaleS, SpiNNaker)
- Software simulation on standard MCU

### Integration Considerations

1. **Real-time constraints:** 10 kHz control loop requires < 100 μs inference
2. **Safety limits:** Clip outputs to ±V_max (±27.7 V)
3. **Fallback mechanism:** Traditional PI controller as backup
4. **Monitoring:** Watchdog for model anomalies

## References

- Field-Oriented Control fundamentals
- Neural network function approximation
- Spiking neural network conversion methods
- Neuromorphic computing architectures

