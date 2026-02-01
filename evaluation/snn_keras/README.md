# Keras/Akida SNN Models for PMSM Control

This module provides Keras implementations of PMSM controllers that are compatible with BrainChip's **Akida 1.0** neuromorphic processor.

## Overview

The PyTorch SNN models in `evaluation/snn/` use explicit spiking neurons (LIF) during training. For Akida deployment, we use a different approach:

1. **Training**: Standard Keras Dense layers with ReLU activations
2. **Quantization**: Convert float32 weights to 4-bit integers using `quantizeml`
3. **Conversion**: Transform to Akida SNN format using `cnn2snn`
4. **Deployment**: Run on Akida hardware (Raspberry Pi Kit)

The hardware handles the spiking neuron dynamics - you don't simulate them in software during training.

## Akida 1.0 Constraints

**These rules are critical - violating them will prevent deployment on the chip:**

| Rule | Constraint |
|------|------------|
| **Activations** | ReLU only (no Tanh, Sigmoid, LeakyReLU) |
| **Layers** | Dense, Conv2D (no LSTM, GRU, or RNNs) |
| **Weights** | 4-bit quantized integers |
| **Outputs** | Quantized integers (scale in post-processing) |

## Installation

### Prerequisites

- **Python 3.11** (Recommended for Akida 2.18+ and TensorFlow 2.19+)
- **Windows 10/11** or Linux

### Setup

Since Poetry's dependency resolver can be strict with TensorFlow versions on Windows, we recommend installing the Akida stack using `pip` inside the poetry environment:

```bash
# 1. Set up the base environment with Poetry
poetry env use 3.11
poetry install

# 2. Install Akida and TensorFlow via pip (bypassing strict lock checks)
poetry run pip install akida==2.18.2 cnn2snn==2.18.2 akida-models==1.12.0
```

This installs:
- `akida` 2.18.2
- `cnn2snn` 2.18.2
- `akida-models` 1.12.0
- `tensorflow` ~2.19.1

### Verification

Run the following command to verify installation:

```bash
poetry run python -c "import akida; import cnn2snn; import tensorflow as tf; print(f'Akida: {akida.__version__}, TF: {tf.__version__}')"
```

### Troubleshooting Windows Installation

1. **"File name too long" / `FileNotFoundError`**:
   TensorFlow has deeply nested file paths that exceed the Windows 260-character limit.
   - Enable "Long Paths" in Windows Registry or Group Policy.
   - Or use WSL2 (Windows Subsystem for Linux).

2. **"Unable to find installation candidates for tensorflow-io-gcs-filesystem"**:
   This occurs when using Python 3.11 with older TensorFlow versions (like 2.10).
   - **Solution**: Downgrade to Python 3.10.

3. **"Model will not run on chip"**:
   - Ensure you use `ReLU` activations only.
   - Ensure you use `quantizeml` to quantize to 4-bit.

### Manual Installation (pip)

If Poetry fails due to path issues, you can try pip inside the virtual environment:

```bash
poetry shell
pip install akida akida-models cnn2snn quantizeml tensorflow
```

## Model Architecture

The `AkidaController` mirrors the PyTorch `LearnedLinearSNNController`:

```
Input (5 features)
    ↓
Dense(64) + ReLU    ─────────────────────┐
    ↓                                     │  Hidden layers
Dense(64) + ReLU    ─────────────────────┘  (configurable)
    ↓
Dense(2) + Linear   ← Output (u_d, u_q)
```

**Input features** (5 total):
- `i_d`: Normalized d-axis current
- `i_q`: Normalized q-axis current  
- `e_d`: Amplified d-axis error (reference - actual)
- `e_q`: Amplified q-axis error
- `n`: Normalized motor speed (RPM)

**Outputs** (2 total):
- `u_d`: Normalized d-axis voltage command
- `u_q`: Normalized q-axis voltage command

## Quick Start

### Training

```bash
# Full training with Akida export
python -m evaluation.snn_keras.utils.train \
    --data_dir data/raw/train \
    --epochs 100 \
    --hidden_sizes 64 64

# Quick test (limited data)
python -m evaluation.snn_keras.utils.train \
    --data_dir data/raw/train \
    --max_files 5 \
    --epochs 10 \
    --no_akida
```

### Python API

```python
from evaluation.snn_keras import AkidaConfig, AkidaController
from evaluation.snn_keras.utils.dataset import PMSMKerasDataset

# Load data
dataset = PMSMKerasDataset("data/raw/train")
x_train, y_train = dataset.get_flattened_arrays()

# Create model
config = AkidaConfig(hidden_sizes=[64, 64])
controller = AkidaController(config=config)

# Train
controller.compile(optimizer="adamw", loss="mse")
controller.fit(x_train, y_train, epochs=100, validation_split=0.2)

# Export to Akida
controller.export_akida("models/akida_model.fbz", calibration_data=x_train[:1000])
```

### Manual Quantization & Export

```python
from evaluation.snn_keras import AkidaController

# Load trained model
controller = AkidaController.load("trained_models/akida/final_model")

# Step 1: Quantize to 4-bit
qmodel = controller.quantize(calibration_data=x_train[:100])

# Optional: Fine-tune quantized model
qmodel.compile(optimizer="adam", loss="mse")
qmodel.fit(x_train, y_train, epochs=10)

# Step 2: Convert to Akida
akida_model = controller.convert_to_akida()

# Step 3: Save for Raspberry Pi
akida_model.save("akida_model.fbz")
```

## Deployment on Raspberry Pi

Copy the `.fbz` file to your Raspberry Pi with Akida hardware:

```python
import akida
import numpy as np

# Load model
model = akida.Model("akida_model.fbz")

# Check available devices
devices = akida.devices()
print(f"Akida devices: {devices}")

# Map to hardware (if available)
if devices:
    model.map(devices[0])

# Run inference
inputs = np.array([[i_d, i_q, e_d, e_q, n]], dtype=np.float32)
outputs = model.predict(inputs)

# Scale outputs back to voltage
U_MAX = 350.0  # Your voltage limit
u_d = outputs[0, 0] * scale_factor  # Apply your scale
u_q = outputs[0, 1] * scale_factor
```

**Important**: Akida outputs quantized integers. You need to multiply by a scale factor to get the actual voltage values. The scale factor depends on your quantization settings.

## Output Scaling for Regression

Since Akida outputs 4-bit integers (range ~[-8, 7] or [0, 15] depending on quantization), you need to scale back to voltage:

```python
# During inference on Raspberry Pi
def denormalize_output(akida_output, u_max=350.0, quant_scale=0.1):
    """Convert Akida integer output to voltage.
    
    Args:
        akida_output: Integer from Akida (e.g., [3, -2])
        u_max: Maximum voltage limit
        quant_scale: Learned scale from quantization
        
    Returns:
        Voltage in physical units
    """
    # Akida output → normalized → physical
    normalized = akida_output * quant_scale
    voltage = normalized * u_max
    return voltage

# Example
akida_out = model.predict(inputs)  # e.g., [3, -2]
u_d, u_q = denormalize_output(akida_out[0])
```

## Configuration Reference

### AkidaConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 5 | Number of input features |
| `hidden_sizes` | [64, 64] | Hidden layer sizes |
| `output_size` | 2 | Number of outputs (u_d, u_q) |
| `use_batch_norm` | False | Enable batch normalization |
| `dropout_rate` | 0.0 | Dropout during training |
| `weight_bits` | 4 | Quantization bit width |
| `activation_bits` | 4 | Activation quantization |

### Training Arguments

```bash
python -m evaluation.snn_keras.utils.train --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | data/raw/train | Training data directory |
| `--epochs` | 100 | Training epochs |
| `--hidden_sizes` | 64 64 | Hidden layer sizes |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--no_akida` | False | Skip Akida export |
| `--quantize_epochs` | 10 | Fine-tuning after quantization |

## File Structure

```
evaluation/snn_keras/
├── __init__.py              # Module exports
├── README.md                # This file
├── models/
│   ├── __init__.py
│   ├── config.py            # AkidaConfig dataclass
│   └── akida_controller.py  # Main model class
└── utils/
    ├── __init__.py
    ├── dataset.py           # TensorFlow data loading
    └── train.py             # Training script
```

## Troubleshooting

### "Model will not run on chip"

Check these common issues:
1. **Wrong activation**: Only ReLU is supported
2. **Unsupported layer**: No LSTM, GRU, or custom layers
3. **Quantization failed**: Try increasing `weight_bits` to 8

### "quantizeml not found"

```bash
pip install akida-models quantizeml
```

### "cnn2snn conversion error"

Ensure your model uses only supported layers and activations. Check the Akida documentation for the full list.

### Low accuracy after quantization

1. Increase `quantize_epochs` for more fine-tuning
2. Use more calibration data
3. Try 8-bit weights instead of 4-bit

## Comparison with PyTorch SNN

| Aspect | PyTorch SNN | Keras/Akida |
|--------|-------------|-------------|
| Training | LIF neurons, surrogate gradients | Standard Dense + ReLU |
| Inference | Simulated spikes | Hardware spikes (Akida) |
| Output | Continuous voltage | Quantized integer |
| Hardware | CPU/GPU | Akida neuromorphic chip |

Both approaches produce similar functional behavior, but Akida deployment requires the quantization-aware workflow.

## References

- [Akida Documentation](https://doc.brainchipinc.com/)
- [QuantizeML Guide](https://doc.brainchipinc.com/quantizeml/)
- [CNN2SNN Conversion](https://doc.brainchipinc.com/cnn2snn/)
