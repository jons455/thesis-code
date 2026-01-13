# Edge Impulse Training Guide

## Overview

This guide describes the process of training neural network models using Edge Impulse Studio to replicate the FOC controller behavior.

## Prerequisites

1. Edge Impulse account (free at https://studio.edgeimpulse.com/)
2. Prepared training data (see `data-preparation/prepare_edge_impulse.py`)
3. Basic understanding of neural network regression

## Data Preparation

### Step 1: Generate Training Data

```bash
cd data-preparation
python prepare_edge_impulse.py
```

Select downsampling factor:
- **Factor 1:** 10 kHz (original, for ANN training)
- **Factor 10:** 1 kHz (for SNN training)

Output files will be created in `edge_impulse_data/`:
- `basic_*_edge_impulse_train.csv`
- `basic_*_edge_impulse_validation.csv`
- `basic_*_edge_impulse_test.csv`

### Step 2: Data Format Verification

Ensure CSV files have the following structure:
```csv
i_d,i_q,n,u_d,u_q
0.0,0.0,1000,0.0,5.31
-0.007,-0.367,1000,0.204,7.91
...
```

- **Features (inputs):** `i_d`, `i_q`, `n`
- **Targets (outputs):** `u_d`, `u_q`

## Edge Impulse Project Setup

### 1. Create New Project

1. Go to https://studio.edgeimpulse.com/
2. Click "Create new project"
3. Name: "PMSM-FOC-Controller"
4. Project type: **Regression**

### 2. Upload Data

1. Navigate to **Data acquisition** tab
2. Click **Upload data**
3. Upload `basic_*_edge_impulse_train.csv`
   - **Label field:** Leave empty or use "training" (this is just a category label, not the regression targets)
   - **Category:** Select "Training"
4. Upload `basic_*_edge_impulse_validation.csv`
   - **Label field:** Leave empty or use "validation"
   - **Category:** Select "Testing"

**Important:** The "Label" field during upload is just for organizing samples. The actual regression targets (`u_d`, `u_q`) are configured later in Impulse Design.

### 3. Configure Regression Labels (CRITICAL STEP)

**Problem:** Edge Impulse may show only 1 output instead of 2. The Regression block may not be editable if labels weren't configured during upload.

**Solution: Use CSV Transformations during upload (RECOMMENDED)**

**IMPORTANT:** Labels must be defined **during CSV upload**, not after. If you've already uploaded files, delete them and re-upload with transformations.

**Step-by-step:**

1. **Delete existing uploads (if any):**
   - Go to **Data acquisition** tab
   - Click three dots next to each CSV file → **Delete**

2. **Upload with CSV Transformations:**
   - Click **"Upload data"**
   - Select your CSV file (`basic_*_edge_impulse_train.csv`)
   - **Click "Use CSV Transformations"** or **"Transform CSV"** button (appears after file selection)
   - In the transformation wizard:
     - **Features (Inputs):** Select `timestamp`, `i_d`, `i_q`, `n`
     - **Labels (Outputs):** Select `u_d`, `u_q`
   - Click **"Apply transformation"** or **"Import"**
   - Repeat for validation and test files

3. **Verify in Impulse Design:**
   - Go to **Impulse design** tab
   - **Flatten block:** Should show only `i_d`, `i_q`, `n` as inputs
   - **Regression block:** Should show **"2" outputs** (not "1")
   - If still showing "1", see troubleshooting below

**Alternative: If CSV Transformations button is not visible:**

1. **During upload:**
   - **Label field:** Leave empty or enter "regression"
   - **Category:** Select "Training" / "Testing"
   
2. **After upload:**
   - Go to **Data acquisition** → Click on your uploaded file
   - Click three dots → **"Edit labels"** or **"Configure"**
   - Select `u_d` and `u_q` as **Label columns**
   - Go back to **Impulse design** → Should now show 2 outputs

**Troubleshooting:**

- **Regression block still shows "1 output":**
  - Delete all uploaded files and re-upload with CSV Transformations
  - Ensure `u_d` and `u_q` columns contain numeric values (no NaN, no text)
  - Check that CSV file has correct column headers: `timestamp,i_d,i_q,n,u_d,u_q`

## Model Architecture

### Impulse Design

1. Go to **Impulse design** tab
2. Add processing block:
   - Type: **Flatten** or **Raw Data**
   - Window size: 1 (instantaneous mapping)
3. Add learning block:
   - Type: **Regression**
4. Click **Save Impulse**

### Neural Network Configuration

Navigate to **Regression** tab under Learning blocks.

**Recommended Architecture:**
```
Input Layer: 3 neurons (i_d, i_q, n)
  ↓
Dense Layer: 64 neurons, ReLU activation
  ↓
Dense Layer: 32 neurons, ReLU activation
  ↓
Dense Layer: 16 neurons, ReLU activation
  ↓
Output Layer: 2 neurons (u_d, u_q), Linear activation
```

**Training Parameters:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001 (or auto)
- Batch size: 256
- Number of epochs: 50-100
- Validation split: Use uploaded validation set

**Advanced Settings:**
- Early stopping: Enabled (patience: 10 epochs)
- Dropout: 0.1-0.2 (optional, prevents overfitting)
- Data augmentation: Disabled (not applicable for regression)

## Training Process

### 1. Feature Generation

1. Go to **Flatten** (or Raw Data) tab
2. Click **Generate features**
3. Wait for processing to complete
4. Review feature explorer for data distribution

### 2. Model Training

1. Go to **Regression** tab
2. Configure architecture (see above)
3. Click **Start training**
4. Monitor training progress:
   - Training loss should decrease
   - Validation loss should follow training loss
   - Watch for overfitting (validation loss increasing)

### 3. Expected Performance

**Target Metrics:**
- Mean Absolute Error (MAE) < 1.0 V for both u_d and u_q
- R² score > 0.95
- Maximum error < 5.0 V

**Training Time:**
- Typically 10-30 minutes for 1.4M training samples
- Depends on model size and hardware

## Model Evaluation

### 1. Test Set Evaluation

1. Go to **Model testing** tab
2. Click **Classify all**
3. Review test set performance:
   - MAE for u_d and u_q separately
   - Error distribution
   - Outliers analysis

### 2. Performance Analysis

**Good Performance Indicators:**
- MAE < 1.0 V on both train and validation sets
- Predictions within ±28 V (motor voltage limits)
- No systematic bias (mean error ≈ 0)

**Warning Signs:**
- Validation MAE >> Training MAE → Overfitting
  - Solution: Add dropout, reduce model size
- High MAE on both sets → Underfitting
  - Solution: Increase model capacity, train longer
- Predictions outside voltage limits → Model divergence
  - Solution: Check data normalization, add output clipping

## Model Deployment

### 1. Export Options

1. Go to **Deployment** tab
2. Select target platform:
   - **TensorFlow Lite** (recommended for MCU)
   - **Keras** (for further processing)
   - **ONNX** (for cross-platform compatibility)

### 2. Optimization

**Quantization:**
- Enable **INT8 quantization** for smaller model size
- Reduces model size by ~4×
- Minimal accuracy loss (< 5%)

**Model Size:**
- Expected: 10-50 KB (uncompressed)
- With INT8: 5-20 KB
- Suitable for most 32-bit microcontrollers

### 3. Download

1. Click **Build** to generate deployment package
2. Download model files
3. Extract and review:
   - Model file (`.tflite` or `.h5`)
   - C++ library (if selected)
   - Example code

## Integration Example

### C++ Integration (TensorFlow Lite)

```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

// Initialize model
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::ops::builtin::BuiltinOpResolver resolver;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Prepare input
float input[3] = {i_d, i_q, n};
interpreter->typed_input_tensor<float>(0)[0] = input[0];
interpreter->typed_input_tensor<float>(0)[1] = input[1];
interpreter->typed_input_tensor<float>(0)[2] = input[2];

// Run inference
interpreter->Invoke();

// Get output
float u_d = interpreter->typed_output_tensor<float>(0)[0];
float u_q = interpreter->typed_output_tensor<float>(0)[1];

// Apply voltage limits
u_d = constrain(u_d, -V_max, V_max);
u_q = constrain(u_q, -V_max, V_max);
```

## Troubleshooting

### High Training Error (MAE > 2V)

**Possible causes:**
- Model too simple → Increase neurons or layers
- Underfitting → Train for more epochs
- Wrong loss function → Ensure MSE is used
- Learning rate too high → Reduce to 0.0001

**Solutions:**
- Increase model capacity (64→128 neurons)
- Train for more epochs (100+)
- Check data loading (verify column order)

### Overfitting

**Symptoms:** Training MAE << Validation MAE

**Solutions:**
- Add dropout (0.1-0.2)
- Reduce model size
- Increase training data
- Early stopping (already enabled)

### Model Size Too Large

**Solutions:**
- Enable INT8 quantization
- Reduce model size (32-16-8 instead of 64-32-16)
- Use model pruning (advanced)

## Performance Benchmarks

### Expected Results

| Metric | Target | Excellent | Good | Acceptable |
|--------|--------|-----------|------|------------|
| MAE (u_d) | < 1.0 V | < 0.5 V | < 0.8 V | < 1.5 V |
| MAE (u_q) | < 1.0 V | < 0.5 V | < 0.8 V | < 1.5 V |
| R² Score | > 0.95 | > 0.98 | > 0.95 | > 0.90 |
| Model Size | < 50 KB | < 20 KB | < 35 KB | < 50 KB |

### Inference Performance

**On ARM Cortex-M4 @ 100 MHz:**
- Expected latency: 1-2 ms
- Required: < 100 μs for 10 kHz control
- **Note:** May require optimization or hardware acceleration

## References

- Edge Impulse Documentation: https://docs.edgeimpulse.com/
- TensorFlow Lite: https://www.tensorflow.org/lite
- Neural Network Regression: Standard machine learning practice

