# Edge Impulse Training Guide: PMSM FOC Controller

## Overview

This guide explains how to train a neural network model in Edge Impulse to mimic a Field-Oriented Control (FOC) controller for a Permanent Magnet Synchronous Motor (PMSM). The model learns to map motor measurements to voltage commands, effectively replicating the controller behavior for edge deployment.

---

## 🎯 What Are We Modeling?

### Controller Function
The FOC controller implements a PI-based current control loop that:
- **Reads:** Motor currents (`i_d`, `i_q`) and speed (`n`)
- **Outputs:** Voltage commands (`u_d`, `u_q`) to control the motor

### Why This Matters
Training a neural network to mimic this controller allows:
- **Edge deployment** on microcontrollers without complex control algorithms
- **Faster inference** than traditional PI controllers in some cases
- **Learned behavior** that adapts to the specific motor characteristics
- **Reduced code complexity** for embedded systems

---

## 📊 Data Generation Details

### MATLAB Simulation Setup

The data was generated using MATLAB/Simulink with the following parameters:

#### Motor Specifications
| Parameter | Value | Description |
|-----------|-------|-------------|
| **I_nominal** | 4.2 A | Nominal operating current |
| **I_max** | 10.8 A | Maximum current limit |
| **V_DC** | 48 V | DC bus voltage |
| **V_max** | ~27.7 V | Maximum voltage (with SVPWM) |
| **n_nominal** | 3000 RPM | Nominal speed |
| **L_d** | 1.13 mH | Direct-axis inductance |
| **L_q** | 1.42 mH | Quadrature-axis inductance |
| **R_s** | 0.543 Ω | Stator resistance |
| **Pole pairs** | 3 | Number of pole pairs |
| **Ψ_PM** | 16.9 mWb | Permanent magnet flux linkage |
| **Sampling time** | 100 μs | 10 kHz control frequency |

#### Simulation Parameters
- **Number of runs:** 1000 simulations
- **Duration per run:** 0.2 seconds (200 ms)
- **Samples per run:** 2001 timesteps
- **Total samples:** 2,001,000 data points

#### **🔑 Critical: Randomized Operating Points**

Each of the 1000 simulation runs uses **random setpoints** to ensure diverse training data:

```matlab
id_ref = rand * I_nenn;  % Random d-axis current: 0 to 4.2 A
iq_ref = rand * I_nenn;  % Random q-axis current: 0 to 4.2 A
n_ref  = rand * n_nenn;  % Random speed: 0 to 3000 RPM
```

This randomization means:
- ✅ **1000 different operating conditions** (combinations of current and speed setpoints)
- ✅ **Full coverage** of the motor's operating envelope
- ✅ **Diverse controller responses** to different targets
- ✅ **Excellent training data** for neural network generalization

### Data Structure

#### Available Files
```
data-preperation/data/merged/
├── merged_panel.csv       # All runs with run_id (2M rows × 7 cols)
├── merged_panel.parquet   # Same data, compressed format
├── merged_stacked.csv     # Continuous time series
└── merged_stacked.parquet # Same data, compressed format
```

#### Data Schema

**Panel format (recommended for Edge Impulse):**
```csv
run_id,time,i_d,i_q,n,u_d,u_q
1,0.0000,0.000,-0.367,1000,0.204,7.911
1,0.0001,-0.007,-0.353,1000,0.276,7.905
...
1000,0.1999,-5.0,5.0,1000,-4.946,6.249
```

| Column | Type | Role | Range | Description |
|--------|------|------|-------|-------------|
| `run_id` | int | Metadata | 1-1000 | Simulation run identifier |
| `time` | float | Metadata | 0.0-0.2 | Time in seconds |
| **`i_d`** | **float** | **Input** | -10.8 to 10.8 A | Direct-axis current (measured) |
| **`i_q`** | **float** | **Input** | -10.8 to 10.8 A | Quadrature-axis current (measured) |
| **`n`** | **int** | **Input** | 0-3000 RPM | Rotational speed (measured) |
| **`u_d`** | **float** | **Output** | ~-28 to 28 V | Direct-axis voltage (controller output) |
| **`u_q`** | **float** | **Output** | ~-28 to 28 V | Quadrature-axis voltage (controller output) |

---

## 🔧 Edge Impulse Training Strategy

### Model Type: **Regression**

This is a **multi-output regression problem**, not classification:
- **Inputs:** 3 features (i_d, i_q, n)
- **Outputs:** 2 continuous values (u_d, u_q)

### Recommended Architecture

```
┌─────────────────────────────┐
│  Input Layer: [i_d, i_q, n] │
│         (3 features)         │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Dense Layer (64 neurons)   │
│      Activation: ReLU        │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Dense Layer (32 neurons)   │
│      Activation: ReLU        │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Dense Layer (16 neurons)   │
│      Activation: ReLU        │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│  Output Layer: [u_d, u_q]    │
│      Activation: Linear      │
│         (2 outputs)          │
└──────────────────────────────┘
```

**Architecture Rationale:**
- **3 → 64 → 32 → 16 → 2** gives enough capacity to learn the FOC mapping
- **ReLU activations** for hidden layers (standard for regression)
- **Linear output** layer for continuous voltage predictions
- ~3,000-5,000 parameters (lightweight for edge deployment)

---

## 📋 Step-by-Step Edge Impulse Setup

### Step 1: Data Preparation

#### Option A: Use Full Panel Data
```python
import pandas as pd

# Load the merged data
df = pd.read_parquet('data-preperation/data/merged/merged_panel.parquet')
# or
df = pd.read_csv('data-preperation/data/merged/merged_panel.csv')

# Select only the features and targets
df_model = df[['i_d', 'i_q', 'n', 'u_d', 'u_q']].copy()

# Export for Edge Impulse
df_model.to_csv('edge_impulse_data.csv', index=False)
```

#### Option B: Add Time Windows (for temporal patterns)
```python
# Create sliding windows to capture controller dynamics
window_size = 5  # Use last 5 samples (0.5ms of history)

def create_windows(df, window_size=5):
    features = ['i_d', 'i_q', 'n']
    targets = ['u_d', 'u_q']
    
    windows = []
    for run_id in df['run_id'].unique():
        run_data = df[df['run_id'] == run_id].reset_index(drop=True)
        
        for i in range(window_size, len(run_data)):
            # Get window of inputs
            window = run_data.loc[i-window_size:i-1, features].values.flatten()
            # Get current target
            target = run_data.loc[i, targets].values
            
            windows.append(list(window) + list(target))
    
    # Create column names
    cols = []
    for t in range(window_size):
        for f in features:
            cols.append(f'{f}_t_{t}')
    cols.extend(targets)
    
    return pd.DataFrame(windows, columns=cols)

df_windowed = create_windows(df)
df_windowed.to_csv('edge_impulse_windowed.csv', index=False)
```

#### Recommended: Use Simple Instantaneous Mapping (Option A)
For this FOC controller, the instantaneous mapping works well because:
- Controllers are typically memoryless (PI control at each timestep)
- Simpler model trains faster and deploys more easily
- The 1000 diverse runs already provide rich training data

### Step 2: Data Split Strategy

**Recommended split:**
```python
# Split by run_id to ensure no data leakage
train_runs = range(1, 701)       # 70% - Runs 1-700
val_runs = range(701, 851)       # 15% - Runs 701-850
test_runs = range(851, 1001)     # 15% - Runs 851-1000

df_train = df[df['run_id'].isin(train_runs)]
df_val = df[df['run_id'].isin(val_runs)]
df_test = df[df['run_id'].isin(test_runs)]

# Export
df_train[['i_d', 'i_q', 'n', 'u_d', 'u_q']].to_csv('train.csv', index=False)
df_val[['i_d', 'i_q', 'n', 'u_d', 'u_q']].to_csv('validation.csv', index=False)
df_test[['i_d', 'i_q', 'n', 'u_d', 'u_q']].to_csv('test.csv', index=False)
```

**Why split by runs?**
- Prevents the model from memorizing specific trajectories
- Tests generalization to unseen operating conditions
- Each run has a unique setpoint combination

### Step 3: Create Edge Impulse Project

1. **Go to Edge Impulse Studio:** https://studio.edgeimpulse.com/
2. **Create new project:** "PMSM-FOC-Controller"
3. **Select project type:** Regression

### Step 4: Data Upload

#### Upload via CSV:
1. Go to **Data acquisition** tab
2. Click **Upload data**
3. Upload `train.csv` - label as "Training"
4. Upload `validation.csv` - label as "Testing"
5. Set **Label** to "controller_data" (or any name)

#### Configure data format:
- **Data type:** Tabular data / CSV
- **Features:** `i_d`, `i_q`, `n`
- **Labels:** `u_d`, `u_q` (regression targets)

### Step 5: Create Impulse

1. Go to **Impulse design** tab
2. Add **Processing block:**
   - Select **"Flatten"** or **"Raw Data"** (for tabular data)
   - Window size: 1 (no time windows for instantaneous mapping)
3. Add **Learning block:**
   - Select **"Regression"**
4. Click **Save Impulse**

### Step 6: Feature Generation

1. Go to **Flatten** (or Raw Data) tab
2. Click **Generate features**
3. Wait for processing
4. Verify feature explorer shows good distribution

### Step 7: Configure Neural Network

1. Go to **Regression** tab under Learning blocks
2. **Configure architecture:**

```
Input layer (3): [i_d, i_q, n]
  ↓
Dense layer: 64 neurons, ReLU activation
  ↓
Dense layer: 32 neurons, ReLU activation
  ↓
Dense layer: 16 neurons, ReLU activation
  ↓
Output layer (2): [u_d, u_q], Linear activation
```

3. **Training settings:**
   - **Number of epochs:** 50-100
   - **Learning rate:** 0.001 (or use auto)
   - **Batch size:** 256 or 512
   - **Validation split:** 20% (or use uploaded validation set)
   - **Loss function:** Mean Squared Error (MSE)
   - **Metrics:** MAE (Mean Absolute Error)

4. **Advanced settings:**
   - **Data augmentation:** OFF (not needed for regression)
   - **Early stopping:** ON (patience: 10 epochs)
   - **Dropout:** 0.1-0.2 (optional, prevents overfitting)

### Step 8: Train Model

1. Click **Start training**
2. Monitor training progress
3. **Target metrics:**
   - **MAE < 0.5 V** (excellent)
   - **MAE < 1.0 V** (good)
   - **MAE < 2.0 V** (acceptable)
   - Higher error may indicate need for more neurons or epochs

### Step 9: Validate Performance

1. Go to **Model testing** tab
2. Click **Classify all**
3. Check test set performance:
   - Look at MAE for u_d and u_q separately
   - Check for any systematic bias (should be near zero)
   - Verify predictions are within motor voltage limits (±28V)

4. **Analyze results:**
   ```python
   # If you download test results
   import pandas as pd
   results = pd.read_csv('test_results.csv')
   
   # Calculate errors
   results['error_u_d'] = results['u_d_pred'] - results['u_d_true']
   results['error_u_q'] = results['u_q_pred'] - results['u_q_true']
   
   print("Mean Absolute Error:")
   print(f"u_d: {results['error_u_d'].abs().mean():.3f} V")
   print(f"u_q: {results['error_u_q'].abs().mean():.3f} V")
   
   # Plot predictions vs actual
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12, 5))
   
   plt.subplot(1, 2, 1)
   plt.scatter(results['u_d_true'], results['u_d_pred'], alpha=0.1)
   plt.plot([-28, 28], [-28, 28], 'r--', label='Perfect prediction')
   plt.xlabel('True u_d (V)')
   plt.ylabel('Predicted u_d (V)')
   plt.title('Direct-axis Voltage')
   plt.legend()
   
   plt.subplot(1, 2, 2)
   plt.scatter(results['u_q_true'], results['u_q_pred'], alpha=0.1)
   plt.plot([-28, 28], [-28, 28], 'r--', label='Perfect prediction')
   plt.xlabel('True u_q (V)')
   plt.ylabel('Predicted u_q (V)')
   plt.title('Quadrature-axis Voltage')
   plt.legend()
   
   plt.tight_layout()
   plt.show()
   ```

### Step 10: Deploy Model

1. Go to **Deployment** tab
2. Select target platform:
   - **Arduino library** (for Arduino boards)
   - **C++ library** (for custom embedded systems)
   - **TensorFlow Lite** (for various platforms)
3. **Optimize model:**
   - Enable **INT8 quantization** for smaller size and faster inference
   - Check memory requirements fit your target MCU
4. Click **Build** to download deployment package

---

## 🎛️ Advanced Topics

### Adding Current Error Signals

The controller is trying to reach setpoints. Adding error signals can improve performance:

```python
# Note: You need to extract setpoints from MATLAB if saved
# For now, we can approximate or regenerate

df['i_d_error'] = df['i_d_setpoint'] - df['i_d']  # If setpoints available
df['i_q_error'] = df['i_q_setpoint'] - df['i_q']

# Train with 5 inputs: i_d, i_q, n, i_d_error, i_q_error
```

### Time-Series Window Approach

If instantaneous mapping doesn't work well, try time windows:

**Edge Impulse setup:**
1. **Window size:** 10 samples (1 ms of history)
2. **Window increase:** 1 sample (sliding window)
3. **Features:** i_d, i_q, n (3 × 10 = 30 features)
4. **Model:** 1D CNN or Dense network

**Architecture for windowed data:**
```
Input: [i_d×10, i_q×10, n×10] = 30 features
  ↓
Dense (128, ReLU)
  ↓
Dense (64, ReLU)
  ↓
Dense (32, ReLU)
  ↓
Output: [u_d, u_q] = 2 outputs
```

### Handling Multiple Speeds

Your data has varying speeds (0-3000 RPM). If the model struggles:

**Option 1: Speed-specific models**
Train separate models for speed ranges:
- Low speed: 0-1000 RPM
- Medium speed: 1000-2000 RPM
- High speed: 2000-3000 RPM

**Option 2: Speed normalization**
```python
df['n_norm'] = df['n'] / 3000.0  # Normalize to [0, 1]
```

### Hyperparameter Tuning

**If model underperforms, try:**

1. **Deeper network:**
   - Add more layers: 3 → 64 → 64 → 32 → 16 → 2

2. **Wider network:**
   - More neurons: 3 → 128 → 64 → 32 → 2

3. **Different activation:**
   - Try **tanh** or **ELU** instead of ReLU

4. **Learning rate schedule:**
   - Start with 0.001
   - Reduce by 0.5 if validation loss plateaus

5. **Regularization:**
   - Add dropout (0.1-0.3) after each dense layer
   - Add L2 regularization (0.001-0.01)

---

## 📊 Expected Performance

### Target Metrics

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| **MAE (u_d)** | < 0.3 V | < 0.8 V | < 1.5 V |
| **MAE (u_q)** | < 0.3 V | < 0.8 V | < 1.5 V |
| **Max Error** | < 2.0 V | < 4.0 V | < 6.0 V |
| **R² Score** | > 0.98 | > 0.95 | > 0.90 |

### Why This Should Work Well

1. ✅ **Deterministic system** - FOC controllers are mathematical mappings
2. ✅ **Rich training data** - 1000 diverse operating conditions
3. ✅ **Clean data** - Simulation data without measurement noise
4. ✅ **Appropriate architecture** - Neural networks excel at function approximation
5. ✅ **Sufficient samples** - 2 million data points for robust training

### Potential Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Overfitting** | Training MAE ≪ validation MAE | Add dropout, reduce model size, more training data |
| **Underfitting** | High MAE on both train and validation | Increase model capacity, train longer |
| **Edge effects** | Poor performance at extreme currents/speeds | Check data distribution, add more samples at extremes |
| **Oscillations** | Predicted voltages fluctuate unrealistically | Use time windows, add smoothing, check sampling rate |

---

## 🚀 Deployment Considerations

### Model Size
- **Expected size:** 10-50 KB (uncompressed)
- **With INT8 quantization:** 5-20 KB
- **Suitable for:** Most 32-bit microcontrollers (STM32, ESP32, nRF52, etc.)

### Inference Speed
- **Expected latency:** 1-5 ms on Cortex-M4 @ 100 MHz
- **Required frequency:** 10 kHz (100 μs per control cycle)
- **⚠️ Critical:** Verify inference time is << 100 μs on your target MCU

### Integration Tips

1. **Pre-processing:** Normalize inputs to same scale as training
2. **Post-processing:** Clip outputs to voltage limits (±V_max)
3. **Safety:** Add watchdog for model anomalies (e.g., NaN outputs)
4. **Fallback:** Keep simple PI controller as backup if model fails

### Example C++ Integration

```cpp
// Initialize model (Edge Impulse generated code)
ei_impulse_result_t result = { 0 };
signal_t signal;

// Read motor measurements
float i_d = read_current_d();  // From ADC/sensor
float i_q = read_current_q();
float n = read_speed();        // From encoder

// Prepare input buffer
float features[] = { i_d, i_q, n };
signal.total_length = 3;
signal.get_data = &get_feature_data;

// Run inference
EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

// Extract voltage commands
float u_d = result.regression.predictions[0];
float u_q = result.regression.predictions[1];

// Apply voltage limits
u_d = constrain(u_d, -V_max, V_max);
u_q = constrain(u_q, -V_max, V_max);

// Send to PWM/inverter
set_voltage_commands(u_d, u_q);
```

---

## 🔍 Troubleshooting

### Problem: High Training Error (MAE > 2V)

**Possible causes:**
1. Model too simple → Increase neurons or layers
2. Underfitting → Train for more epochs
3. Wrong loss function → Ensure MSE is used
4. Learning rate too high → Reduce to 0.0001

**Debug steps:**
- Check if loss is decreasing each epoch
- Visualize predictions on a single run
- Verify data loading correctly (check column order)

### Problem: Model Works on Validation but Fails in Real Hardware

**Possible causes:**
1. **Sim-to-real gap** - Real sensor noise not in training data
2. **Timing issues** - Inference too slow for 10 kHz control
3. **Numerical precision** - INT8 quantization too aggressive
4. **Input scaling** - Real measurements outside training range

**Solutions:**
- Add noise augmentation to training data
- Profile inference time carefully
- Use INT16 or float models if memory allows
- Ensure sensor calibration matches simulation units

### Problem: Model Diverges After Deployment

**Possible causes:**
1. Positive feedback loop (model → motor → model)
2. Accumulated numerical errors
3. Missing safety limits

**Solutions:**
- Add output clipping/saturation
- Implement state observers to detect divergence
- Use model predictions as suggestions, not absolute commands
- Blend model outputs with traditional controller (hybrid approach)

---

## 📚 Additional Resources

### Understanding FOC
- [Motor Control Fundamentals](https://www.ti.com/lit/an/sprabq7/sprabq7.pdf)
- [Field-Oriented Control Explained](https://www.mathworks.com/videos/understanding-field-oriented-control-1574010149579.html)

### Edge Impulse Documentation
- [Regression Tutorial](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/regression)
- [Model Optimization](https://docs.edgeimpulse.com/docs/edge-impulse-studio/deployment-optimization)
- [C++ SDK Guide](https://docs.edgeimpulse.com/docs/run-inference/cpp-library)

### Neural Network Training
- [Deep Learning for Control](https://arxiv.org/abs/1806.07366)
- [Learning to Control with Neural Networks](https://sites.google.com/view/icra19-neural-control)

---

## ✅ Quick Start Checklist

- [ ] Export training data from panel format (i_d, i_q, n, u_d, u_q)
- [ ] Split data by run_id (70% train, 15% validation, 15% test)
- [ ] Create Edge Impulse project (Regression type)
- [ ] Upload train and validation CSV files
- [ ] Create impulse with Flatten + Regression blocks
- [ ] Configure neural network (3 → 64 → 32 → 16 → 2)
- [ ] Train model with MSE loss (50-100 epochs)
- [ ] Validate MAE < 1.0 V on test set
- [ ] Deploy model to target platform
- [ ] Test inference speed (must be < 100 μs)
- [ ] Integrate into motor control firmware
- [ ] Add safety limits and monitoring
- [ ] Validate on real hardware

---

## 🎓 Summary

You have **excellent training data** with:
- ✅ 2 million labeled samples
- ✅ 1000 diverse operating conditions (random setpoints)
- ✅ Full coverage of motor operating range
- ✅ Clean simulation data ready for training

**Expected outcome:**
A lightweight neural network (10-50 KB) that mimics your MATLAB FOC controller with MAE < 1V, suitable for real-time deployment on microcontrollers.

**Key success factors:**
1. Use simple instantaneous mapping (i_d, i_q, n → u_d, u_q)
2. Split by run_id to ensure generalization
3. Train with 3-layer dense network (64-32-16 neurons)
4. Validate inference time meets 10 kHz control requirement
5. Add safety limits and monitoring in deployment

Good luck with your Edge Impulse training! 🚀

