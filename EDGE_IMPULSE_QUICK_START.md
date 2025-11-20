# Edge Impulse Quick Start - PMSM FOC Controller

## 🎯 Goal
Train a neural network to mimic a PMSM Field-Oriented Control (FOC) controller using Edge Impulse.

---

## 📊 Your Data

**What you have:**
- 2,001,000 samples from 1000 simulation runs
- Each run: 200ms @ 10kHz sampling (2001 timesteps)
- **1000 different operating conditions** (random current & speed setpoints)

**Data structure:**
```
Inputs (Features):  i_d, i_q, n        (3 values)
Outputs (Targets):  u_d, u_q           (2 values)
```

---

## ⚡ Quick Start (5 Steps)

### Step 1: Prepare Data (5 minutes)
```bash
cd data-preperation
python prepare_edge_impulse.py
```

This creates:
- `basic_edge_impulse_train.csv` (70% of runs)
- `basic_edge_impulse_validation.csv` (15% of runs)
- `basic_edge_impulse_test.csv` (15% of runs)

### Step 2: Create Edge Impulse Project (2 minutes)
1. Go to https://studio.edgeimpulse.com/
2. Create new project: "PMSM-FOC-Controller"
3. Select project type: **Regression**

### Step 3: Upload Data (3 minutes)
1. **Data acquisition** tab → **Upload data**
2. Upload `basic_edge_impulse_train.csv` → Label: "Training"
3. Upload `basic_edge_impulse_validation.csv` → Label: "Testing"
4. Verify: Features = `i_d, i_q, n` | Labels = `u_d, u_q`

### Step 4: Create Impulse (2 minutes)
1. **Impulse design** tab
2. Add processing block: **"Flatten"** or **"Raw Data"**
   - Window size: 1 (instantaneous)
3. Add learning block: **"Regression"**
4. Save Impulse

### Step 5: Train Model (10-30 minutes)
1. **Flatten** tab → Generate features
2. **Regression** tab → Configure:
   ```
   Architecture:
   Input (3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Output(2, Linear)
   
   Settings:
   - Epochs: 50-100
   - Learning rate: 0.001
   - Batch size: 256
   - Loss: Mean Squared Error (MSE)
   ```
3. Click **Start training**
4. Wait for completion

---

## 🎯 Success Criteria

| Metric | Target |
|--------|--------|
| **MAE (u_d)** | < 1.0 V |
| **MAE (u_q)** | < 1.0 V |
| **Training time** | 10-30 min |
| **Model size** | 10-50 KB |

---

## 🏗️ Model Architecture (Recommended)

```
┌────────────────┐
│ Input: 3       │  i_d, i_q, n
└───────┬────────┘
        │
┌───────▼────────┐
│ Dense: 64      │  ReLU activation
└───────┬────────┘
        │
┌───────▼────────┐
│ Dense: 32      │  ReLU activation
└───────┬────────┘
        │
┌───────▼────────┐
│ Dense: 16      │  ReLU activation
└───────┬────────┘
        │
┌───────▼────────┐
│ Output: 2      │  u_d, u_q (Linear)
└────────────────┘

Parameters: ~3,500
Memory: 10-20 KB
```

---

## 📈 What to Expect

### ✅ Good Performance Indicators:
- Training loss decreases smoothly
- Validation loss follows training loss closely
- MAE < 1.0 V on both train and validation
- Predictions within ±28V (motor voltage limits)

### ⚠️ Warning Signs:
- Validation loss much higher than training → **Overfitting**
  - Solution: Add dropout (0.1-0.2), reduce model size
- Both losses high and flat → **Underfitting**
  - Solution: Increase neurons (64→128), add layers, train longer
- Predictions outside ±28V → **Model divergence**
  - Solution: Check data normalization, add output clipping

---

## 🚀 Deployment

### After Training:
1. **Deployment** tab → Select target platform
2. Enable **INT8 quantization** (optional, for smaller size)
3. Download deployment package
4. **Critical:** Test inference time < 100 μs (10 kHz control requirement)

### Integration Example (C++):
```cpp
float features[] = { i_d, i_q, n };
run_classifier(&signal, &result, false);
float u_d = result.regression.predictions[0];
float u_q = result.regression.predictions[1];
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't upload CSV | Check format: headers must be `i_d,i_q,n,u_d,u_q` |
| High MAE (> 2V) | Increase neurons to 128, train for 100 epochs |
| Overfitting | Add dropout 0.2, reduce to 32-16-8 architecture |
| Inference too slow | Use INT8 quantization, reduce model size |

---

## 📚 Full Documentation

For detailed explanations, see:
- **`EDGE_IMPULSE_TRAINING_GUIDE.md`** - Complete training guide
- **`data-preperation/README.md`** - Data structure details
- **`pmsm_init.m`** - Original MATLAB simulation code

---

## ✅ Checklist

- [ ] Run `prepare_edge_impulse.py` to prepare data
- [ ] Create Edge Impulse project (Regression)
- [ ] Upload train and validation CSV files
- [ ] Create impulse (Flatten + Regression)
- [ ] Configure neural network (3→64→32→16→2)
- [ ] Train model (50-100 epochs)
- [ ] Verify MAE < 1.0 V
- [ ] Deploy to target platform
- [ ] Test inference speed < 100 μs
- [ ] Integrate into motor control system

---

## 💡 Pro Tips

1. **Start simple:** Use basic instantaneous mapping first
2. **Monitor training:** Stop early if validation loss increases
3. **Check predictions:** Plot predicted vs. actual voltages
4. **Test thoroughly:** Verify on unused test data before deployment
5. **Add safety:** Implement voltage clipping and watchdogs in firmware

---

## 🆘 Need Help?

- Edge Impulse Docs: https://docs.edgeimpulse.com/
- Edge Impulse Forum: https://forum.edgeimpulse.com/
- Your data is excellent - this should work well! 🎉

---

**Last Updated:** 2025
**Data:** 1000 runs × 2001 samples = 2,001,000 training points
**Success Rate:** Very High (deterministic FOC control + rich data)

