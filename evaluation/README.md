# Evaluation & Benchmarking Tools

This directory contains scripts and tools for training, validating, and benchmarking control models (PI, SNN, ANN) for the PMSM system. It includes specific workflows for BrainChip Akida hardware and Edge Impulse integration.

## 1. Core Evaluation (Main Benchmark)

The primary entry point for evaluating PyTorch-based SNN models against the PI baseline.

```bash
# Run comparison with a specific trained model
poetry run python evaluation/core/run_evaluation.py --model trained_models/linear_speed_final/best_model.pt --speed 1500 --iq-ref 3.0
```

**Key Arguments:**
- `--model`: Path to the PyTorch `.pt` checkpoint.
- `--speed`: Motor speed in RPM (default: 1000).
- `--iq-ref`: Target q-axis current in Amps (default: 5.0).
- `--id-ref`: Target d-axis current in Amps (default: 0.0).

---

## 2. Akida / Keras Workflow

Tools for developing and testing models compatible with BrainChip Akida (ANN/SNN).

### A. Training (Float32 ANN)
Train a Keras model compatible with Akida constraints (ReLU only, no bias in certain layers, etc.).

```bash
poetry run python -m evaluation.snn_keras.utils.train --data_dir data/raw/train --epochs 50 --run_name my_akida_model
```

### B. Validation (Visual Check)
Quickly visualize model predictions against a ground-truth trajectory.

```bash
# Validate Keras Float model
poetry run python -m evaluation.snn_keras.validate --model akida/final_model.keras --data data/raw/train/trajectory_0.csv

# Validate Akida .fbz (Hardware/Quantized) model
poetry run python -m evaluation.snn_keras.validate --model akida/akida_model.fbz --data data/raw/train/trajectory_0.csv
```

### C. Benchmarking (Closed-Loop)
Run the model in the closed-loop simulation harness to measure control metrics (RMSE, Settling Time, Efficiency).

**Float Model (.keras):**
```bash
poetry run python evaluation/snn_keras/run_benchmark.py --model akida/final_model.keras --speed 1500 --iq-ref 3.0
```

**Hardware Model (.fbz):**
*Note: On Windows, you may need to set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP conflicts between Torch/Akida.*

```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; poetry run python evaluation/snn_keras/run_benchmark.py --model akida/akida_model.fbz --speed 1500 --iq-ref 3.0
```

---

## 3. PyTorch SNN Workflow

Scripts for the native PyTorch SNN implementations (Surrogate Gradient).

### Compare Models
Batch compare multiple trained SNN architectures against the PI baseline.

```bash
poetry run python -m evaluation.snn.compare_models
```

---

## 4. Data Preparation

Tools to prepare simulation data for external training frameworks like Edge Impulse.

### Merge Simulation Runs
Combine multiple raw CSV output files into a single dataset.

```bash
poetry run python evaluation/edge_impulse_prep/merge_simulation_data.py --input_dir data/raw --output_dir data/processed
```

### Prepare for Edge Impulse
Format merged data into the specific JSON/CSV structure required by Edge Impulse Studio.

```bash
poetry run python evaluation/edge_impulse_prep/prepare_edge_impulse.py
```
