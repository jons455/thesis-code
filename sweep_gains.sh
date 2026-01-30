#!/bin/bash
set -e

echo "Starting Final Sweep..."

# === 1. The Safe Bet (Gain 5) ===
# Expected: Smoother than your current plot, still fast.
echo "Running Linear Gain 5.0..."
poetry run python evaluation/snn/utils/train.py \
    --model_type learned_linear \
    --epochs 30 \
    --device cuda \
    --error_gain 5.0 \
    --run_name linear_gain_5

# === 2. The Smooth Operator (Gain 2) ===
# Expected: Very thin/smooth lines, maybe slightly slower rise time.
echo "Running Linear Gain 2.0..."
poetry run python evaluation/snn/utils/train.py \
    --model_type learned_linear \
    --epochs 30 \
    --device cuda \
    --error_gain 2.0 \
    --run_name linear_gain_2

# === 3. The Aggressive One (Gain 10 - Baseline) ===
# Retraining this for 30 epochs to see if the noise reduces with time.
echo "Running Linear Gain 10.0..."
poetry run python evaluation/snn/utils/train.py \
    --model_type learned_linear \
    --epochs 30 \
    --device cuda \
    --error_gain 10.0 \
    --run_name linear_gain_10

# === 4. The Redemption Arc (Population with LOW Gain) ===
# Giving Population one last chance. Gain=10 saturated it. Gain=2 might fix it.
echo "Running Population Gain 2.0..."
poetry run python evaluation/snn/utils/train.py \
    --model_type population \
    --epochs 30 \
    --device cuda \
    --neurons_per_output 100 \
    --error_gain 2.0 \
    --run_name pop_gain_2_redemption

# === 5. The Delta Redemption (Delta with LOW Gain) ===
# Low gain (2.0) helps Delta ignore noise (jitter).
echo "Running Delta Gain 2.0..."
poetry run python evaluation/snn/utils/train.py \
    --model_type delta \
    --epochs 30 \
    --device cuda \
    --delta_scale 0.005 \
    --error_gain 2.0 \
    --run_name delta_gain_2

echo "Sweep Complete."
