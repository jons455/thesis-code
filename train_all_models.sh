#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting training for all SNN models..."

echo "----------------------------------------------------------------"
echo "Training Delta SNN..."
poetry run python evaluation/snn/utils/train.py --model_type delta --epochs 10 --device cuda --delta_scale 0.01

echo "----------------------------------------------------------------"
echo "Training Population SNN..."
poetry run python evaluation/snn/utils/train.py --model_type population --epochs 10 --device cuda --neurons_per_output 50

echo "----------------------------------------------------------------"
echo "Training Recurrent SNN..."
poetry run python evaluation/snn/utils/train.py --model_type recurrent --epochs 10 --device cuda

echo "----------------------------------------------------------------"
echo "Training Learned Linear SNN..."
poetry run python evaluation/snn/utils/train.py --model_type learned_linear --epochs 10 --device cuda

echo "----------------------------------------------------------------"
echo "Training Membrane SNN..."
poetry run python evaluation/snn/utils/train.py --model_type membrane --epochs 10 --device cuda

echo "----------------------------------------------------------------"
echo "Training TTFS SNN..."
poetry run python evaluation/snn/utils/train.py --model_type ttfs --epochs 10 --device cuda --ttfs_time_window 20

echo "----------------------------------------------------------------"
echo "All training runs completed successfully!"
