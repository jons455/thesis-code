$ErrorActionPreference = "Stop"

Write-Host "Starting training for all SNN models..."

Write-Host "----------------------------------------------------------------"
Write-Host "Training Delta SNN..."
poetry run python evaluation/snn/utils/train.py --model_type delta --epochs 10 --device cuda --delta_scale 0.01

Write-Host "----------------------------------------------------------------"
Write-Host "Training Population SNN..."
poetry run python evaluation/snn/utils/train.py --model_type population --epochs 10 --device cuda --neurons_per_output 50

Write-Host "----------------------------------------------------------------"
Write-Host "Training Recurrent SNN..."
poetry run python evaluation/snn/utils/train.py --model_type recurrent --epochs 10 --device cuda

Write-Host "----------------------------------------------------------------"
Write-Host "Training Learned Linear SNN..."
poetry run python evaluation/snn/utils/train.py --model_type learned_linear --epochs 10 --device cuda

Write-Host "----------------------------------------------------------------"
Write-Host "Training Membrane SNN..."
poetry run python evaluation/snn/utils/train.py --model_type membrane --epochs 10 --device cuda

Write-Host "----------------------------------------------------------------"
Write-Host "Training TTFS SNN..."
poetry run python evaluation/snn/utils/train.py --model_type ttfs --epochs 10 --device cuda --ttfs_time_window 20

Write-Host "----------------------------------------------------------------"
Write-Host "All training runs completed successfully!"
