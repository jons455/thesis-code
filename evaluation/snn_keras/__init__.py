"""
Keras/Akida-compatible SNN models for PMSM control.

This module provides Keras implementations of the SNN controller models
that are compatible with BrainChip's Akida neuromorphic processor.

Key differences from PyTorch SNN models:
- Uses standard Dense layers with ReLU activations (no explicit LIF neurons)
- Akida hardware handles the spiking neuron dynamics
- Quantization-aware training for 4-bit integer inference
- Export to .fbz format for Akida deployment

Example:
    from evaluation.snn_keras import AkidaController, AkidaConfig
    from evaluation.snn_keras.utils.train import train_and_export

    config = AkidaConfig(hidden_size=64)
    model = AkidaController(config=config)

    # Train and export
    train_and_export(model, data_dir="data/raw/train")

"""

from evaluation.snn_keras.models import AkidaConfig, AkidaController

__all__ = [
    "AkidaConfig",
    "AkidaController",
]
