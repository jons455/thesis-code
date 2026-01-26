"""
SNN Controller Module for PMSM Current Control
===============================================

This module provides spiking neural network controllers for PMSM
current control, trained via imitation learning from PI controller
trajectories.

Components:
- models: SNN architectures (SimpleSNNController)
- dataset: Data loading for PI trajectories
- train: Training utilities

Example:
    from evaluation.snn import SimpleSNNController, PMSMDataset

    # Load trained model
    model = SimpleSNNController.load("models/checkpoints/best_model.pt")

    # Inference
    state = torch.tensor([i_d, i_q, e_d, e_q])
    voltage, new_state = model(state, snn_state)
"""

from evaluation.snn.akida_export import (
    AkidaCompatibilityReport,
    PopulationCodingOutput,
    RateCodingOutput,
    export_to_nir,
    export_to_onnx,
    validate_akida_compatibility,
)
from evaluation.snn.dataset import PMSMDataset
from evaluation.snn.models import SimpleSNNController

__all__ = [
    "SimpleSNNController",
    "PMSMDataset",
    "validate_akida_compatibility",
    "export_to_nir",
    "export_to_onnx",
    "RateCodingOutput",
    "PopulationCodingOutput",
    "AkidaCompatibilityReport",
]
