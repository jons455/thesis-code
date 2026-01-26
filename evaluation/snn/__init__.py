"""
SNN Controller Module for PMSM Current Control
===============================================

This module provides spiking neural network controllers for PMSM
current control, trained via imitation learning from PI controller
trajectories.

Components:
- models: SNN architectures (Membrane, Population, LearnedLinear, Delta)
- output_layers: Decoding strategies (population, learned linear, delta)
- dataset: Data loading for PI trajectories
- train: Training utilities

Example:
    from evaluation.snn import MembraneSNNController, PMSMDataset

    # Load trained model
    model = MembraneSNNController.load("models/checkpoints/best_model.pt")

    # Inference
    state = torch.tensor([i_d, i_q, e_d, e_q])
    voltage, new_state = model(state, snn_state)
"""

from evaluation.snn.akida_export import (
    AkidaCompatibilityReport,
    export_to_nir,
    export_to_onnx,
    validate_akida_compatibility,
)
from evaluation.snn.dataset import PMSMDataset
from evaluation.snn.models import (
    DeltaSNNController,
    LearnedLinearSNNController,
    MembraneSNNController,
    PopulationSNNController,
    SimpleSNNController,
    load_snn_model,
)
from evaluation.snn.output_layers import (
    DeltaCodingOutput,
    LearnedLinearOutput,
    PopulationCodingOutput,
)

__all__ = [
    "MembraneSNNController",
    "PopulationSNNController",
    "LearnedLinearSNNController",
    "DeltaSNNController",
    "SimpleSNNController",
    "load_snn_model",
    "PMSMDataset",
    "validate_akida_compatibility",
    "export_to_nir",
    "export_to_onnx",
    "PopulationCodingOutput",
    "LearnedLinearOutput",
    "DeltaCodingOutput",
    "AkidaCompatibilityReport",
]
