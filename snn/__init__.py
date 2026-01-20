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
    from snn import SimpleSNNController, PMSMDataset
    
    # Load trained model
    model = SimpleSNNController.load("checkpoints/best_model.pt")
    
    # Inference
    state = torch.tensor([i_d, i_q, e_d, e_q])
    voltage, new_state = model(state, snn_state)
"""

from snn.models import SimpleSNNController
from snn.dataset import PMSMDataset

__all__ = ["SimpleSNNController", "PMSMDataset"]
