"""SNN controller models for PMSM current control."""

import torch
import torch.nn as nn
from pathlib import Path

from .config import SNNConfig
from .membrane import MembraneSNNController
from .population import PopulationSNNController
from .learned_linear import LearnedLinearSNNController
from .delta import DeltaSNNController
from .ttfs import TTFSSNNController
from .recurrent import RecurrentSNNController

# Aliases for compatibility
SimpleSNNController = MembraneSNNController
SNN = MembraneSNNController


def load_snn_model(path: str, device: str = "cpu") -> nn.Module:
    """Load SNN model from checkpoint, automatically detecting type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Detect model type based on keys
    if "ttfs_out.fc.weight" in state_dict:
        model_cls = TTFSSNNController
    elif "delta_out.fc.weight" in state_dict:
        model_cls = DeltaSNNController
    elif "linear_out.decoder.weight" in state_dict:
        model_cls = LearnedLinearSNNController
    elif "pop_out.fc.weight" in state_dict:
        model_cls = PopulationSNNController
    # Distinguish between recurrent and membrane (feedforward)
    # Recurrent layers use snn.RLeaky which has different param names often?
    # snntorch RLeaky has params like 'recurrent.weight' if internal linear is used?
    # Or just check for the specific structure of RecurrentSNNController.
    # Our RecurrentSNNController uses RLeaky but doesn't have a unique output layer name
    # compared to Membrane (both use fc_out/lif_out).
    # However, hidden layers in Recurrent are RLeaky.
    # Let's check if the state dict implies recurrence.
    # Actually, RLeaky usually registers recurrent weights.
    # But for now, since RecurrentSNNController is the only one using RLeaky,
    # we can try to infer or fallback.
    # Alternatively, config might be in checkpoint.
    elif (
        checkpoint.get("config")
        and hasattr(checkpoint["config"], "model_type")
        and checkpoint["config"].model_type == "recurrent"
    ):
        model_cls = RecurrentSNNController
    elif "fc_out.weight" in state_dict:
        # Default to Membrane if it looks like standard structure
        model_cls = MembraneSNNController
    else:
        # Fallback
        print("Warning: Unknown model structure, trying MembraneSNNController")
        model_cls = MembraneSNNController

    model = model_cls(config=checkpoint.get("config"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


__all__ = [
    "SNNConfig",
    "MembraneSNNController",
    "PopulationSNNController",
    "LearnedLinearSNNController",
    "DeltaSNNController",
    "TTFSSNNController",
    "RecurrentSNNController",
    "SimpleSNNController",
    "SNN",
    "load_snn_model",
]
