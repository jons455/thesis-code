"""Wrapper for ANN controllers to match TensorController protocol."""

from __future__ import annotations

from typing import Any

import torch

from embark.benchmark.interfaces import TensorController


class ANNControllerWrapper(TensorController):
    """Wrap a PyTorch ANN model for the benchmark harness."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass
