"""Wrapper for SNN controllers to match TensorController protocol."""

from __future__ import annotations

from typing import Any

import torch

from embark.benchmark.interfaces import TensorController


class SNNControllerWrapper(TensorController):
    """Wrap a PyTorch SNN model for the benchmark harness."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._state: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        self._state = None
        if hasattr(self.model, "reset"):
            self.model.reset()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward"):
            output = self.model(observation)
        else:
            output = self.model(observation)
        if isinstance(output, tuple) and len(output) >= 2:
            action, info = output[0], output[1]
            if isinstance(info, dict):
                self.last_info = info
            return action
        return output

    def get_state(self) -> dict[str, Any]:
        return {"state": self._state}

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = state.get("state")
