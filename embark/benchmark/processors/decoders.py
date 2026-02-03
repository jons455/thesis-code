"""Action decoding processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from embark.benchmark.interfaces import ActionDict, ActionProcessor, SystemConfig


@dataclass
class LinearActionProcessor(ActionProcessor):
    """Scale normalized action tensor to physical units."""

    output_keys: Sequence[str]
    bounds: dict[str, tuple[float, float]]

    def configure(self, physics_config: SystemConfig) -> None:  # noqa: ARG002
        pass

    def __call__(self, action: torch.Tensor, physics_config: SystemConfig) -> ActionDict:  # noqa: ARG002
        action = action.detach().cpu().flatten().tolist()
        if len(action) < len(self.output_keys):
            raise ValueError("Action tensor smaller than number of output keys.")
        output: ActionDict = {}
        for idx, key in enumerate(self.output_keys):
            low, high = self.bounds[key]
            output[key] = float(low + (action[idx] + 1) * (high - low) / 2)
        return output
