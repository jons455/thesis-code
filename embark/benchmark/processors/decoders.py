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
    bounds: dict[str, tuple[float, float]] | None = None

    def configure(self, physics_config: SystemConfig) -> None:
        """Auto-configure bounds from physics config limits."""
        if self.bounds is None:
            # Default assumption: symmetric voltage limits [-u_max, u_max]
            # This covers typical PMSM control (v_d, v_q)
            u_max = getattr(physics_config, "u_max", 1.0)
            self.bounds = {key: (-u_max, u_max) for key in self.output_keys}

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:  # noqa: ARG002
        action = action.detach().cpu().flatten().tolist()
        if len(action) < len(self.output_keys):
            raise ValueError("Action tensor smaller than number of output keys.")

        if self.bounds is None:
            # Fallback if configure wasn't called (shouldn't happen in harness)
            self.configure(physics_config)

        output: ActionDict = {}
        for idx, key in enumerate(self.output_keys):
            # Ensure bounds exist (configure guarantees it, but type checker might complain)
            bounds = self.bounds.get(key, (-1.0, 1.0)) if self.bounds else (-1.0, 1.0)
            low, high = bounds
            output[key] = float(low + (action[idx] + 1) * (high - low) / 2)
        return output
