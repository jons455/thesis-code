"""State normalization processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from embark.benchmark.interfaces import (
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
)


@dataclass
class StandardScalerProcessor(StateProcessor):
    """Standardize inputs with mean/std per key."""

    input_keys: Sequence[str]
    reference_keys: Sequence[str]
    mean: dict[str, float] | None = None
    std: dict[str, float] | None = None
    _output_dim: int = 0

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask
    ) -> None:  # noqa: ARG002
        if self.mean is None:
            self.mean = {
                k: 0.0 for k in list(self.input_keys) + list(self.reference_keys)
            }
        if self.std is None:
            self.std = {
                k: 1.0 for k in list(self.input_keys) + list(self.reference_keys)
            }
        self._output_dim = len(self.input_keys) + len(self.reference_keys)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        values = []
        for key in self.input_keys:
            values.append((state[key] - self.mean[key]) / self.std[key])
        for key in self.reference_keys:
            values.append((reference[key] - self.mean[key]) / self.std[key])
        return torch.tensor(values, dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


@dataclass
class MinMaxProcessor(StateProcessor):
    """Min-max normalize inputs to [-1, 1]."""

    input_keys: Sequence[str]
    reference_keys: Sequence[str]
    bounds: dict[str, tuple[float, float]] | None = None
    _output_dim: int = 0

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask
    ) -> None:  # noqa: ARG002
        if self.bounds is None:
            self.bounds = {}
            for key in list(self.input_keys) + list(self.reference_keys):
                if key.startswith("i_") or key.startswith("e_"):
                    self.bounds[key] = (-physics_config.i_max, physics_config.i_max)
                elif key == "omega":
                    omega_max = getattr(physics_config, "omega_max", 1.0)
                    self.bounds[key] = (-omega_max, omega_max)
                else:
                    self.bounds[key] = (-1.0, 1.0)
        self._output_dim = len(self.input_keys) + len(self.reference_keys)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        values = []
        for key in self.input_keys:
            low, high = self.bounds[key]
            values.append(2 * (state[key] - low) / (high - low) - 1)
        for key in self.reference_keys:
            low, high = self.bounds[key]
            values.append(2 * (reference[key] - low) / (high - low) - 1)
        return torch.tensor(values, dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim
