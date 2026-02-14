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
class SNNStateProcessor(StateProcessor):
    """
    Normalize state for SNN controllers trained via imitation learning.

    Produces a 5-element tensor ``[i_d, i_q, e_d, e_q, n]`` that exactly
    matches the normalization used by
    ``evaluation.pytorch_snn.utils.dataset.PMSMDataset``:

    - ``i_d, i_q``: measured currents divided by ``i_max``
    - ``e_d, e_q``: tracking errors ``(ref - meas) / i_max * error_gain``,
      clipped to [-1, 1]
    - ``n``: motor speed ``n_rpm / n_max`` (converted from omega)

    Parameters
    ----------
    error_gain : float
        Amplification factor for error signals (default 10.0, must match
        the value used during training).
    n_max : float
        Maximum speed in RPM for normalization (default 4000.0, must match
        the value used during training dataset preparation).

    Notes
    -----
    Speed normalization uses RPM-based scaling to match the training datasets
    (PyTorch and Akida), which normalize speed as ``n_rpm / 4000.0``. The
    physics engine provides ``omega`` in rad/s, which is converted to RPM
    before normalization.

    """

    error_gain: float = 10.0
    n_max: float = 4000.0
    _i_max: float = 1.0

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask  # noqa: ARG002
    ) -> None:
        self._i_max = physics_config.i_max

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        import math

        i_d = state["i_d"] / self._i_max
        i_q = state["i_q"] / self._i_max

        e_d = (reference["i_d_ref"] - state["i_d"]) / self._i_max * self.error_gain
        e_q = (reference["i_q_ref"] - state["i_q"]) / self._i_max * self.error_gain

        e_d = max(-1.0, min(1.0, e_d))
        e_q = max(-1.0, min(1.0, e_q))

        # Convert omega (rad/s) to RPM, then normalize
        # This matches the training dataset normalization exactly
        omega = state.get("omega", 0.0)
        n_rpm = omega * 60.0 / (2.0 * math.pi)
        n = n_rpm / self.n_max

        return torch.tensor([i_d, i_q, e_d, e_q, n], dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return 5


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
