"""Identity processors for simple passthrough behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from embark.benchmark.interfaces import (
    ActionDict,
    ActionProcessor,
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
)


@dataclass
class IdentityStateProcessor(StateProcessor):
    """Concatenate state and reference into a flat tensor."""

    state_keys: Sequence[str] | None = None
    reference_keys: Sequence[str] | None = None
    _output_dim: int = 0

    def configure(self, physics_config: SystemConfig, task: ClosedLoopTask) -> None:
        if self.state_keys is None:
            self.state_keys = sorted(task.physics_engine.state_keys)
        if self.reference_keys is None:
            self.reference_keys = sorted(task.reference_keys)
        self._output_dim = len(self.state_keys) + len(self.reference_keys)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        keys = list(self.state_keys or []) + list(self.reference_keys or [])
        values = [state[k] for k in self.state_keys or []] + [
            reference[k] for k in self.reference_keys or []
        ]
        if len(values) != len(keys):
            raise KeyError(
                "State/reference missing required keys for identity processor."
            )
        return torch.tensor(values, dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


@dataclass
class IdentityActionProcessor(ActionProcessor):
    """Map action tensor entries to action dict keys."""

    action_keys: Iterable[str] | None = None

    def configure(self, physics_config: SystemConfig) -> None:
        if self.action_keys is None:
            raise ValueError("IdentityActionProcessor requires action_keys.")

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:
        action = action.detach().cpu().flatten().tolist()
        keys = list(self.action_keys or [])
        if len(action) < len(keys):
            raise ValueError("Action tensor smaller than number of action keys.")
        return {key: float(action[i]) for i, key in enumerate(keys)}
