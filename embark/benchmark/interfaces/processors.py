"""Protocol definitions for state/action processors."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from .task import ClosedLoopTask
from .types import ActionDict, ReferenceDict, StateDict, SystemConfig

if TYPE_CHECKING:
    import torch


class StateProcessor(Protocol):
    """Converts physics state dict → controller observation tensor."""

    def configure(self, physics_config: SystemConfig, task: ClosedLoopTask) -> None:
        """Called once at harness setup to learn normalization bounds."""
        ...

    def __call__(self, state: StateDict, reference: ReferenceDict) -> "torch.Tensor":
        """Process state and reference into observation tensor."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of output tensor."""
        ...


class ActionProcessor(Protocol):
    """Converts controller action tensor → physics action dict."""

    def configure(self, physics_config: SystemConfig) -> None:
        """Called once at harness setup to learn action bounds."""
        ...

    def __call__(
        self, action: "torch.Tensor", physics_config: SystemConfig
    ) -> ActionDict:
        """Convert action tensor to physical units."""
        ...
