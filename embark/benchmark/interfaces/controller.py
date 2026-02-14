"""Protocol definitions for controller policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .types import ActionDict, ReferenceDict, StateDict, SystemConfig

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Controller(Protocol):
    """
    Unified controller interface for the harness.

    Both classical (DictController) and neural (TensorController) controllers must
    implement this interface, either directly or via an adapter.

    """

    def reset(self) -> None:
        """Reset internal state."""
        ...

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        """Compute action from state and reference."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        ...


@runtime_checkable
class TensorController(Protocol):
    """
    Neural network controllers (SNN, ANN).

    These require wrapping with TensorControllerAdapter to be used with the
    ClosedLoopHarness.

    """

    def reset(self) -> None:
        """Reset internal state (membrane potentials, hidden states)."""
        ...

    def forward(self, observation: "torch.Tensor") -> "torch.Tensor":
        """Compute action from observation tensor."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        ...


# DictController is now an alias for Controller (same interface)
DictController = Controller
