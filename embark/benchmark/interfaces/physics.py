"""Protocol definition for physical dynamics engines."""

from __future__ import annotations

from typing import Any, Protocol

from .types import ActionDict, StateDict, SystemConfig


class PhysicsEngine(Protocol):
    """Abstract interface for physical dynamical systems."""

    @property
    def config(self) -> SystemConfig:
        """Immutable physical properties (R, L, J, friction, limits)."""
        ...

    def reset(self, seed: int | None = None) -> StateDict:
        """
        Reset to initial state.

        Returns initial state dict.

        """
        ...

    def step(self, action: ActionDict) -> tuple[StateDict, dict[str, Any]]:
        """
        Execute one physics step.

        Args:
            action: Physical units (e.g., {"v_alpha": 12.0, "v_beta": -5.0})

        Returns:
            (next_state, debug_info)

        """
        ...

    def close(self) -> None:
        """Clean up resources (simulator handles, etc.)."""
        ...

    @property
    def state_keys(self) -> set[str]:
        """Keys present in state dict."""
        ...

    @property
    def action_keys(self) -> set[str]:
        """Keys expected in action dict."""
        ...
