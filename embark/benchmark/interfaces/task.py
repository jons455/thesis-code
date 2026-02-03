"""Protocol definition for closed-loop tasks."""

from __future__ import annotations

from typing import Protocol

from .physics import PhysicsEngine
from .types import ActionDict, ReferenceDict, StateDict


class ClosedLoopTask(Protocol):
    """Defines the control objective. Owns a PhysicsEngine."""

    @property
    def physics_engine(self) -> PhysicsEngine:
        """The underlying dynamical system."""
        ...

    @property
    def reference_keys(self) -> set[str]:
        """Keys provided in reference dict."""
        ...

    @property
    def max_steps(self) -> int | None:
        """Maximum episode length (None for infinite)."""
        ...

    def reset(self, seed: int | None = None) -> tuple[StateDict, ReferenceDict]:
        """Reset task and physics."""
        ...

    def step(
        self, action: ActionDict
    ) -> tuple[StateDict, ReferenceDict, bool]:
        """Step physics and update reference."""
        ...
