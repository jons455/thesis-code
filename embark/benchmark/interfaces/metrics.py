"""Protocol definition for metric accumulators."""

from __future__ import annotations

from typing import Any, Protocol

from .types import ActionDict, ReferenceDict, StateDict


class MetricAccumulator(Protocol):
    """Stateful metric that observes the control loop."""

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    def reset(self) -> None:
        """Reset accumulated state."""
        ...

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,
        next_state: StateDict,
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        """Update metric with one timestep of data."""
        ...

    def compute(self) -> float | dict[str, float]:
        """Compute final metric value(s)."""
        ...
