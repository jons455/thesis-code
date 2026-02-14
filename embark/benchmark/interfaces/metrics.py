"""Protocol definition for metric accumulators."""

from __future__ import annotations

from typing import Any, Protocol

from .types import ActionDict, ReferenceDict, StateDict


class MetricAccumulator(Protocol):
    """
    Stateful metric that observes the closed-loop control loop.

    Metric accumulators are attached to a ``ClosedLoopHarness`` and
    called at every simulation timestep via ``update()``.  After the
    episode ends, ``compute()`` returns the final metric value(s).

    **Performance contract:** ``update()`` must run in O(1) time per
    call (no iteration over history, no sorting).  Expensive computation
    should be deferred to ``compute()``.

    Implementors include control metrics (``TrackingMAE``, ``TrackingITAE``,
    ``SettlingTime``, ``Overshoot``) and NeuroBench adapters.

    """

    @property
    def name(self) -> str:
        """
        Unique identifier for this metric.

        Used as a key prefix in the results dictionary returned by the
        harness.

        Returns:
            String identifier (e.g. ``"rmse"``, ``"syops"``).

        """
        ...

    def reset(self) -> None:
        """
        Reset accumulated state for a new episode.

        Called by the harness at the start of each ``run()`` before the
        control loop begins.

        """
        ...

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,
        next_state: StateDict,
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Update the metric with one timestep of data.

        Called once per simulation step, after the controller has acted
        and the physics engine has advanced.

        Args:
            state: System state *before* the action was applied.
            reference: Reference signals for this timestep.
            action: Control action applied by the controller.
            next_state: System state *after* the physics step.
            controller_info: Optional dictionary of controller metadata
                (e.g. spike counts, SyOps, inference latency).  ``None``
                for classical controllers.

        """
        ...

    def compute(self) -> float | dict[str, float]:
        """
        Compute the final metric value(s) for the completed episode.

        Called once by the harness after the control loop ends.

        Returns:
            Either a single float, or a dictionary mapping sub-metric
            names to float values (e.g.
            ``{"total_syops": 1234.0, "syops_per_step": 12.34}``).

        """
        ...
