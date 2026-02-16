"""Neuromorphic efficiency metric accumulators.

These accumulators read spike statistics from ``controller_info`` produced by
SNN controllers (via ``last_info``).  When no spike data is available (e.g.
classical PI controller), all metrics silently return zeros — safe to include
in every benchmark run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


@dataclass
class SpikeCount(MetricAccumulator):
    """Accumulate total spike count across all control steps.

    Reads ``controller_info["total_spikes"]`` per step.

    Returns:
        ``{"total_spikes": float, "spikes_per_step": float}``
    """

    _total: int = field(default=0, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)

    @property
    def name(self) -> str:
        return "spike_count"

    def reset(self) -> None:
        self._total = 0
        self._steps = 0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        if controller_info and "total_spikes" in controller_info:
            self._total += int(controller_info["total_spikes"])
            self._steps += 1

    def compute(self) -> dict[str, float]:
        return {
            "total_spikes": float(self._total),
            "spikes_per_step": (
                float(self._total) / self._steps if self._steps > 0 else 0.0
            ),
        }


@dataclass
class SynapticOps(MetricAccumulator):
    """Accumulate synaptic operations (SyOps) across all control steps.

    Reads ``controller_info["syops"]`` per step.  SyOps count the number of
    multiply-accumulate equivalents triggered by spikes propagating through
    weighted connections.

    Returns:
        ``{"total_syops": float, "syops_per_step": float}``
    """

    _total: int = field(default=0, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)

    @property
    def name(self) -> str:
        return "synaptic_ops"

    def reset(self) -> None:
        self._total = 0
        self._steps = 0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        if controller_info and "syops" in controller_info:
            self._total += int(controller_info["syops"])
            self._steps += 1

    def compute(self) -> dict[str, float]:
        return {
            "total_syops": float(self._total),
            "syops_per_step": (
                float(self._total) / self._steps if self._steps > 0 else 0.0
            ),
        }


@dataclass
class ActivationSparsity(MetricAccumulator):
    """Track neuron activation sparsity over the episode.

    Reads ``controller_info["sparsity"]`` per step (a float in [0, 1] where
    1.0 means all neurons are silent).

    Returns:
        ``{"mean_sparsity": float, "min_sparsity": float, "max_sparsity": float}``
    """

    _values: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def name(self) -> str:
        return "activation_sparsity"

    def reset(self) -> None:
        self._values.clear()

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        if controller_info and "sparsity" in controller_info:
            self._values.append(float(controller_info["sparsity"]))

    def compute(self) -> dict[str, float]:
        if not self._values:
            return {
                "mean_sparsity": 0.0,
                "min_sparsity": 0.0,
                "max_sparsity": 0.0,
            }
        return {
            "mean_sparsity": sum(self._values) / len(self._values),
            "min_sparsity": min(self._values),
            "max_sparsity": max(self._values),
        }
