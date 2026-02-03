"""Efficiency-related metric accumulators."""

from __future__ import annotations

from dataclasses import dataclass

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


@dataclass
class ControlEffort(MetricAccumulator):
    """Total control effort based on action magnitude."""

    _sum: float = 0.0

    @property
    def name(self) -> str:
        return "control_effort"

    def reset(self) -> None:
        self._sum = 0.0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        self._sum += sum(abs(v) for v in action.values())

    def compute(self) -> float:
        return float(self._sum)


@dataclass
class EnergyConsumption(MetricAccumulator):
    """Energy proxy based on squared action magnitude."""

    _sum: float = 0.0

    @property
    def name(self) -> str:
        return "energy_consumption"

    def reset(self) -> None:
        self._sum = 0.0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        self._sum += sum(float(v) ** 2 for v in action.values())

    def compute(self) -> float:
        return float(self._sum)
