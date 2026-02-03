"""Neuromorphic metric accumulators."""

from __future__ import annotations

from dataclasses import dataclass

from embark.benchmark.interfaces import ActionDict, MetricAccumulator, ReferenceDict, StateDict


@dataclass
class SyOpsAccumulator(MetricAccumulator):
    """Accumulate synaptic operations from controller info."""

    total_syops: int = 0
    steps: int = 0

    @property
    def name(self) -> str:
        return "syops"

    def reset(self) -> None:
        self.total_syops = 0
        self.steps = 0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,
    ) -> None:
        if controller_info and "total_syops" in controller_info:
            self.total_syops += int(controller_info["total_syops"])
        elif controller_info and "syops" in controller_info:
            self.total_syops += int(controller_info["syops"])
        self.steps += 1

    def compute(self) -> dict[str, float]:
        per_step = self.total_syops / max(self.steps, 1)
        return {"total_syops": float(self.total_syops), "syops_per_step": per_step}


@dataclass
class SpikeCountAccumulator(MetricAccumulator):
    """Accumulate spike counts from controller info."""

    total_spikes: int = 0
    steps: int = 0

    @property
    def name(self) -> str:
        return "spike_count"

    def reset(self) -> None:
        self.total_spikes = 0
        self.steps = 0

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,
    ) -> None:
        if controller_info and "total_spikes" in controller_info:
            self.total_spikes += int(controller_info["total_spikes"])
        elif controller_info and "spikes" in controller_info:
            self.total_spikes += int(controller_info["spikes"])
        self.steps += 1

    def compute(self) -> dict[str, float]:
        per_step = self.total_spikes / max(self.steps, 1)
        return {"total_spikes": float(self.total_spikes), "spikes_per_step": per_step}
