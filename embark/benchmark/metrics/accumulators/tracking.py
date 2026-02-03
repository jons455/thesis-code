"""Tracking-related metric accumulators."""

from __future__ import annotations

from dataclasses import dataclass

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


@dataclass
class TrackingRMSE(MetricAccumulator):
    """RMSE of tracking error for selected keys."""

    tracked_keys: list[str]
    _sum_sq: dict[str, float] | None = None
    _count: int = 0

    @property
    def name(self) -> str:
        return "tracking_rmse"

    def reset(self) -> None:
        self._sum_sq = {key: 0.0 for key in self.tracked_keys}
        self._count = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = reference[ref_key] - state[key]
            self._sum_sq[key] += float(error * error)
        self._count += 1

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for key in self.tracked_keys:
            denom = max(self._count, 1)
            results[f"rmse_{key}"] = (self._sum_sq[key] / denom) ** 0.5
        return results


@dataclass
class TrackingMAE(MetricAccumulator):
    """MAE of tracking error for selected keys."""

    tracked_keys: list[str]
    _sum_abs: dict[str, float] | None = None
    _count: int = 0

    @property
    def name(self) -> str:
        return "tracking_mae"

    def reset(self) -> None:
        self._sum_abs = {key: 0.0 for key in self.tracked_keys}
        self._count = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = abs(reference[ref_key] - state[key])
            self._sum_abs[key] += float(error)
        self._count += 1

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for key in self.tracked_keys:
            denom = max(self._count, 1)
            results[f"mae_{key}"] = self._sum_abs[key] / denom
        return results


@dataclass
class TrackingITAE(MetricAccumulator):
    """ITAE (time-weighted absolute error) for selected keys."""

    tracked_keys: list[str]
    time_key: str = "time"
    _sum_itae: dict[str, float] | None = None
    _prev_time: float | None = None

    @property
    def name(self) -> str:
        return "tracking_itae"

    def reset(self) -> None:
        self._sum_itae = {key: 0.0 for key in self.tracked_keys}
        self._prev_time = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        time = state.get(self.time_key, 0.0)
        dt = 0.0 if self._prev_time is None else float(time - self._prev_time)
        self._prev_time = float(time)
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = abs(reference[ref_key] - state[key])
            self._sum_itae[key] += float(error * time * dt)

    def compute(self) -> dict[str, float]:
        return {f"itae_{key}": self._sum_itae[key] for key in self.tracked_keys}
