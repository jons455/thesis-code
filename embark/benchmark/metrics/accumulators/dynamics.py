"""Dynamic response metric accumulators."""

from __future__ import annotations

from dataclasses import dataclass

from embark.benchmark.interfaces import ActionDict, MetricAccumulator, ReferenceDict, StateDict


@dataclass
class SettlingTime(MetricAccumulator):
    """Time until error stays within a threshold."""

    tracked_key: str
    threshold: float = 0.02
    time_key: str = "time"
    _first_within: float | None = None
    _last_outside: float | None = None

    @property
    def name(self) -> str:
        return "settling_time"

    def reset(self) -> None:
        self._first_within = None
        self._last_outside = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        time = float(state.get(self.time_key, 0.0))
        ref_key = f"{self.tracked_key}_ref"
        error = abs(reference[ref_key] - state[self.tracked_key])
        if error <= self.threshold:
            if self._first_within is None:
                self._first_within = time
        else:
            self._last_outside = time

    def compute(self) -> float:
        if self._first_within is None:
            return float("inf")
        if self._last_outside is None:
            return float(self._first_within)
        return float(self._last_outside)


@dataclass
class Overshoot(MetricAccumulator):
    """Percent overshoot relative to reference step."""

    tracked_key: str
    _max_value: float | None = None
    _final_ref: float | None = None

    @property
    def name(self) -> str:
        return "overshoot"

    def reset(self) -> None:
        self._max_value = None
        self._final_ref = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict | None = None,  # noqa: ARG002
    ) -> None:
        value = state[self.tracked_key]
        if self._max_value is None or value > self._max_value:
            self._max_value = float(value)
        ref_key = f"{self.tracked_key}_ref"
        self._final_ref = reference[ref_key]

    def compute(self) -> float:
        if self._max_value is None or self._final_ref is None:
            return 0.0
        if self._final_ref == 0:
            return 0.0
        return max(0.0, (self._max_value - self._final_ref) / abs(self._final_ref) * 100)
