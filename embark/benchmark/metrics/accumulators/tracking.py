"""Tracking-related metric accumulators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


@dataclass
class TrackingITAE(MetricAccumulator):
    """
    ITAE (Integral Time Absolute Error) over a fixed transient window.

    Integrates ``t * |e(t)| * dt`` over the first ``window_s`` seconds
    after episode start, which captures only the transient response and
    avoids polluting the score with steady-state drift.

    Formula (per axis)::

        ITAE = integral_0^{window_s} t * |ref(t) - meas(t)| dt

    Default window: 50 ms (500 steps at 10 kHz) — covers the full
    transient for a well-tuned PMSM current controller.

    Output keys: ``itae_i_q``, ``itae_i_d``
    Units: A·s²

    """

    tracked_keys: list[str]
    time_key: str = "time"
    window_s: float = 0.05  # integrate over first 50 ms only
    _sum_itae: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _prev_time: float | None = field(default=None, init=False, repr=False)

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
        time = float(state.get(self.time_key, 0.0))
        dt = 0.0 if self._prev_time is None else time - self._prev_time
        self._prev_time = time

        # Only integrate within the transient window
        if time > self.window_s:
            return

        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = abs(float(reference[ref_key]) - float(state[key]))
            self._sum_itae[key] += error * time * dt

    def compute(self) -> dict[str, float]:
        return {f"itae_{key}": self._sum_itae[key] for key in self.tracked_keys}


@dataclass
class MaximumError(MetricAccumulator):
    """
    Maximum absolute tracking error (worst-case safety metric).

    Tracks ``e_max = max(|ref - meas|)`` over the entire episode for each
    tracked key.  This is critical for certifying controller safety — a
    controller with low average error but high max-error may have dangerous spikes.

    """

    tracked_keys: list[str]
    _max_errors: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return "max_error"

    def reset(self) -> None:
        self._max_errors = {key: 0.0 for key in self.tracked_keys}

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
            error = abs(float(reference[ref_key]) - float(state[key]))
            if error > self._max_errors.get(key, 0.0):
                self._max_errors[key] = error

    def compute(self) -> dict[str, float]:
        return {
            f"max_error_{key}": self._max_errors.get(key, 0.0)
            for key in self.tracked_keys
        }


@dataclass
class SteadyStateRMS(MetricAccumulator):
    """
    RMS of tracking error over the steady-state window.

    Captures torque ripple and residual steady-state deviation after the
    transient has died out.  Only samples taken after ``transient_s``
    seconds are included, up to episode end.

    Formula::

        RMS = sqrt( (1/N) * sum_{k=T_ss}^{T} (ref[k] - meas[k])^2 )

    where ``T_ss`` is the first step after ``transient_s``.

    A per-axis mean reference (``ref_mean``) is subtracted from the
    reference before squaring so that a non-zero setpoint does not inflate
    the result — this gives the RMS of the *error signal*, not the signal
    itself.

    Output keys: ``rms_i_q``, ``rms_i_d``
    Units: A

    """

    tracked_keys: list[str]
    time_key: str = "time"
    transient_s: float = 0.05  # exclude first 50 ms (transient)
    _sum_sq: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    @property
    def name(self) -> str:
        return "steady_state_rms"

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
        time = float(state.get(self.time_key, 0.0))
        if time < self.transient_s:
            return

        self._count += 1
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = float(reference[ref_key]) - float(state[key])
            self._sum_sq[key] = self._sum_sq.get(key, 0.0) + error * error

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        n = max(self._count, 1)
        for key in self.tracked_keys:
            results[f"rms_{key}"] = math.sqrt(self._sum_sq.get(key, 0.0) / n)
        return results


# ---------------------------------------------------------------------------
# TrackingMAE: full-episode MAE.  Included in the default metric factory
# alongside SteadyStateRMS (steady-state) and TrackingITAE (transient).
# ---------------------------------------------------------------------------


@dataclass
class TrackingMAE(MetricAccumulator):
    """
    MAE (Mean Absolute Error) of tracking error over the full episode.

    Formula (per axis)::

        MAE = (1/N) * sum(|ref[k] - meas[k]|)

    Industry-standard metric for motor control validation. Intuitive and
    robust measure of overall tracking quality across the entire episode,
    including both transient and steady-state phases.

    Output keys: ``mae_i_q``, ``mae_i_d``
    Units: A

    """

    tracked_keys: list[str]
    _sum_abs: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

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
            error = abs(float(reference[ref_key]) - float(state[key]))
            self._sum_abs[key] = self._sum_abs.get(key, 0.0) + error
        self._count += 1

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for key in self.tracked_keys:
            results[f"mae_{key}"] = self._sum_abs.get(key, 0.0) / max(self._count, 1)
        return results
