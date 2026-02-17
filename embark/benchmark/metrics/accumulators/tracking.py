"""Tracking-related metric accumulators."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)
from embark.benchmark.utils.validation import validate_state_reference


def _validate_tracked_keys(tracked_keys: list[str], metric_name: str) -> None:
    """Ensure metric tracked keys are configured correctly."""
    if not isinstance(tracked_keys, list) or not tracked_keys:
        raise ValueError(f"{metric_name}: tracked_keys must be a non-empty list.")
    invalid = [
        key for key in tracked_keys if not isinstance(key, str) or not key.strip()
    ]
    if invalid:
        raise ValueError(
            f"{metric_name}: tracked_keys must contain non-empty strings; "
            f"invalid entries: {invalid!r}."
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
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._sum_itae = {key: 0.0 for key in self.tracked_keys}
        self._prev_time = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        parsed_time = validate_state_reference(
            state,
            reference,
            self.tracked_keys,
            self.name,
            time_key=self.time_key,
        )
        assert parsed_time is not None
        time = parsed_time
        dt = 0.0 if self._prev_time is None else time - self._prev_time
        if dt < 0.0:
            raise ValueError(
                f"{self.name}: non-monotonic time detected (dt={dt}). "
                "State time must be non-decreasing."
            )
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
class MultiStepITAE(MetricAccumulator):
    """
    ITAE for multi-step profiles with both global and per-step reporting.

    For each tracked axis:
    - Global ITAE integrates over the full episode:
      ``integral t * |e(t)| dt``
    - Per-step ITAE resets local time at each detected reference transition:
      ``integral (t - t_step_start) * |e(t)| dt``

    Output keys (for tracked key ``i_q``)::
        multi_step_itae_i_q_global
        multi_step_itae_i_q_per_step_mean
        multi_step_itae_i_q_per_step_worst
        multi_step_itae_i_q_per_step_std
        multi_step_itae_i_q_num_steps
    """

    tracked_keys: list[str]
    time_key: str = "time"
    step_epsilon: float = 1e-12
    _prev_time: float | None = field(default=None, init=False, repr=False)
    _global_itae: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _prev_refs: dict[str, float | None] = field(default_factory=dict, init=False, repr=False)
    _step_start_time: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _active_step_itae: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _per_step_itaes: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return "multi_step_itae"

    def reset(self) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._prev_time = None
        self._global_itae = {key: 0.0 for key in self.tracked_keys}
        self._prev_refs = {key: None for key in self.tracked_keys}
        self._step_start_time = {key: 0.0 for key in self.tracked_keys}
        self._active_step_itae = {key: 0.0 for key in self.tracked_keys}
        self._per_step_itaes = {key: [] for key in self.tracked_keys}

    def _finalize_step(self, key: str) -> None:
        self._per_step_itaes[key].append(self._active_step_itae[key])
        self._active_step_itae[key] = 0.0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        parsed_time = validate_state_reference(
            state,
            reference,
            self.tracked_keys,
            self.name,
            time_key=self.time_key,
        )
        assert parsed_time is not None
        time = parsed_time
        dt = 0.0 if self._prev_time is None else time - self._prev_time
        if dt < 0.0:
            raise ValueError(
                f"{self.name}: non-monotonic time detected (dt={dt}). "
                "State time must be non-decreasing."
            )

        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            ref = float(reference[ref_key])
            error = abs(ref - float(state[key]))
            prev_ref = self._prev_refs[key]

            if prev_ref is None:
                self._prev_refs[key] = ref
                self._step_start_time[key] = time
            elif abs(ref - prev_ref) > self.step_epsilon:
                self._finalize_step(key)
                self._step_start_time[key] = time
                self._prev_refs[key] = ref

            self._global_itae[key] += error * time * dt
            local_t = max(0.0, time - self._step_start_time[key])
            self._active_step_itae[key] += error * local_t * dt

        self._prev_time = time

    def compute(self) -> dict[str, float]:
        _validate_tracked_keys(self.tracked_keys, self.name)
        result: dict[str, float] = {}
        for key in self.tracked_keys:
            # Final active step
            self._finalize_step(key)
            values = self._per_step_itaes[key]
            count = len(values)
            result[f"multi_step_itae_{key}_global"] = self._global_itae[key]
            result[f"multi_step_itae_{key}_num_steps"] = float(count)
            if count == 0:
                result[f"multi_step_itae_{key}_per_step_mean"] = 0.0
                result[f"multi_step_itae_{key}_per_step_worst"] = 0.0
                result[f"multi_step_itae_{key}_per_step_std"] = 0.0
                continue
            result[f"multi_step_itae_{key}_per_step_mean"] = statistics.mean(values)
            result[f"multi_step_itae_{key}_per_step_worst"] = max(values)
            result[f"multi_step_itae_{key}_per_step_std"] = (
                statistics.pstdev(values) if count >= 2 else 0.0
            )
        return result


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
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._max_errors = {key: 0.0 for key in self.tracked_keys}

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        validate_state_reference(state, reference, self.tracked_keys, self.name)
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
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._sum_sq = {key: 0.0 for key in self.tracked_keys}
        self._count = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        parsed_time = validate_state_reference(
            state,
            reference,
            self.tracked_keys,
            self.name,
            time_key=self.time_key,
        )
        assert parsed_time is not None
        time = parsed_time
        if time < self.transient_s:
            return

        self._count += 1
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            error = float(reference[ref_key]) - float(state[key])
            self._sum_sq[key] = self._sum_sq.get(key, 0.0) + error * error

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        if self._count == 0:
            # No samples in steady-state window (e.g. episode ended before transient_s)
            for key in self.tracked_keys:
                results[f"rms_{key}"] = float("nan")
        else:
            for key in self.tracked_keys:
                results[f"rms_{key}"] = math.sqrt(
                    self._sum_sq.get(key, 0.0) / self._count
                )
        return results


@dataclass
class MultiStepRMS(MetricAccumulator):
    """
    RMS error for multi-step profiles with global and per-step reporting.

    For each tracked axis:
    - Global RMS is computed over the entire episode.
    - Per-step RMS is computed within each detected reference segment and then
      summarized (mean/worst/std).

    Output keys (for tracked key ``i_q``)::
        multi_step_rms_i_q_global
        multi_step_rms_i_q_per_step_mean
        multi_step_rms_i_q_per_step_worst
        multi_step_rms_i_q_per_step_std
        multi_step_rms_i_q_num_steps
    """

    tracked_keys: list[str]
    step_epsilon: float = 1e-12
    _global_sum_sq: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _global_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _prev_refs: dict[str, float | None] = field(default_factory=dict, init=False, repr=False)
    _step_sum_sq: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _step_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _per_step_rms: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return "multi_step_rms"

    def reset(self) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._global_sum_sq = {key: 0.0 for key in self.tracked_keys}
        self._global_count = {key: 0 for key in self.tracked_keys}
        self._prev_refs = {key: None for key in self.tracked_keys}
        self._step_sum_sq = {key: 0.0 for key in self.tracked_keys}
        self._step_count = {key: 0 for key in self.tracked_keys}
        self._per_step_rms = {key: [] for key in self.tracked_keys}

    def _finalize_step(self, key: str) -> None:
        count = self._step_count[key]
        if count <= 0:
            return
        rms = math.sqrt(self._step_sum_sq[key] / count)
        self._per_step_rms[key].append(rms)
        self._step_sum_sq[key] = 0.0
        self._step_count[key] = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        validate_state_reference(state, reference, self.tracked_keys, self.name)
        for key in self.tracked_keys:
            ref_key = f"{key}_ref"
            ref = float(reference[ref_key])
            error = ref - float(state[key])
            prev_ref = self._prev_refs[key]

            if prev_ref is None:
                self._prev_refs[key] = ref
            elif abs(ref - prev_ref) > self.step_epsilon:
                self._finalize_step(key)
                self._prev_refs[key] = ref

            self._global_sum_sq[key] += error * error
            self._global_count[key] += 1
            self._step_sum_sq[key] += error * error
            self._step_count[key] += 1

    def compute(self) -> dict[str, float]:
        _validate_tracked_keys(self.tracked_keys, self.name)
        result: dict[str, float] = {}
        for key in self.tracked_keys:
            self._finalize_step(key)
            global_count = max(self._global_count[key], 1)
            global_rms = math.sqrt(self._global_sum_sq[key] / global_count)
            values = self._per_step_rms[key]
            count = len(values)
            result[f"multi_step_rms_{key}_global"] = global_rms
            result[f"multi_step_rms_{key}_num_steps"] = float(count)
            if count == 0:
                result[f"multi_step_rms_{key}_per_step_mean"] = 0.0
                result[f"multi_step_rms_{key}_per_step_worst"] = 0.0
                result[f"multi_step_rms_{key}_per_step_std"] = 0.0
                continue
            result[f"multi_step_rms_{key}_per_step_mean"] = statistics.mean(values)
            result[f"multi_step_rms_{key}_per_step_worst"] = max(values)
            result[f"multi_step_rms_{key}_per_step_std"] = (
                statistics.pstdev(values) if count >= 2 else 0.0
            )
        return result


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
        _validate_tracked_keys(self.tracked_keys, self.name)
        self._sum_abs = {key: 0.0 for key in self.tracked_keys}
        self._count = 0

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        _validate_tracked_keys(self.tracked_keys, self.name)
        validate_state_reference(state, reference, self.tracked_keys, self.name)
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
