"""Dynamic response metric accumulators."""

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


@dataclass
class SettlingTime(MetricAccumulator):
    """
    Time for the tracked signal to enter and *remain* within a 2% band.

    The threshold is computed as ``band_fraction * |step_size|``, where
    ``step_size`` is the magnitude of the reference change detected at
    the first non-zero reference.  This means the band scales with the
    commanded step — a 2 A step uses a ±0.04 A band, a 1 A step ±0.02 A.

    A **dwell requirement** is enforced: the signal must stay inside the
    band for at least ``dwell_s`` seconds continuously.  This prevents
    a zero-crossing from being falsely counted as settling.

    Algorithm:
    - Track ``_step_ref``: first non-zero reference seen (the step target).
    - At each step: check if ``|ref - meas| <= band``.
    - Record the start of each in-band run in ``_candidate_entry``.
    - If the signal leaves the band, reset ``_candidate_entry``.
    - If the signal has been in-band for >= ``dwell_s``, record as settled.
    - ``compute()`` returns the latest ``_last_outside`` time if still
      not settled, else ``_settled_at``.

    Output key: ``settling_time_i_q`` (or whichever ``tracked_key`` is used)
    Units: seconds (s), or ``inf`` if the signal never settles.

    """

    tracked_key: str
    band_fraction: float = 0.02  # 2% of step size
    dwell_s: float = 0.001  # must stay in band for 1 ms
    time_key: str = "time"

    _step_ref: float | None = field(default=None, init=False, repr=False)
    _band: float | None = field(default=None, init=False, repr=False)
    _candidate_entry: float | None = field(default=None, init=False, repr=False)
    _settled_at: float | None = field(default=None, init=False, repr=False)
    _last_outside: float | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "settling_time"

    def reset(self) -> None:
        self._step_ref = None
        self._band = None
        self._candidate_entry = None
        self._settled_at = None
        self._last_outside = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        # Already confirmed settled — no more work needed
        if self._settled_at is not None:
            return

        time = float(state.get(self.time_key, 0.0))
        ref_key = f"{self.tracked_key}_ref"
        ref = float(reference[ref_key])
        meas = float(state[self.tracked_key])

        # Detect step: latch the first non-zero reference as the step target
        if self._step_ref is None and ref != 0.0:
            self._step_ref = ref
            self._band = self.band_fraction * abs(ref)

        # Before the step fires, nothing to check
        if self._band is None or self._band == 0.0:
            return

        error = abs(ref - meas)

        if error <= self._band:
            # Start of a potential in-band run
            if self._candidate_entry is None:
                self._candidate_entry = time
            # Check if dwell requirement is met
            if (time - self._candidate_entry) >= self.dwell_s:
                self._settled_at = self._candidate_entry
        else:
            # Out of band — reset dwell tracking
            self._candidate_entry = None
            self._last_outside = time

    def compute(self) -> dict[str, float]:
        key = f"settling_time_{self.tracked_key}"
        if self._settled_at is not None:
            return {key: self._settled_at}
        if self._last_outside is not None:
            return {key: float("inf")}
        return {key: float("inf")}


@dataclass
class Overshoot(MetricAccumulator):
    """
    Percent overshoot relative to the commanded step size (direction-aware).

    For a positive step (target > 0), tracks the maximum of the measured signal
    and computes overshoot as (peak - step_ref) / |step_ref| * 100.
    For a negative step (target < 0), tracks the minimum and computes
    (step_ref - trough) / |step_ref| * 100, so that overshoot is the
    absolute deviation beyond the target in the direction of the step,
    normalised by the step magnitude.

    Formula (robust for either sign)::

        overshoot (%) = max(0, deviation_in_step_direction / |step_ref| * 100)

    where deviation is (peak - step_ref) for step_ref > 0 and
    (step_ref - trough) for step_ref < 0. Returns 0.0 when no step or no overshoot.

    Output key: ``overshoot``
    Units: %

    """

    tracked_key: str
    _step_ref: float | None = field(default=None, init=False, repr=False)
    _peak: float | None = field(default=None, init=False, repr=False)
    _trough: float | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "overshoot"

    def reset(self) -> None:
        self._step_ref = None
        self._peak = None
        self._trough = None

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        ref_key = f"{self.tracked_key}_ref"
        ref = float(reference[ref_key])
        meas = float(state[self.tracked_key])

        # Latch step reference (single step from 0 to ref; step magnitude = |ref|)
        if self._step_ref is None and ref != 0.0:
            self._step_ref = ref

        # Track extreme in the direction of the step: peak above target or trough below
        if self._step_ref is not None:
            if self._step_ref > 0.0:
                if self._peak is None or meas > self._peak:
                    self._peak = meas
            else:
                if self._trough is None or meas < self._trough:
                    self._trough = meas

    def compute(self) -> float:
        if self._step_ref is None or self._step_ref == 0.0:
            return 0.0
        step_mag = abs(self._step_ref)
        if self._step_ref > 0.0:
            if self._peak is None:
                return 0.0
            return max(0.0, (self._peak - self._step_ref) / step_mag * 100.0)
        else:
            if self._trough is None:
                return 0.0
            return max(0.0, (self._step_ref - self._trough) / step_mag * 100.0)


@dataclass
class MultiStepSettlingTime(MetricAccumulator):
    """
    Settling-time statistics across *every* detected reference step.

    A step is detected whenever ``tracked_key_ref`` changes by more than
    ``step_epsilon``. Each step is evaluated independently using a band
    scaled by that step size (``band_fraction * |delta_ref|``) and the same
    dwell criterion as :class:`SettlingTime`.

    Output keys (for tracked key ``i_q``)::

        multi_step_settling_time_i_q_worst
        multi_step_settling_time_i_q_mean
        multi_step_settling_time_i_q_std
        multi_step_settling_time_i_q_num_steps
        multi_step_settling_time_i_q_num_settled

    ``worst`` is ``inf`` if any detected step never settles.
    ``mean``/``std`` are computed over settled steps only and become ``inf``
    when no step settles.
    """

    tracked_key: str
    band_fraction: float = 0.02
    dwell_s: float = 0.001
    time_key: str = "time"
    step_epsilon: float = 1e-12

    _prev_ref: float | None = field(default=None, init=False, repr=False)
    _active_ref: float | None = field(default=None, init=False, repr=False)
    _active_band: float | None = field(default=None, init=False, repr=False)
    _active_candidate_entry: float | None = field(default=None, init=False, repr=False)
    _active_settled_at: float | None = field(default=None, init=False, repr=False)
    _active_recorded: bool = field(default=False, init=False, repr=False)
    _step_settling_times: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def name(self) -> str:
        return "multi_step_settling_time"

    def reset(self) -> None:
        self._prev_ref = None
        self._active_ref = None
        self._active_band = None
        self._active_candidate_entry = None
        self._active_settled_at = None
        self._active_recorded = False
        self._step_settling_times = []

    def _start_step(self, *, prev_ref: float, ref: float) -> None:
        step_size = abs(ref - prev_ref)
        if step_size <= self.step_epsilon:
            self._active_ref = None
            self._active_band = None
            self._active_candidate_entry = None
            self._active_settled_at = None
            self._active_recorded = False
            return
        self._active_ref = ref
        self._active_band = self.band_fraction * step_size
        self._active_candidate_entry = None
        self._active_settled_at = None
        self._active_recorded = False

    def _finalize_active_step(self) -> None:
        if self._active_band is None or self._active_recorded:
            return
        value = (
            self._active_settled_at
            if self._active_settled_at is not None
            else float("inf")
        )
        self._step_settling_times.append(value)
        self._active_recorded = True

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        time = float(state.get(self.time_key, 0.0))
        ref_key = f"{self.tracked_key}_ref"
        ref = float(reference[ref_key])
        meas = float(state[self.tracked_key])

        if self._prev_ref is None:
            self._prev_ref = ref
            if abs(ref) > self.step_epsilon:
                self._start_step(prev_ref=0.0, ref=ref)
        elif abs(ref - self._prev_ref) > self.step_epsilon:
            self._finalize_active_step()
            self._start_step(prev_ref=self._prev_ref, ref=ref)
            self._prev_ref = ref

        if self._active_band is None or self._active_recorded:
            return

        error = abs(self._active_ref - meas)
        if error <= self._active_band:
            if self._active_candidate_entry is None:
                self._active_candidate_entry = time
            if (time - self._active_candidate_entry) >= self.dwell_s:
                self._active_settled_at = self._active_candidate_entry
                self._finalize_active_step()
        else:
            self._active_candidate_entry = None

    def compute(self) -> dict[str, float]:
        self._finalize_active_step()
        key = f"multi_step_settling_time_{self.tracked_key}"
        num_steps = len(self._step_settling_times)
        finite = [v for v in self._step_settling_times if math.isfinite(v)]
        num_settled = len(finite)

        if num_steps == 0:
            return {
                f"{key}_worst": float("inf"),
                f"{key}_mean": float("inf"),
                f"{key}_std": float("inf"),
                f"{key}_num_steps": 0.0,
                f"{key}_num_settled": 0.0,
            }

        worst = max(self._step_settling_times)
        mean = statistics.mean(finite) if finite else float("inf")
        std = statistics.pstdev(finite) if len(finite) >= 2 else 0.0
        return {
            f"{key}_worst": worst,
            f"{key}_mean": mean,
            f"{key}_std": std,
            f"{key}_num_steps": float(num_steps),
            f"{key}_num_settled": float(num_settled),
        }


@dataclass
class MultiStepOvershoot(MetricAccumulator):
    """
    Overshoot statistics across every detected reference step.

    For each step transition in ``tracked_key_ref``, overshoot is measured
    relative to that step's target and magnitude, then aggregated.

    Output keys (for tracked key ``i_q``)::

        multi_step_overshoot_i_q_worst
        multi_step_overshoot_i_q_mean
        multi_step_overshoot_i_q_std
        multi_step_overshoot_i_q_num_steps
    """

    tracked_key: str
    step_epsilon: float = 1e-12

    _prev_ref: float | None = field(default=None, init=False, repr=False)
    _active_ref: float | None = field(default=None, init=False, repr=False)
    _active_step_size: float | None = field(default=None, init=False, repr=False)
    _active_peak_overshoot: float = field(default=0.0, init=False, repr=False)
    _active_recorded: bool = field(default=False, init=False, repr=False)
    _step_overshoots: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def name(self) -> str:
        return "multi_step_overshoot"

    def reset(self) -> None:
        self._prev_ref = None
        self._active_ref = None
        self._active_step_size = None
        self._active_peak_overshoot = 0.0
        self._active_recorded = False
        self._step_overshoots = []

    def _start_step(self, *, prev_ref: float, ref: float) -> None:
        step_size = ref - prev_ref
        if abs(step_size) <= self.step_epsilon:
            self._active_ref = None
            self._active_step_size = None
            self._active_peak_overshoot = 0.0
            self._active_recorded = False
            return
        self._active_ref = ref
        self._active_step_size = step_size
        self._active_peak_overshoot = 0.0
        self._active_recorded = False

    def _finalize_active_step(self) -> None:
        if self._active_step_size is None or self._active_recorded:
            return
        self._step_overshoots.append(self._active_peak_overshoot)
        self._active_recorded = True

    def update(
        self,
        state: StateDict,
        reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        _controller_info: dict | None = None,
    ) -> None:
        ref_key = f"{self.tracked_key}_ref"
        ref = float(reference[ref_key])
        meas = float(state[self.tracked_key])

        if self._prev_ref is None:
            self._prev_ref = ref
            if abs(ref) > self.step_epsilon:
                self._start_step(prev_ref=0.0, ref=ref)
        elif abs(ref - self._prev_ref) > self.step_epsilon:
            self._finalize_active_step()
            self._start_step(prev_ref=self._prev_ref, ref=ref)
            self._prev_ref = ref

        if self._active_step_size is None or self._active_recorded:
            return

        step_size = self._active_step_size
        target = self._active_ref
        if step_size > 0.0:
            overshoot = max(0.0, (meas - target) / abs(step_size) * 100.0)
        else:
            overshoot = max(0.0, (target - meas) / abs(step_size) * 100.0)
        self._active_peak_overshoot = max(self._active_peak_overshoot, overshoot)

    def compute(self) -> dict[str, float]:
        self._finalize_active_step()
        key = f"multi_step_overshoot_{self.tracked_key}"
        num_steps = len(self._step_overshoots)
        if num_steps == 0:
            return {
                f"{key}_worst": 0.0,
                f"{key}_mean": 0.0,
                f"{key}_std": 0.0,
                f"{key}_num_steps": 0.0,
            }
        return {
            f"{key}_worst": max(self._step_overshoots),
            f"{key}_mean": statistics.mean(self._step_overshoots),
            f"{key}_std": statistics.pstdev(self._step_overshoots)
            if num_steps >= 2
            else 0.0,
            f"{key}_num_steps": float(num_steps),
        }
