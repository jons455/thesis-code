"""Dynamic response metric accumulators."""

from __future__ import annotations

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
    Percent overshoot relative to the commanded step size.

    Tracks the peak value of the measured signal and computes::

        overshoot (%) = max(0, (peak - step_ref) / |step_ref| * 100)

    where ``step_ref`` is the first non-zero reference (the step target).
    Returns 0.0 when there is no step or no overshoot.

    Output key: ``overshoot``
    Units: %

    """

    tracked_key: str
    _step_ref: float | None = field(default=None, init=False, repr=False)
    _peak: float | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "overshoot"

    def reset(self) -> None:
        self._step_ref = None
        self._peak = None

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

        # Latch step reference
        if self._step_ref is None and ref != 0.0:
            self._step_ref = ref

        # Track peak magnitude in the direction of the step
        if self._step_ref is not None:
            if self._peak is None or meas > self._peak:
                self._peak = meas

    def compute(self) -> float:
        if self._peak is None or self._step_ref is None or self._step_ref == 0.0:
            return 0.0
        return max(0.0, (self._peak - self._step_ref) / abs(self._step_ref) * 100.0)
