"""Unit tests for safety limits and metric correctness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from embark.benchmark.interfaces import ActionDict, ReferenceDict, StateDict
from embark.benchmark.metrics.accumulators.dynamics import (
    MultiStepOvershoot,
    MultiStepSettlingTime,
    Overshoot,
    SettlingTime,
)
from embark.benchmark.metrics.accumulators.tracking import (
    MaximumError,
    MultiStepITAE,
    MultiStepRMS,
    TrackingITAE,
    TrackingMAE,
)
from embark.benchmark.tasks.pmsm_current_control import (
    PMSMCurrentControlTask,
    SafetyLimits,
)


class TestSafetyLimits:
    """Test safety limit enforcement."""

    def test_check_action_limits(self):
        """Test action limits (voltage)."""
        limits = SafetyLimits(max_voltage_v=50.0)

        # Safe actions
        assert limits.check_action({"v_d": 49.0, "v_q": -49.0}) is None

        # Unsafe actions
        assert "voltage_limit_exceeded" in limits.check_action(
            {"v_d": 51.0, "v_q": 0.0}
        )
        assert "voltage_limit_exceeded" in limits.check_action(
            {"v_d": 0.0, "v_q": -51.0}
        )

        # NaN actions
        assert "action_nan" in limits.check_action({"v_d": float("nan"), "v_q": 0.0})

    def test_check_state_limits(self):
        """Test state limits (current, speed)."""
        limits = SafetyLimits(max_current_a=20.0, max_speed_rpm=1000.0)

        # Safe state
        safe_state = {
            "i_d": 19.0,
            "i_q": 19.0,
            "omega": 100.0,
        }  # omega in rad/s, check logic
        # 100 rad/s * 60 / 2pi approx 955 rpm -> Safe
        assert limits.check_state(safe_state) is None

        # Unsafe current
        assert "current_limit_exceeded" in limits.check_state(
            {"i_d": 21.0, "i_q": 0.0, "omega": 0.0}
        )
        assert "current_limit_exceeded" in limits.check_state(
            {"i_d": 0.0, "i_q": -21.0, "omega": 0.0}
        )

        # Unsafe speed
        # 110 rad/s * 9.55 = 1050 rpm -> Unsafe
        assert "speed_limit_exceeded" in limits.check_state(
            {"i_d": 0.0, "i_q": 0.0, "omega": 110.0}
        )

        # NaN state
        assert "state_nan" in limits.check_state(
            {"i_d": float("nan"), "i_q": 0.0, "omega": 0.0}
        )

    def test_task_integration_safety(self):
        """Test that task terminates on safety violation."""
        # Create a mock physics engine that just returns the input action as state (for simplicity)
        # or we can use the real one but force a violation.
        # Easier: Mock the physics engine.

        limits = SafetyLimits(max_voltage_v=10.0)
        task = PMSMCurrentControlTask(safety_limits=limits)

        # Inject mock physics engine
        mock_physics = MagicMock()
        mock_physics.reset.return_value = {
            "i_d": 0.0,
            "i_q": 0.0,
            "omega": 0.0,
            "time": 0.0,
        }
        mock_physics.step.return_value = (
            {"i_d": 0.0, "i_q": 0.0, "omega": 0.0, "time": 0.1},
            {},
        )
        task.physics_engine = mock_physics
        task.reset()

        # Safe step
        _, _, done = task.step({"v_d": 5.0, "v_q": 5.0})
        assert not done
        assert not task.terminated_by_safety

        # Unsafe step (Voltage violation)
        _, _, done = task.step({"v_d": 15.0, "v_q": 0.0})
        assert done
        assert task.terminated_by_safety
        assert "voltage_limit_exceeded" in task.last_violation_reason

    def test_legacy_check_combined_path(self):
        """Legacy check() returns bool from action/state checks."""
        limits = SafetyLimits(max_voltage_v=10.0, max_current_a=5.0)

        assert (
            limits.check(state={"i_d": 0.0, "i_q": 0.0}, action={"v_d": 20.0}) is True
        )
        assert (
            limits.check(
                state={"i_d": 6.0, "i_q": 0.0},
                action={"v_d": 1.0},
            )
            is True
        )
        assert (
            limits.check(
                state={"i_d": 1.0, "i_q": 1.0},
                action={"v_d": 1.0},
            )
            is False
        )

    def test_task_reference_keys_property(self):
        task = PMSMCurrentControlTask(safety_limits=SafetyLimits())
        assert task.reference_keys == {"i_d_ref", "i_q_ref"}

    def test_task_safety_callback_invoked(self):
        """on_safety_violation callback is called with reason and state."""
        callback = MagicMock()
        limits = SafetyLimits(max_voltage_v=1.0)
        task = PMSMCurrentControlTask(
            safety_limits=limits, on_safety_violation=callback
        )

        mock_physics = MagicMock()
        next_state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0, "time": 0.1}
        mock_physics.reset.return_value = {
            "i_d": 0.0,
            "i_q": 0.0,
            "omega": 0.0,
            "time": 0.0,
        }
        mock_physics.step.return_value = (next_state, {})

        task.physics_engine = mock_physics
        task.reset()
        task.step({"v_d": 5.0, "v_q": 0.0})  # action violation

        callback.assert_called_once()
        cb_state, cb_reason = callback.call_args.args
        assert cb_state == next_state
        assert "voltage_limit_exceeded" in cb_reason


class TestMetricCorrectness:
    """Test mathematical correctness of metric accumulators."""

    def test_tracking_mae_calculation(self):
        """Verify MAE calculation with known values."""
        metric = TrackingMAE(tracked_keys=["val"])
        metric.reset()

        # Error sequence: 3.0, 4.0
        # Absolute errors: 3.0, 4.0
        # Mean absolute error: 3.5

        metric.update(
            state={"val": 7.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        metric.update(
            state={"val": 6.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        result = metric.compute()
        assert result["mae_val"] == pytest.approx(3.5, abs=1e-6)

    def test_tracking_itae_calculation(self):
        """Verify ITAE calculation with simple time steps within the 50 ms window."""
        metric = TrackingITAE(tracked_keys=["val"], window_s=0.05)
        metric.reset()

        # Two steps within the transient window: t=0.0 and t=0.01 with error 2.0
        # First step: dt=0   → contribution = 0
        # Second step: dt=0.01, t=0.01 → contribution = 2.0 * 0.01 * 0.01 = 0.0002
        metric.update(
            state={"val": 8.0, "time": 0.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        metric.update(
            state={"val": 8.0, "time": 0.01},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        result = metric.compute()
        # dt=0.01, t=0.01, error=2.0 → 2.0 * 0.01 * 0.01 = 0.0002
        assert result["itae_val"] == pytest.approx(0.0002, abs=1e-9)

    def test_tracking_itae_ignores_steps_outside_window(self):
        """ITAE should not accumulate contributions after the window ends."""
        metric = TrackingITAE(tracked_keys=["val"], window_s=0.05)
        metric.reset()

        metric.update(
            state={"val": 8.0, "time": 0.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        # This step is past the 50ms window — should NOT contribute
        metric.update(
            state={"val": 8.0, "time": 1.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        result = metric.compute()
        assert result["itae_val"] == pytest.approx(0.0, abs=1e-9)

    def test_maximum_error_across_steps(self):
        """MaximumError tracks worst-case absolute error."""
        metric = MaximumError(tracked_keys=["val"])
        metric.reset()

        metric.update(
            state={"val": 9.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )  # error = 1
        metric.update(
            state={"val": 4.0},
            reference={"val_ref": 10.0},
            _action={},
            _next_state={},
            _controller_info={},
        )  # error = 6

        result = metric.compute()
        assert result["max_error_val"] == pytest.approx(6.0, abs=1e-6)


class TestDynamicMetrics:
    """Tests for dynamic response metrics (settling time, overshoot)."""

    def test_settling_time_settles_with_dwell(self):
        """SettlingTime reports entry time once dwell is satisfied."""
        # ref=2.0 → 2% band = 0.04 A, dwell=0.001 s
        metric = SettlingTime(
            tracked_key="val", band_fraction=0.02, dwell_s=0.001, time_key="time"
        )
        metric.reset()

        # t=0.0: reference not yet fired (0) → no band computed yet
        metric.update(
            state={"val": 0.0, "time": 0.0},
            reference={"val_ref": 0.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        # t=0.005: step fires (ref=2.0), meas=1.97 → error=0.03 > band(0.04)? NO → within band
        metric.update(
            state={"val": 1.97, "time": 0.005},
            reference={"val_ref": 2.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        # t=0.007: still within band, dwell = 0.007-0.005 = 0.002 >= 0.001 → settled!
        metric.update(
            state={"val": 1.98, "time": 0.007},
            reference={"val_ref": 2.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        result = metric.compute()
        # Should have settled at t=0.005 (entry of the band run)
        assert result["settling_time_val"] == pytest.approx(0.005, abs=1e-9)

    def test_settling_time_dwell_not_met_returns_inf(self):
        """If dwell is never satisfied, returns inf."""
        # ref=2.0 → band=0.04, dwell=0.01
        metric = SettlingTime(
            tracked_key="val", band_fraction=0.02, dwell_s=0.01, time_key="time"
        )
        metric.reset()

        # Enter band at t=0.005, but leave immediately at t=0.006 (dwell=0.001 < 0.01)
        metric.update(
            state={"val": 1.97, "time": 0.005},
            reference={"val_ref": 2.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        metric.update(
            state={"val": 0.0, "time": 0.006},
            reference={"val_ref": 2.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        result = metric.compute()
        assert result["settling_time_val"] == float("inf")

    def test_settling_time_returns_inf_when_never_within(self):
        """Returns inf when signal never enters the band."""
        metric = SettlingTime(
            tracked_key="val", band_fraction=0.02, dwell_s=0.001, time_key="time"
        )
        metric.reset()

        # Error is always large (well outside 2% band of ref=2.0 → band=0.04)
        for t in [0.0, 0.01, 0.02]:
            metric.update(
                state={"val": 0.0, "time": t},
                reference={"val_ref": 2.0},
                _action={},
                _next_state={},
                _controller_info={},
            )

        result = metric.compute()
        assert result["settling_time_val"] == float("inf")

    def test_overshoot_computation(self):
        """Overshoot returns positive percentage when max exceeds final ref."""
        metric = Overshoot(tracked_key="val")
        metric.reset()

        # Step up to reference 1.0 with a peak at 1.5
        metric.update(
            state={"val": 0.5},
            reference={"val_ref": 1.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        metric.update(
            state={"val": 1.5},
            reference={"val_ref": 1.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        overshoot = metric.compute()
        assert overshoot == pytest.approx(50.0, abs=1e-6)

    def test_overshoot_negative_step_directionality(self):
        """Overshoot for negative step: deviation below target (undershoot) as % of |step|."""
        metric = Overshoot(tracked_key="val")
        metric.reset()

        # Step down to reference -1.0 with undershoot to -1.4
        metric.update(
            state={"val": 0.0},
            reference={"val_ref": -1.0},
            _action={},
            _next_state={},
            _controller_info={},
        )
        metric.update(
            state={"val": -1.4},
            reference={"val_ref": -1.0},
            _action={},
            _next_state={},
            _controller_info={},
        )

        overshoot = metric.compute()
        # (step_ref - trough) / |step_ref| * 100 = (-1 - (-1.4)) / 1 * 100 = 40%
        assert overshoot == pytest.approx(40.0, abs=1e-6)

    def test_multi_step_settling_time_reports_worst_and_consistency(self):
        """MultiStepSettlingTime tracks each transition independently."""
        metric = MultiStepSettlingTime(
            tracked_key="val", band_fraction=0.02, dwell_s=0.001, time_key="time"
        )
        metric.reset()

        samples = [
            # Step 1: 0 -> 2 settles quickly at 0.001
            (0.000, 0.0, 0.0),
            (0.001, 2.0, 1.99),  # in-band entry
            (0.003, 2.0, 2.00),  # dwell satisfied
            # Step 2: 2 -> -2 never settles (always outside 2% of 4A => 0.08A)
            (0.010, -2.0, -1.60),
            (0.012, -2.0, -1.70),
            # Step 3: -2 -> 1 settles at 0.020
            (0.020, 1.0, 0.95),  # in-band entry (band=0.06)
            (0.022, 1.0, 1.00),  # dwell satisfied
        ]
        for t, ref, meas in samples:
            metric.update(
                state={"val": meas, "time": t},
                reference={"val_ref": ref},
                _action={},
                _next_state={},
                _controller_info={},
            )

        result = metric.compute()
        assert result["multi_step_settling_time_val_num_steps"] == pytest.approx(3.0)
        assert result["multi_step_settling_time_val_num_settled"] == pytest.approx(2.0)
        assert result["multi_step_settling_time_val_worst"] == float("inf")
        assert result["multi_step_settling_time_val_mean"] == pytest.approx(
            (0.001 + 0.020) / 2.0, abs=1e-9
        )
        assert result["multi_step_settling_time_val_std"] == pytest.approx(
            0.0095, abs=1e-9
        )

    def test_multi_step_overshoot_reports_worst_and_mean(self):
        """MultiStepOvershoot computes per-step overshoot statistics."""
        metric = MultiStepOvershoot(tracked_key="val")
        metric.reset()

        samples = [
            # Step 1: 0 -> 2, peak at 2.4 => 20%
            (0.000, 0.0, 0.0),
            (0.001, 2.0, 2.4),
            (0.002, 2.0, 2.1),
            # Step 2: 2 -> -2, peak negative overshoot at -2.8 => 20%
            (0.010, -2.0, -2.8),
            (0.011, -2.0, -2.1),
            # Step 3: -2 -> 1, no overshoot
            (0.020, 1.0, 0.9),
            (0.021, 1.0, 1.0),
        ]
        for t, ref, meas in samples:
            metric.update(
                state={"val": meas, "time": t},
                reference={"val_ref": ref},
                _action={},
                _next_state={},
                _controller_info={},
            )

        result = metric.compute()
        assert result["multi_step_overshoot_val_num_steps"] == pytest.approx(3.0)
        assert result["multi_step_overshoot_val_worst"] == pytest.approx(20.0, abs=1e-9)
        assert result["multi_step_overshoot_val_mean"] == pytest.approx(
            40.0 / 3.0, abs=1e-9
        )


class TestMultiStepTrackingMetrics:
    """Tests for multi-step ITAE and RMS tracking metrics."""

    def test_multi_step_itae_global_and_per_step(self):
        metric = MultiStepITAE(tracked_keys=["val"], time_key="time")
        metric.reset()

        # Two steps:
        # step1 (ref=1): t=0.0 -> 0.1, error 0.5
        # step2 (ref=2): t=0.2 -> 0.3, error 1.0
        samples = [
            (0.0, 1.0, 0.5),
            (0.1, 1.0, 0.5),
            (0.2, 2.0, 1.0),
            (0.3, 2.0, 1.0),
        ]
        for t, ref, meas in samples:
            metric.update(
                state={"val": meas, "time": t},
                reference={"val_ref": ref},
                _action={},
                _next_state={},
                _controller_info={},
            )

        result = metric.compute()
        # Global ITAE: 0.5*0.1*0.1 + 1.0*0.2*0.1 + 1.0*0.3*0.1 = 0.055
        assert result["multi_step_itae_val_global"] == pytest.approx(0.055, abs=1e-9)
        assert result["multi_step_itae_val_num_steps"] == pytest.approx(2.0)
        # Per-step ITAE values: [0.005, 0.01]
        assert result["multi_step_itae_val_per_step_mean"] == pytest.approx(
            0.0075, abs=1e-9
        )
        assert result["multi_step_itae_val_per_step_worst"] == pytest.approx(
            0.01, abs=1e-9
        )

    def test_multi_step_rms_global_and_per_step(self):
        metric = MultiStepRMS(tracked_keys=["val"])
        metric.reset()

        # Two steps:
        # step1 errors = [1,1] => RMS=1
        # step2 errors = [2,2] => RMS=2
        samples = [
            (0.0, 1.0, 0.0),
            (0.1, 1.0, 0.0),
            (0.2, 2.0, 0.0),
            (0.3, 2.0, 0.0),
        ]
        for t, ref, meas in samples:
            metric.update(
                state={"val": meas, "time": t},
                reference={"val_ref": ref},
                _action={},
                _next_state={},
                _controller_info={},
            )

        result = metric.compute()
        # Global RMS = sqrt((1^2+1^2+2^2+2^2)/4) = sqrt(2.5)
        assert result["multi_step_rms_val_global"] == pytest.approx(
            2.5**0.5, abs=1e-9
        )
        assert result["multi_step_rms_val_num_steps"] == pytest.approx(2.0)
        assert result["multi_step_rms_val_per_step_mean"] == pytest.approx(1.5, abs=1e-9)
        assert result["multi_step_rms_val_per_step_worst"] == pytest.approx(
            2.0, abs=1e-9
        )
