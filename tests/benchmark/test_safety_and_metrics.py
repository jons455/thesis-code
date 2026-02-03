"""Unit tests for safety limits and metric correctness."""

import pytest
from unittest.mock import MagicMock

from embark.benchmark.tasks.pmsm_current_control import PMSMCurrentControlTask, SafetyLimits
from embark.benchmark.metrics.accumulators.tracking import TrackingRMSE, TrackingMAE
from embark.benchmark.interfaces import StateDict, ReferenceDict, ActionDict


class TestSafetyLimits:
    """Test safety limit enforcement."""

    def test_check_action_limits(self):
        """Test action limits (voltage)."""
        limits = SafetyLimits(max_voltage_v=50.0)
        
        # Safe actions
        assert limits.check_action({"v_d": 49.0, "v_q": -49.0}) is None
        
        # Unsafe actions
        assert "voltage_limit_exceeded" in limits.check_action({"v_d": 51.0, "v_q": 0.0})
        assert "voltage_limit_exceeded" in limits.check_action({"v_d": 0.0, "v_q": -51.0})
        
        # NaN actions
        assert "action_nan" in limits.check_action({"v_d": float("nan"), "v_q": 0.0})

    def test_check_state_limits(self):
        """Test state limits (current, speed)."""
        limits = SafetyLimits(max_current_a=20.0, max_speed_rpm=1000.0)
        
        # Safe state
        safe_state = {"i_d": 19.0, "i_q": 19.0, "omega": 100.0} # omega in rad/s, check logic
        # 100 rad/s * 60 / 2pi approx 955 rpm -> Safe
        assert limits.check_state(safe_state) is None
        
        # Unsafe current
        assert "current_limit_exceeded" in limits.check_state({"i_d": 21.0, "i_q": 0.0, "omega": 0.0})
        assert "current_limit_exceeded" in limits.check_state({"i_d": 0.0, "i_q": -21.0, "omega": 0.0})
        
        # Unsafe speed
        # 110 rad/s * 9.55 = 1050 rpm -> Unsafe
        assert "speed_limit_exceeded" in limits.check_state({"i_d": 0.0, "i_q": 0.0, "omega": 110.0})
        
        # NaN state
        assert "state_nan" in limits.check_state({"i_d": float("nan"), "i_q": 0.0, "omega": 0.0})

    def test_task_integration_safety(self):
        """Test that task terminates on safety violation."""
        # Create a mock physics engine that just returns the input action as state (for simplicity)
        # or we can use the real one but force a violation.
        # Easier: Mock the physics engine.
        
        limits = SafetyLimits(max_voltage_v=10.0)
        task = PMSMCurrentControlTask(safety_limits=limits)
        
        # Inject mock physics engine
        mock_physics = MagicMock()
        mock_physics.reset.return_value = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0, "time": 0.0}
        mock_physics.step.return_value = ({"i_d": 0.0, "i_q": 0.0, "omega": 0.0, "time": 0.1}, {})
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


class TestMetricCorrectness:
    """Test mathematical correctness of metric accumulators."""

    def test_tracking_rmse_calculation(self):
        """Verify RMSE calculation with known values."""
        metric = TrackingRMSE(tracked_keys=["val"])
        metric.reset()
        
        # Error sequence: 3.0, 4.0
        # Squared errors: 9.0, 16.0
        # Mean squared error: 12.5
        # RMSE: sqrt(12.5) approx 3.5355
        
        # Step 1: Ref=10, State=7 -> Error=3
        metric.update(
            state={"val": 7.0},
            reference={"val_ref": 10.0},
            action={}, next_state={}, controller_info={}
        )
        
        # Step 2: Ref=10, State=6 -> Error=4
        metric.update(
            state={"val": 6.0},
            reference={"val_ref": 10.0},
            action={}, next_state={}, controller_info={}
        )
        
        result = metric.compute()
        expected = (12.5) ** 0.5
        assert result["rmse_val"] == pytest.approx(expected, abs=1e-6)

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
            action={}, next_state={}, controller_info={}
        )
        metric.update(
            state={"val": 6.0},
            reference={"val_ref": 10.0},
            action={}, next_state={}, controller_info={}
        )
        
        result = metric.compute()
        assert result["mae_val"] == pytest.approx(3.5, abs=1e-6)
