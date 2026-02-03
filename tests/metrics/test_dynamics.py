"""
Unit tests for dynamics metrics.

Tests rise time, settling time, overshoot, peak time calculations.

"""

import numpy as np
import pytest

from embark.metrics.benchmark_metrics import compute_dynamics_metrics


class TestDynamicsMetrics:
    """Test dynamics metric calculations."""

    @pytest.fixture
    def ideal_step_response(self):
        """Create ideal first-order step response."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)

        step_time = 0.01
        tau = 0.005
        target = 5.0

        i_q = np.where(
            time >= step_time,
            target * (1 - np.exp(-(time - step_time) / tau)),
            0.0,
        )
        i_q_ref = np.where(time >= step_time, target, 0.0)
        i_d = np.zeros(n)
        i_d_ref = np.zeros(n)

        return time, i_d, i_q, i_d_ref, i_q_ref, step_time

    @pytest.fixture
    def step_with_overshoot(self):
        """Create underdamped step response with overshoot."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)

        step_time = 0.01
        omega_n = 500
        zeta = 0.3
        target = 5.0

        t_rel = np.maximum(0, time - step_time)
        omega_d = omega_n * np.sqrt(1 - zeta**2)

        i_q = np.where(
            time >= step_time,
            target
            * (
                1
                - np.exp(-zeta * omega_n * t_rel)
                * (
                    np.cos(omega_d * t_rel)
                    + zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t_rel)
                )
            ),
            0.0,
        )
        i_q_ref = np.where(time >= step_time, target, 0.0)
        i_d = np.zeros(n)
        i_d_ref = np.zeros(n)

        return time, i_d, i_q, i_d_ref, i_q_ref, step_time

    def test_rise_time_positive(self, ideal_step_response):
        """Rise time should be positive."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = ideal_step_response
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)

        assert metrics.rise_time_iq > 0

    def test_rise_time_reasonable(self, ideal_step_response):
        """Rise time should be approximately 2.2*tau for first-order."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = ideal_step_response
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)

        assert metrics.rise_time_iq == pytest.approx(0.011, rel=0.2)
