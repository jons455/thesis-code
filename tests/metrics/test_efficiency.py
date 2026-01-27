"""
Unit tests for efficiency metrics.

Tests copper losses, electrical power, efficiency calculations.
"""

import numpy as np
import pytest

from embark.metrics.benchmark_metrics import compute_efficiency_metrics, DEFAULT_MOTOR


class TestEfficiencyMetrics:
    """Test efficiency metric calculations."""

    @pytest.fixture
    def efficiency_data(self):
        """Synthetic data for efficiency calculation."""
        dt = 1e-4
        n = 1000
        time = np.arange(n) * dt

        # Constant operation
        i_d = np.zeros(n)
        i_q = np.full(n, 5.0)  # 5A
        u_d = np.zeros(n)
        u_q = np.full(n, 10.0)  # 10V
        speed = np.full(n, 1000.0)  # 1000 rpm

        return time, i_d, i_q, u_d, u_q, speed

    def test_copper_losses(self, efficiency_data):
        """Copper losses should match 1.5 * R * i^2 formula."""
        time, i_d, i_q, u_d, u_q, speed = efficiency_data
        metrics = compute_efficiency_metrics(time, i_d, i_q, u_d, u_q, speed)

        # P_cu = 1.5 * R_s * (i_d^2 + i_q^2)
        # R_s = DEFAULT_MOTOR.R_s (usually 2.8Ohm)
        # i^2 = 25
        expected_loss = 1.5 * DEFAULT_MOTOR.R_s * 25.0

        assert metrics.P_copper_mean == pytest.approx(expected_loss)

    def test_electrical_power(self, efficiency_data):
        """Electrical power should match 1.5 * (u_d*i_d + u_q*i_q)."""
        time, i_d, i_q, u_d, u_q, speed = efficiency_data
        metrics = compute_efficiency_metrics(time, i_d, i_q, u_d, u_q, speed)

        # P_elec = 1.5 * (0 + 10*5) = 75W
        assert metrics.P_elec_mean == pytest.approx(75.0)

    def test_zero_power_efficiency(self):
        """Efficiency should be 0 when input power is 0."""
        n = 100
        time = np.arange(n) * 1e-4
        zeros = np.zeros(n)

        metrics = compute_efficiency_metrics(time, zeros, zeros, zeros, zeros, zeros)
        assert metrics.eta_mean == 0.0

    def test_current_magnitude(self, efficiency_data):
        """Current magnitude should be correct."""
        time, i_d, i_q, u_d, u_q, speed = efficiency_data
        metrics = compute_efficiency_metrics(time, i_d, i_q, u_d, u_q, speed)

        assert metrics.i_magnitude_mean == pytest.approx(5.0)
