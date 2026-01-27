"""
Unit tests for safety and constraint metrics.

Tests current/voltage violations, safety margins.
"""

import numpy as np
import pytest

from embark.metrics.benchmark_metrics import compute_safety_metrics, DEFAULT_MOTOR


class TestSafetyMetrics:
    """Test safety metric calculations."""

    @pytest.fixture
    def safe_data(self):
        """Data within limits."""
        dt = 1e-4
        n = 100
        time = np.arange(n) * dt
        i_d = np.zeros(n)
        i_q = np.full(n, 1.0)  # Well below limit
        u_d = np.zeros(n)
        u_q = np.full(n, 10.0)
        return time, i_d, i_q, u_d, u_q

    @pytest.fixture
    def unsafe_data(self):
        """Data violating current limits."""
        dt = 1e-4
        n = 100
        time = np.arange(n) * dt
        i_d = np.zeros(n)
        # Exceed limit
        limit = DEFAULT_MOTOR.I_max
        i_q = np.full(n, limit * 1.5)
        u_d = np.zeros(n)
        u_q = np.zeros(n)
        return time, i_d, i_q, u_d, u_q

    def test_no_violations_safe_data(self, safe_data):
        """Safe data should have 0 violations."""
        time, i_d, i_q, u_d, u_q = safe_data
        metrics = compute_safety_metrics(time, i_d, i_q, u_d, u_q)

        assert metrics.current_violations == 0
        assert metrics.voltage_violations == 0
        assert metrics.current_violation_rate == 0.0

    def test_violations_detected(self, unsafe_data):
        """Unsafe data should report violations."""
        time, i_d, i_q, u_d, u_q = unsafe_data
        metrics = compute_safety_metrics(time, i_d, i_q, u_d, u_q)

        assert metrics.current_violations == 100
        assert metrics.current_violation_rate == 100.0
        assert metrics.current_max_excess > 0

    def test_oscillation_detection(self):
        """Should detect rapid oscillations."""
        dt = 1e-4
        n = 1000
        time = np.arange(n) * dt
        i_d = np.zeros(n)

        # High frequency oscillation
        freq = 500  # Hz
        i_q = np.sin(2 * np.pi * freq * time)
        u_d = np.zeros(n)
        u_q = np.zeros(n)

        metrics = compute_safety_metrics(time, i_d, i_q, u_d, u_q)
        assert metrics.oscillation_detected
