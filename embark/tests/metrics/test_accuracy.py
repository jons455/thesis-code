"""
Unit tests for accuracy metrics.

Tests ITAE, IAE, ISE, MAE, RMSE, steady-state error calculations.
"""

import numpy as np
import pytest

from embark.metrics.benchmark_metrics import compute_accuracy_metrics


class TestAccuracyMetrics:
    """Test accuracy metric calculations."""

    @pytest.fixture
    def perfect_tracking(self):
        """Data with perfect tracking (zero error)."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        i_d = np.zeros(n)
        i_q = np.ones(n) * 2.0
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        return time, i_d, i_q, i_d_ref, i_q_ref

    @pytest.fixture
    def constant_error(self):
        """Data with constant tracking error."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        i_d = np.zeros(n)
        i_q = np.ones(n) * 1.5
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        return time, i_d, i_q, i_d_ref, i_q_ref

    def test_perfect_tracking_zero_errors(self, perfect_tracking):
        """Perfect tracking should have zero error metrics."""
        time, i_d, i_q, i_d_ref, i_q_ref = perfect_tracking
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)

        assert metrics.MAE_iq == pytest.approx(0.0, abs=1e-10)
        assert metrics.MAE_id == pytest.approx(0.0, abs=1e-10)
        assert metrics.RMSE_iq == pytest.approx(0.0, abs=1e-10)
        assert metrics.SS_error_iq == pytest.approx(0.0, abs=1e-10)

    def test_constant_error_mae(self, constant_error):
        """Constant 0.5A error should give MAE = 0.5."""
        time, i_d, i_q, i_d_ref, i_q_ref = constant_error
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)

        assert metrics.MAE_iq == pytest.approx(0.5, abs=1e-3)

    def test_itae_penalizes_late_errors(self):
        """ITAE should penalize late errors more than early ones."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)

        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        i_d = np.zeros(n)

        i_q_early = i_q_ref.copy()
        i_q_early[: n // 4] += 0.5

        i_q_late = i_q_ref.copy()
        i_q_late[-n // 4 :] += 0.5

        metrics_early = compute_accuracy_metrics(time, i_d, i_q_early, i_d_ref, i_q_ref)
        metrics_late = compute_accuracy_metrics(time, i_d, i_q_late, i_d_ref, i_q_ref)

        assert metrics_late.ITAE_iq > metrics_early.ITAE_iq
