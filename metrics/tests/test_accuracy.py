"""
Unit tests for accuracy metrics.

Tests ITAE, IAE, ISE, MAE, RMSE, steady-state error calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_metrics import compute_accuracy_metrics


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
        i_q = np.ones(n) * 1.5  # Constant 0.5A error
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
        
        # Early error
        i_q_early = i_q_ref.copy()
        i_q_early[:n//4] += 0.5  # Error in first quarter
        
        # Late error
        i_q_late = i_q_ref.copy()
        i_q_late[-n//4:] += 0.5  # Error in last quarter
        
        metrics_early = compute_accuracy_metrics(time, i_d, i_q_early, i_d_ref, i_q_ref)
        metrics_late = compute_accuracy_metrics(time, i_d, i_q_late, i_d_ref, i_q_ref)
        
        # Late error should have higher ITAE (penalized more)
        assert metrics_late.ITAE_iq > metrics_early.ITAE_iq
    
    def test_rmse_larger_than_mae(self, constant_error):
        """For varying errors, RMSE should be >= MAE."""
        time, i_d, i_q, i_d_ref, i_q_ref = constant_error
        # Add variation
        i_q = i_q + np.sin(time * 100) * 0.1
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        
        assert metrics.RMSE_iq >= metrics.MAE_iq
    
    def test_steady_state_error_uses_last_samples(self, constant_error):
        """SS error should be based on final values."""
        time, i_d, i_q, i_d_ref, i_q_ref = constant_error
        # Perfect at end, bad at start
        i_q = i_q_ref.copy()
        i_q[:len(i_q)//2] = 0  # Bad first half
        
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        
        # SS error should be small (based on end)
        assert metrics.SS_error_iq < 0.1


class TestAccuracyEdgeCases:
    """Test edge cases for accuracy metrics."""
    
    def test_single_sample(self):
        """Should handle single sample gracefully."""
        time = np.array([0.0])
        i_d = np.array([0.0])
        i_q = np.array([1.0])
        i_d_ref = np.array([0.0])
        i_q_ref = np.array([2.0])
        
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        assert metrics.MAE_iq == pytest.approx(1.0)
    
    def test_negative_errors(self):
        """Should handle negative errors (actual < reference)."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        i_d = np.zeros(n)
        i_q = np.ones(n) * 1.0  # Below reference
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        
        assert metrics.MAE_iq == pytest.approx(1.0)  # |1-2| = 1


# Run with: pytest metrics/tests/test_accuracy.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

