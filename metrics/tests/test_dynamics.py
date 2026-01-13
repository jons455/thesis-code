"""
Unit tests for dynamics metrics.

Tests rise time, settling time, overshoot, peak time calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_metrics import compute_dynamics_metrics


class TestDynamicsMetrics:
    """Test dynamics metric calculations."""
    
    @pytest.fixture
    def ideal_step_response(self):
        """Create ideal first-order step response."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        
        step_time = 0.01
        tau = 0.005  # 5ms time constant
        target = 5.0
        
        i_q = np.where(
            time >= step_time,
            target * (1 - np.exp(-(time - step_time) / tau)),
            0.0
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
        omega_n = 500  # Natural frequency
        zeta = 0.3  # Low damping = more overshoot
        target = 5.0
        
        t_rel = np.maximum(0, time - step_time)
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        
        i_q = np.where(
            time >= step_time,
            target * (1 - np.exp(-zeta * omega_n * t_rel) * 
                     (np.cos(omega_d * t_rel) + 
                      zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t_rel))),
            0.0
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
        
        # For first-order system, rise time (10-90%) ≈ 2.2 * tau = 11 ms
        expected_rise_time = 0.011
        assert metrics.rise_time_iq == pytest.approx(expected_rise_time, rel=0.3)
    
    def test_settling_time_positive(self, ideal_step_response):
        """Settling time should be positive."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = ideal_step_response
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        assert metrics.settling_time_iq > 0
    
    def test_settling_time_longer_than_rise(self, ideal_step_response):
        """Settling time should be >= rise time."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = ideal_step_response
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        assert metrics.settling_time_iq >= metrics.rise_time_iq
    
    def test_overshoot_zero_for_first_order(self, ideal_step_response):
        """First-order system should have no overshoot."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = ideal_step_response
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        assert metrics.overshoot_iq == pytest.approx(0.0, abs=0.1)
    
    def test_overshoot_positive_for_underdamped(self, step_with_overshoot):
        """Underdamped system should have positive overshoot."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = step_with_overshoot
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        assert metrics.overshoot_iq > 0
    
    def test_peak_time_exists(self, step_with_overshoot):
        """Peak time should be defined for underdamped response."""
        time, i_d, i_q, i_d_ref, i_q_ref, step_time = step_with_overshoot
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        assert metrics.peak_time_iq > 0


class TestDynamicsEdgeCases:
    """Test edge cases for dynamics metrics."""
    
    def test_no_step(self):
        """Should handle constant reference (no step change)."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        
        # Constant reference from t=0 (already at target)
        i_d = np.zeros(n)
        i_q = np.ones(n) * 2.0  # Already at reference
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        step_time = 0.0  # Step at t=0
        
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        # Should return metrics (rise_time ~ 0 since already at target)
        assert metrics is not None
    
    def test_instant_response(self):
        """Should handle instant step response (zero rise time)."""
        dt = 1e-4
        time = np.arange(0, 0.1, dt)
        n = len(time)
        
        step_time = 0.01
        i_q_ref = np.where(time >= step_time, 5.0, 0.0)
        i_q = i_q_ref.copy()  # Perfect tracking
        i_d = np.zeros(n)
        i_d_ref = np.zeros(n)
        
        metrics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
        
        # Rise time should be minimal
        assert metrics.rise_time_iq < 0.001  # Less than 1 ms


# Run with: pytest metrics/tests/test_dynamics.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

