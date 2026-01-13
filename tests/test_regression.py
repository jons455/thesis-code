"""
Regression tests for benchmark results.

These tests verify that results remain consistent with known-good baselines.
If a test fails, it may indicate:
1. Intentional changes (update the baseline)
2. Unintentional regressions (investigate)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "benchmark"))
sys.path.insert(0, str(project_root / "metrics"))


class TestPIControllerBaseline:
    """
    Regression tests for PI controller performance.
    
    Baseline values established from validated simulations.
    """
    
    # Baseline performance thresholds (established 2026-01-13)
    # These are "not worse than" thresholds, not exact matches
    BASELINE = {
        'max_tracking_error_mA': 100,  # Maximum final tracking error
        'settling_time_ms': 20,         # 2% settling time
        'overshoot_percent': 30,        # Maximum overshoot
    }
    
    def test_tracking_error_regression(self):
        """Tracking error should not exceed baseline."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        env = PMSMEnv(
            n_rpm=1000,
            i_d_ref=0.0,
            i_q_ref=2.0,
            max_steps=500,
        )
        agent = PIControllerAgent()
        
        state, _ = env.reset()
        agent.reset()
        
        for _ in range(500):
            action = agent(state)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        
        data = env.get_episode_data()
        env.close()
        
        # Compute final error in mA
        final_error_mA = np.sqrt(data[-1]['e_d']**2 + data[-1]['e_q']**2) * 1000
        
        assert final_error_mA < self.BASELINE['max_tracking_error_mA'], \
            f"Tracking error {final_error_mA:.2f} mA exceeds baseline {self.BASELINE['max_tracking_error_mA']} mA"
    
    def test_controller_convergence(self):
        """Controller should converge within reasonable time."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        env = PMSMEnv(
            n_rpm=1000,
            i_d_ref=0.0,
            i_q_ref=2.0,
            max_steps=500,
        )
        agent = PIControllerAgent()
        
        state, _ = env.reset()
        agent.reset()
        
        errors = []
        for _ in range(500):
            action = agent(state)
            state, _, done, truncated, _ = env.step(action)
            errors.append(np.sqrt(state[2]**2 + state[3]**2))  # e_d, e_q
            if done or truncated:
                break
        
        env.close()
        
        # Error should decrease over time (converge)
        early_error = np.mean(errors[10:50])
        late_error = np.mean(errors[-50:])
        
        assert late_error < early_error, \
            f"Controller did not converge: early={early_error:.4f}, late={late_error:.4f}"
    
    def test_no_instability(self):
        """Controller should not become unstable."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        env = PMSMEnv(
            n_rpm=1000,
            i_d_ref=0.0,
            i_q_ref=2.0,
            max_steps=500,
        )
        agent = PIControllerAgent()
        
        state, _ = env.reset()
        agent.reset()
        
        max_current = 0
        for _ in range(500):
            action = agent(state)
            state, _, done, truncated, _ = env.step(action)
            i_d, i_q = state[0], state[1]
            current_mag = np.sqrt(i_d**2 + i_q**2)
            max_current = max(max_current, current_mag)
            if done or truncated:
                break
        
        env.close()
        
        # Current should not exceed limit (10.8A for our motor)
        assert max_current < 12.0, f"Current exceeded safe limit: {max_current:.2f} A"


class TestMetricsConsistency:
    """
    Regression tests for metric calculations.
    
    Verifies that metric calculations remain consistent.
    """
    
    def test_itae_formula(self):
        """ITAE calculation should match expected formula."""
        from benchmark_metrics import compute_accuracy_metrics
        
        # Simple known case: constant error of 1.0 for 1 second
        dt = 0.001
        time = np.arange(0, 1.0, dt)
        n = len(time)
        
        i_d = np.zeros(n)
        i_q = np.ones(n)  # Constant 1A
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2  # Reference 2A, so error = 1A
        
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        
        # ITAE = ∫ t * |e(t)| dt = ∫₀¹ t * 1 dt = 0.5
        # (numerical integration may differ slightly)
        expected_itae = 0.5
        assert metrics.ITAE_iq == pytest.approx(expected_itae, rel=0.1)
    
    def test_mae_formula(self):
        """MAE calculation should match expected formula."""
        from benchmark_metrics import compute_accuracy_metrics
        
        dt = 0.001
        time = np.arange(0, 1.0, dt)
        n = len(time)
        
        i_d = np.zeros(n)
        i_q = np.ones(n) * 1.5  # Constant 1.5A, error = 0.5A
        i_d_ref = np.zeros(n)
        i_q_ref = np.ones(n) * 2.0
        
        metrics = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
        
        # MAE = mean(|e|) = 0.5
        assert metrics.MAE_iq == pytest.approx(0.5, rel=0.01)


class TestMatlabEquivalence:
    """
    Regression tests comparing to MATLAB reference.
    
    These ensure GEM simulation matches validated MATLAB results.
    """
    
    # Known good values from MATLAB validation (Run 003)
    MATLAB_REFERENCE = {
        'n_rpm': 1500,
        'i_q_ref': 3.5,
        'expected_steady_state_iq': 3.5,  # After settling
        'tolerance': 0.1,  # Acceptable deviation in A
    }
    
    def test_steady_state_matches_matlab(self):
        """Steady-state current should match MATLAB within tolerance."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        env = PMSMEnv(
            n_rpm=self.MATLAB_REFERENCE['n_rpm'],
            i_d_ref=0.0,
            i_q_ref=self.MATLAB_REFERENCE['i_q_ref'],
            max_steps=1000,  # Longer for full settling
        )
        agent = PIControllerAgent()
        
        state, _ = env.reset()
        agent.reset()
        
        for _ in range(1000):
            action = agent(state)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        
        data = env.get_episode_data()
        env.close()
        
        # Check steady-state i_q (average of last 100 samples)
        steady_state_iq = np.mean([d['i_q'] for d in data[-100:]])
        
        deviation = abs(steady_state_iq - self.MATLAB_REFERENCE['expected_steady_state_iq'])
        assert deviation < self.MATLAB_REFERENCE['tolerance'], \
            f"Steady-state i_q={steady_state_iq:.3f}A deviates from MATLAB reference"


# Run with: pytest tests/test_regression.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

