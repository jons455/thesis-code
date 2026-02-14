"""
Regression tests for benchmark results.

These tests verify that results remain consistent with known-good baselines.
If a test fails, it may indicate:
1. Intentional changes (update the baseline)
2. Unintentional regressions (investigate)

Rewritten 2026-02-07 to use the current architecture:
    PMSMCurrentControlTask + PIControllerAgent
(The original PMSMEnv Gymnasium wrapper was removed in the 2026-02-03 refactoring.)

"""

import numpy as np
import pytest

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.tasks.pmsm_current_control import PMSMCurrentControlTask


class TestPIControllerBaseline:
    """
    Regression tests for PI controller performance.

    Baseline values established from validated simulations.

    """

    # Baseline performance thresholds (established 2026-01-13)
    # These are "not worse than" thresholds; may vary by platform/solver
    BASELINE = {
        "max_tracking_error_mA": 600,  # Maximum final tracking error
        "settling_time_ms": 20,  # 2% settling time
        "overshoot_percent": 30,  # Maximum overshoot
    }

    def _run_episode(self, n_rpm=1000.0, i_d_ref=0.0, i_q_ref=2.0, max_steps=500):
        """Run a PI-controlled episode and return state/reference history."""
        task = PMSMCurrentControlTask.from_config(
            n_rpm=n_rpm, i_d_ref=i_d_ref, i_q_ref=i_q_ref, max_steps=max_steps
        )
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)

        state, reference = task.reset()
        agent.reset()

        states = [state]
        references = [reference]

        for _ in range(max_steps):
            action = agent(state, reference)
            state, reference, done = task.step(action)
            states.append(state)
            references.append(reference)
            if done:
                break

        return states, references

    def test_tracking_error_regression(self):
        """Tracking error should not exceed baseline."""
        states, references = self._run_episode()

        final_state = states[-1]
        final_ref = references[-1]

        final_error_mA = (
            np.sqrt(
                (final_ref["i_d_ref"] - final_state["i_d"]) ** 2
                + (final_ref["i_q_ref"] - final_state["i_q"]) ** 2
            )
            * 1000
        )

        assert final_error_mA < self.BASELINE["max_tracking_error_mA"], (
            f"Tracking error {final_error_mA:.2f} mA exceeds baseline "
            f"{self.BASELINE['max_tracking_error_mA']} mA"
        )

    def test_controller_convergence(self):
        """Controller should converge within reasonable time."""
        states, references = self._run_episode()

        errors = []
        for s, r in zip(states, references):
            e_d = r["i_d_ref"] - s["i_d"]
            e_q = r["i_q_ref"] - s["i_q"]
            errors.append(np.sqrt(e_d**2 + e_q**2))

        # Error should decrease over time (converge)
        early_error = np.mean(errors[10:50])
        late_error = np.mean(errors[-50:])

        assert (
            late_error < early_error
        ), f"Controller did not converge: early={early_error:.4f}, late={late_error:.4f}"

    def test_no_instability(self):
        """Controller should not become unstable."""
        states, _ = self._run_episode()

        max_current = 0.0
        for s in states:
            i_d = s["i_d"]
            i_q = s["i_q"]
            current_mag = np.sqrt(i_d**2 + i_q**2)
            max_current = max(max_current, current_mag)

        # Current should not exceed limit (10.8A for our motor)
        assert max_current < 12.0, f"Current exceeded safe limit: {max_current:.2f} A"


class TestMatlabEquivalence:
    """
    Regression tests comparing to MATLAB reference.

    These ensure GEM simulation matches validated MATLAB results.

    """

    # Known good values from MATLAB validation (Run 003)
    MATLAB_REFERENCE = {
        "n_rpm": 1500,
        "i_q_ref": 3.5,
        "expected_steady_state_iq": 3.5,  # After settling
        "tolerance": 0.1,  # Acceptable deviation in A
    }

    def test_steady_state_matches_matlab(self):
        """Steady-state current should match MATLAB within tolerance."""
        task = PMSMCurrentControlTask.from_config(
            n_rpm=self.MATLAB_REFERENCE["n_rpm"],
            i_d_ref=0.0,
            i_q_ref=self.MATLAB_REFERENCE["i_q_ref"],
            max_steps=1000,
        )
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)

        state, reference = task.reset()
        agent.reset()

        states = []
        for _ in range(1000):
            action = agent(state, reference)
            state, reference, done = task.step(action)
            states.append(state)
            if done:
                break

        # Check steady-state i_q (average of last 100 samples)
        steady_state_iq = np.mean([s["i_q"] for s in states[-100:]])

        deviation = abs(
            steady_state_iq - self.MATLAB_REFERENCE["expected_steady_state_iq"]
        )
        assert (
            deviation < self.MATLAB_REFERENCE["tolerance"]
        ), f"Steady-state i_q={steady_state_iq:.3f}A deviates from MATLAB reference"


# Run with: pytest tests/test_regression.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
