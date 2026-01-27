"""
Unit tests for PMSMEnv Gymnasium wrapper.

Tests the environment in isolation without NeuroBench dependencies.
"""

import numpy as np

from embark.benchmark.pmsm_env import PMSMEnv


class TestPMSMEnvInit:
    """Test environment initialization."""

    def test_creates_successfully(self):
        """Environment can be created with default parameters."""
        env = PMSMEnv()
        assert env is not None
        env.close()

    def test_creates_with_custom_params(self):
        """Environment accepts custom parameters."""
        env = PMSMEnv(
            n_rpm=1500,
            i_d_ref=-1.0,
            i_q_ref=3.0,
            max_steps=1000,
        )
        assert env.n_rpm == 1500
        assert env.i_d_ref == -1.0
        assert env.i_q_ref == 3.0
        assert env.max_steps == 1000
        env.close()

    def test_observation_space_defined(self):
        """Observation space is properly defined."""
        env = PMSMEnv()
        assert env.observation_space is not None
        assert env.observation_space.shape == (4,)
        env.close()

    def test_action_space_defined(self):
        """Action space is properly defined."""
        env = PMSMEnv()
        assert env.action_space is not None
        assert env.action_space.shape == (2,)
        env.close()


class TestPMSMEnvReset:
    """Test environment reset behavior."""

    def test_reset_returns_state_and_info(self):
        """Reset returns (state, info) tuple."""
        env = PMSMEnv()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        state, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    def test_reset_state_shape(self):
        """Reset returns state with correct shape."""
        env = PMSMEnv()
        state, _ = env.reset()
        assert state.shape == (4,)
        env.close()

    def test_reset_clears_step_count(self):
        """Reset clears step counter."""
        env = PMSMEnv(max_steps=100)
        env.reset()
        assert env.current_step == 0
        env.close()
