"""
Unit tests for PMSMEnv Gymnasium wrapper.

Tests the environment in isolation without NeuroBench dependencies.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmsm_env import PMSMEnv


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
        assert env.observation_space.shape == (4,)  # i_d, i_q, e_d, e_q
        env.close()
    
    def test_action_space_defined(self):
        """Action space is properly defined."""
        env = PMSMEnv()
        assert env.action_space is not None
        assert env.action_space.shape == (2,)  # u_d, u_q (normalized)
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
        assert state.shape == (4,)  # i_d, i_q, e_d, e_q
        env.close()
    
    def test_reset_clears_step_count(self):
        """Reset clears step counter."""
        env = PMSMEnv(max_steps=100)
        env.reset()
        for _ in range(10):
            env.step(np.array([0.0, 0.0]))
        env.reset()
        assert env.current_step == 0
        env.close()
    
    def test_reset_with_seed(self):
        """Reset accepts seed parameter."""
        env = PMSMEnv()
        state1, _ = env.reset(seed=42)
        state2, _ = env.reset(seed=42)
        # Note: GEM may not be fully deterministic
        assert state1 is not None
        assert state2 is not None
        env.close()


class TestPMSMEnvStep:
    """Test environment step behavior."""
    
    def test_step_returns_five_values(self):
        """Step returns (state, reward, done, truncated, info)."""
        env = PMSMEnv()
        env.reset()
        result = env.step(np.array([0.0, 0.0]))
        assert isinstance(result, tuple)
        assert len(result) == 5
        state, reward, done, truncated, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_step_advances_counter(self):
        """Step increments step counter."""
        env = PMSMEnv()
        env.reset()
        assert env.current_step == 0
        env.step(np.array([0.0, 0.0]))
        assert env.current_step == 1
        env.step(np.array([0.0, 0.0]))
        assert env.current_step == 2
        env.close()
    
    def test_step_terminates_at_max(self):
        """Step terminates episode at max_steps."""
        env = PMSMEnv(max_steps=10)
        env.reset()
        for i in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        # Episode ends via terminated flag when reaching max_steps
        assert terminated is True
        env.close()
    
    def test_step_accepts_normalized_actions(self):
        """Step accepts actions in [-1, 1] range."""
        env = PMSMEnv()
        env.reset()
        # Should not raise
        env.step(np.array([0.5, -0.5]))
        env.step(np.array([1.0, 1.0]))
        env.step(np.array([-1.0, -1.0]))
        env.close()


class TestPMSMEnvState:
    """Test state extraction and normalization."""
    
    def test_state_contains_currents(self):
        """State contains i_d and i_q."""
        env = PMSMEnv()
        state, _ = env.reset()
        # First two elements are i_d, i_q
        i_d, i_q = state[0], state[1]
        assert isinstance(i_d, (int, float, np.floating))
        assert isinstance(i_q, (int, float, np.floating))
        env.close()
    
    def test_state_contains_errors(self):
        """State contains tracking errors e_d, e_q."""
        env = PMSMEnv(i_d_ref=0.0, i_q_ref=2.0)
        state, _ = env.reset()
        # Elements 2, 3 are e_d, e_q
        e_d, e_q = state[2], state[3]
        # Initially, error should exist (not tracking yet)
        assert isinstance(e_d, (int, float, np.floating))
        assert isinstance(e_q, (int, float, np.floating))
        env.close()


class TestPMSMEnvEpisodeData:
    """Test episode data collection."""
    
    def test_get_episode_data_returns_list(self):
        """get_episode_data returns list of dicts."""
        env = PMSMEnv()
        env.reset()
        for _ in range(5):
            env.step(np.array([0.0, 0.0]))
        data = env.get_episode_data()
        assert isinstance(data, list)
        assert len(data) == 5
        env.close()
    
    def test_episode_data_contains_required_keys(self):
        """Episode data contains all required keys."""
        env = PMSMEnv()
        env.reset()
        env.step(np.array([0.0, 0.0]))
        data = env.get_episode_data()
        # Core control data needed for benchmark metrics
        required_keys = ['i_d', 'i_q', 'i_d_ref', 'i_q_ref', 'e_d', 'e_q', 
                         'u_d', 'u_q', 'time', 'time_in_range']
        for key in required_keys:
            assert key in data[0], f"Missing key: {key}"
        env.close()


# Run with: pytest benchmark/tests/test_pmsm_env.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

