"""Unit tests for PMSM physics engine.

Tests the PMSMPhysicsEngine which wraps GEM for PMSM simulation.
"""

import pytest

pytest.importorskip("gym_electric_motor")

from embark.benchmark.physics import PMSMConfig, PMSMPhysicsEngine  # noqa: E402


class TestPMSMPhysicsEngineInit:
    """Test PMSMPhysicsEngine initialization."""

    def test_creates_successfully(self):
        """Engine can be created with default parameters."""
        engine = PMSMPhysicsEngine()
        assert engine is not None
        engine.close()

    def test_creates_with_custom_params(self):
        """Engine accepts custom config."""
        config = PMSMConfig(i_max=15.0, u_max=60.0)
        engine = PMSMPhysicsEngine(n_rpm=2000, config=config)
        assert engine.config.i_max == 15.0
        assert engine.config.u_max == 60.0
        engine.close()

    def test_state_keys_defined(self):
        """Engine defines expected state keys."""
        engine = PMSMPhysicsEngine()
        assert "i_d" in engine.state_keys
        assert "i_q" in engine.state_keys
        engine.close()

    def test_action_keys_defined(self):
        """Engine defines expected action keys."""
        engine = PMSMPhysicsEngine()
        # Uses alpha-beta frame
        assert "v_alpha" in engine.action_keys or "v_d" in engine.action_keys
        engine.close()


class TestPMSMPhysicsEngineReset:
    """Test PMSMPhysicsEngine reset behavior."""

    def test_reset_returns_state_dict(self):
        """Reset returns state dictionary."""
        engine = PMSMPhysicsEngine()
        state = engine.reset()
        assert isinstance(state, dict)
        assert "i_d" in state
        assert "i_q" in state
        engine.close()

    def test_reset_returns_reasonable_currents(self):
        """Initial currents should be near zero."""
        engine = PMSMPhysicsEngine()
        state = engine.reset()
        assert abs(state["i_d"]) < 1.0
        assert abs(state["i_q"]) < 1.0
        engine.close()
