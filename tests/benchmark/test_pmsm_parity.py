"""Parity smoke tests between old and new PMSM implementations."""

import numpy as np
import pytest


def test_pmsm_physics_engine_produces_valid_state():
    """Test that PMSMPhysicsEngine produces valid initial state."""
    pytest.importorskip("gym_electric_motor")

    from embark.benchmark.physics import PMSMPhysicsEngine

    engine = PMSMPhysicsEngine(n_rpm=1000)
    state = engine.reset()

    # Check state contains expected keys
    assert "i_d" in state
    assert "i_q" in state
    assert "omega" in state

    # Check values are reasonable
    assert abs(state["i_d"]) < 1.0  # Near zero at init
    assert abs(state["i_q"]) < 1.0  # Near zero at init

    engine.close()
