"""Test protocol compliance for controllers."""

from __future__ import annotations

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.interfaces import DictController
from embark.benchmark.physics import PMSMConfig


def test_pi_controller_agent_is_dict_controller():
    """PIControllerAgent follows DictController protocol."""
    config = PMSMConfig()
    controller = PIControllerAgent.from_system_config(config)

    # Protocol requires these methods
    assert hasattr(controller, "__call__")
    assert hasattr(controller, "reset")
    assert hasattr(controller, "get_state")
    assert hasattr(controller, "set_state")

    # Test basic operation
    state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
    reference = {"i_d_ref": 0.0, "i_q_ref": 2.0}
    action = controller(state, reference)

    assert "v_d" in action
    assert "v_q" in action
