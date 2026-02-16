"""Unit tests for benchmark controller agents."""

from __future__ import annotations

import pytest

from embark.benchmark.agents import PIControllerAgent
from embark.utils.config import DEFAULT_PMSM


class TestPIControllerAgent:
    """Test PI controller agent implementing DictController protocol."""

    def test_creates_successfully(self):
        agent = PIControllerAgent()
        assert agent is not None

    def test_creates_with_custom_gains(self):
        agent = PIControllerAgent(kp_d=1.0, ki_d=100.0, kp_q=1.5, ki_q=150.0)
        assert agent.kp_d == 1.0
        assert agent.ki_d == 100.0
        assert agent.kp_q == 1.5
        assert agent.ki_q == 150.0

    def test_reset_clears_integrators(self):
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.1, "i_q_ref": 0.2}
        for _ in range(10):
            agent(state, reference)
        agent.reset()
        assert agent.integral_d == 0.0
        assert agent.integral_q == 0.0

    def test_call_returns_action_dict(self):
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.1, "i_q_ref": 0.2}
        action = agent(state, reference)
        assert isinstance(action, dict)
        assert "v_d" in action
        assert "v_q" in action

    def test_action_in_valid_range(self):
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.5, "i_q_ref": 0.5}
        action = agent(state, reference)
        assert abs(action["v_d"]) <= DEFAULT_PMSM.u_max
        assert abs(action["v_q"]) <= DEFAULT_PMSM.u_max

    def test_responds_to_error(self):
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 1.0, "i_q_ref": 1.0}
        action = agent(state, reference)
        assert action["v_d"] != 0.0 or action["v_q"] != 0.0

    def test_zero_error_minimal_action(self):
        agent = PIControllerAgent()
        agent.reset()
        state = {"i_d": 2.0, "i_q": 2.0, "omega": 0.0}
        reference = {"i_d_ref": 2.0, "i_q_ref": 2.0}
        action = agent(state, reference)
        assert abs(action["v_d"]) < 1.0
        assert abs(action["v_q"]) < 1.0

    def test_get_state_and_set_state(self):
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.1, "i_q_ref": 0.2}
        for _ in range(5):
            agent(state, reference)

        saved_state = agent.get_state()
        assert saved_state["integral_d"] != 0.0

        agent.reset()
        assert agent.integral_d == 0.0

        agent.set_state(saved_state)
        assert agent.integral_d == saved_state["integral_d"]

    def test_from_system_config(self):
        from embark.benchmark.physics import PMSMConfig

        config = PMSMConfig()
        agent = PIControllerAgent.from_system_config(config)
        assert agent is not None
        assert agent.params.u_max == config.u_max


class TestAgentInterface:
    """Test that agents follow expected interface."""

    def test_pi_agent_callable(self):
        agent = PIControllerAgent()
        assert callable(agent)

    def test_agents_have_reset(self):
        agents = [PIControllerAgent()]
        for agent in agents:
            assert hasattr(agent, "reset")

    def test_agents_have_get_state(self):
        agent = PIControllerAgent()
        assert hasattr(agent, "get_state")
        assert hasattr(agent, "set_state")


def test_pi_controller_saturation_and_anti_windup():
    agent = PIControllerAgent(anti_windup=True)
    state = {"i_d": -10.0, "i_q": -10.0, "omega": 0.0}
    reference = {"i_d_ref": 10.0, "i_q_ref": 10.0}

    for _ in range(5):
        action = agent(state, reference)
        assert "v_d" in action and "v_q" in action

    assert abs(agent.integral_d) < 1000
    assert abs(agent.integral_q) < 1000


def test_pi_controller_anti_windup_decay_is_configurable():
    agent = PIControllerAgent(
        anti_windup=True,
        anti_windup_decay=0.5,
        kp_d=100.0,
        ki_d=0.0,
        kp_q=100.0,
        ki_q=0.0,
    )
    state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
    reference = {"i_d_ref": 10.0, "i_q_ref": 10.0}

    _ = agent(state, reference)

    expected_integral = (reference["i_d_ref"] - state["i_d"]) * agent.params.Ts * 0.5
    assert agent.integral_d == pytest.approx(expected_integral)
    assert agent.integral_q == pytest.approx(expected_integral)


def test_pi_controller_rejects_invalid_anti_windup_decay():
    with pytest.raises(ValueError, match="anti_windup_decay"):
        PIControllerAgent(anti_windup_decay=0.0)

    with pytest.raises(ValueError, match="anti_windup_decay"):
        PIControllerAgent(anti_windup_decay=1.1)
