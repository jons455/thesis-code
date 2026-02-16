"""
Unit tests for controller agents.

Tests agent interfaces and basic functionality with new DictController/TensorController
protocols.

"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from embark.benchmark import agents as agents_mod
from embark.benchmark.agents import (
    PIControllerAgent,
    SNNControllerAgent,
    SNNControllerTorchAgent,
)
from embark.utils.config import DEFAULT_PMSM


class TestPIControllerAgent:
    """Test PI controller agent implementing DictController protocol."""

    def test_creates_successfully(self):
        """Agent can be created with default parameters."""
        agent = PIControllerAgent()
        assert agent is not None

    def test_creates_with_custom_gains(self):
        """Agent accepts custom PI gains."""
        agent = PIControllerAgent(kp_d=1.0, ki_d=100.0, kp_q=1.5, ki_q=150.0)
        assert agent.kp_d == 1.0
        assert agent.ki_d == 100.0
        assert agent.kp_q == 1.5
        assert agent.ki_q == 150.0

    def test_reset_clears_integrators(self):
        """Reset clears integrator states."""
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.1, "i_q_ref": 0.2}
        for _ in range(10):
            agent(state, reference)
        agent.reset()
        assert agent.integral_d == 0.0
        assert agent.integral_q == 0.0

    def test_call_returns_action_dict(self):
        """Calling agent returns action dict."""
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.1, "i_q_ref": 0.2}
        action = agent(state, reference)
        assert isinstance(action, dict)
        assert "v_d" in action
        assert "v_q" in action

    def test_action_in_valid_range(self):
        """Agent outputs actions in physical range."""
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 0.5, "i_q_ref": 0.5}
        action = agent(state, reference)
        assert abs(action["v_d"]) <= DEFAULT_PMSM.u_max
        assert abs(action["v_q"]) <= DEFAULT_PMSM.u_max

    def test_responds_to_error(self):
        """Agent produces non-zero action for non-zero error."""
        agent = PIControllerAgent()
        state = {"i_d": 0.0, "i_q": 0.0, "omega": 0.0}
        reference = {"i_d_ref": 1.0, "i_q_ref": 1.0}
        action = agent(state, reference)
        assert action["v_d"] != 0.0 or action["v_q"] != 0.0

    def test_zero_error_minimal_action(self):
        """Agent produces minimal action when error is zero."""
        agent = PIControllerAgent()
        agent.reset()
        state = {"i_d": 2.0, "i_q": 2.0, "omega": 0.0}
        reference = {"i_d_ref": 2.0, "i_q_ref": 2.0}
        action = agent(state, reference)
        assert abs(action["v_d"]) < 1.0
        assert abs(action["v_q"]) < 1.0

    def test_get_state_and_set_state(self):
        """Agent can serialize and restore state."""
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
        """Agent can be created from system config."""
        from embark.benchmark.physics import PMSMConfig

        config = PMSMConfig()
        agent = PIControllerAgent.from_system_config(config)
        assert agent is not None
        assert agent.params.u_max == config.u_max


class TestAgentInterface:
    """Test that agents follow expected interface."""

    def test_pi_agent_callable(self):
        """PI agent is callable."""
        agent = PIControllerAgent()
        assert callable(agent)

    def test_agents_have_reset(self):
        """All agents have reset method."""
        agents = [PIControllerAgent()]
        for agent in agents:
            assert hasattr(agent, "reset")

    def test_agents_have_get_state(self):
        """Agents have get_state/set_state methods."""
        agent = PIControllerAgent()
        assert hasattr(agent, "get_state")
        assert hasattr(agent, "set_state")


def _get_trained_model_checkpoint() -> Path:
    repo_root = Path(__file__).parent.parent.parent
    # Look for both new "speed_final" naming and original model types
    preferred = [
        "linear_speed_final",
        "population_speed_final",
        "membrane",
        "delta",
        "population",
        "recurrent",
        "ttfs",
    ]
    for model_name in preferred:
        candidate = (
            repo_root / "evaluation" / "trained_models" / model_name / "best_model.pt"
        )
        if candidate.exists():
            return candidate

    pytest.skip("No compatible SNN checkpoint available in evaluation/trained_models/")


class TestSNNControllerAgent:
    """Test SNN controller agent implementing TensorController protocol."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to test checkpoint."""
        return _get_trained_model_checkpoint()

    def test_checkpoint_exists(self, checkpoint_path):
        """Verify checkpoint file exists for testing."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available - run training first")

    def test_creates_successfully(self, checkpoint_path):
        """Agent can be created with checkpoint."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        assert agent is not None

    def test_has_model(self, checkpoint_path):
        """Agent has loaded model."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        assert agent.model is not None

    def test_reset_clears_state(self, checkpoint_path):
        """Reset clears neuron membrane states."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        # Provide 5 features [i_d, i_q, e_d, e_q, n]
        obs = torch.randn(1, 5)
        for _ in range(10):
            agent.forward(obs)

        assert agent._snn_state is not None

        agent.reset()
        assert agent._snn_state is None

    def test_forward_returns_tensor(self, checkpoint_path):
        """Forward returns action tensor."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        obs = torch.randn(1, 5)
        action = agent.forward(obs)

        assert isinstance(action, torch.Tensor)
        assert action.shape[-1] == 2

    def test_action_in_valid_range(self, checkpoint_path):
        """Agent outputs actions in [-1, 1] range."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        obs = torch.randn(1, 5) * 0.5
        action = agent.forward(obs)

        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)

    def test_state_persists_across_calls(self, checkpoint_path):
        """Membrane state persists across timesteps."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        agent.reset()

        obs = torch.randn(1, 5)

        agent.forward(obs)
        state_after_1 = agent._snn_state

        agent.forward(obs)
        state_after_2 = agent._snn_state

        assert state_after_1 is not None
        assert state_after_2 is not None

    def test_no_nan_output(self, checkpoint_path):
        """Agent should not produce NaN outputs."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        agent.reset()

        for _ in range(100):
            obs = torch.randn(1, 5) * 0.5
            action = agent.forward(obs)
            assert not torch.isnan(action).any(), "NaN detected in output"

    def test_get_spike_statistics(self, checkpoint_path):
        """Agent returns spike statistics."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path), track_spikes=True)
        agent.reset()

        for _ in range(10):
            obs = torch.randn(1, 5) * 0.5
            agent.forward(obs)

        stats = agent.get_spike_statistics()
        assert isinstance(stats, dict)
        assert "total_spikes" in stats


class TestSNNControllerTorchAgent:
    """Test PyTorch-wrapped SNN controller."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to test checkpoint."""
        return _get_trained_model_checkpoint()

    def test_creates_successfully(self, checkpoint_path):
        """Agent can be created."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerTorchAgent(str(checkpoint_path))
        assert agent is not None

    def test_is_torch_module(self, checkpoint_path):
        """Agent is a PyTorch module."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerTorchAgent(str(checkpoint_path))
        assert isinstance(agent, torch.nn.Module)


class _FakeSNNModel:
    def __init__(self):
        self.layers = [SimpleNamespace(out_features=3), SimpleNamespace(out_features=2)]
        self.fc_out = SimpleNamespace(out_features=2)
        self._call_count = 0

    def eval(self):
        return self

    def get_network_stats(self):
        return {"hidden_size": 3, "num_layers": 2}

    def count_parameters(self):
        return 42

    def __call__(self, observation, state, return_spikes=True):
        self._call_count += 1
        out = torch.full((observation.shape[0], 2), 0.25, dtype=observation.dtype)
        new_state = ("state", self._call_count)
        spike_info = None
        if return_spikes:
            spike_info = {
                "total_spikes": 4,
                "layer_sparsities": [0.5, 0.75],
                "spike_counts": [2, 2],
            }
        return out, new_state, spike_info


class RecurrentSNNControllerFake(_FakeSNNModel):
    pass


class SNNWithPopOut(_FakeSNNModel):
    def __init__(self):
        super().__init__()
        del self.fc_out
        self.pop_out = SimpleNamespace(fc=SimpleNamespace(out_features=7))


class SNNWithTtfsOut(_FakeSNNModel):
    def __init__(self):
        super().__init__()
        del self.fc_out
        self.ttfs_out = SimpleNamespace(fc=SimpleNamespace(out_features=8))


class SNNNoOutputHead(_FakeSNNModel):
    def __init__(self):
        super().__init__()
        del self.fc_out


def _install_fake_load_snn_model(monkeypatch, model_obj):
    eval_mod = ModuleType("evaluation")
    pytorch_snn_mod = ModuleType("evaluation.pytorch_snn")
    models_mod = ModuleType("evaluation.pytorch_snn.models")
    models_mod.load_snn_model = lambda checkpoint_path, device="cpu": model_obj

    monkeypatch.setitem(sys.modules, "evaluation", eval_mod)
    monkeypatch.setitem(sys.modules, "evaluation.pytorch_snn", pytorch_snn_mod)
    monkeypatch.setitem(sys.modules, "evaluation.pytorch_snn.models", models_mod)


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


def test_snn_controller_agent_forward_and_statistics(monkeypatch):
    fake_model = _FakeSNNModel()
    _install_fake_load_snn_model(monkeypatch, fake_model)

    agent = SNNControllerAgent(
        checkpoint_path="dummy.pt",
        track_spikes=True,
        num_inference_steps=2,
    )

    obs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    out = agent.forward(obs)
    assert out.shape == (1, 2)
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)
    assert agent.last_info is not None
    assert agent.last_info["syops"] >= 0

    info = agent.get_info()
    assert info["parameters"] == 42
    assert info["hidden_size"] == 3

    spike_stats = agent.get_spike_statistics()
    assert "total_spikes" in spike_stats
    assert spike_stats["num_control_steps"] >= 1

    state = agent.get_state()
    agent.set_state(state)
    agent.reset()
    assert "error" in agent.get_spike_statistics()


def test_snn_controller_agent_recurrent_fanout_branch(monkeypatch):
    fake_model = RecurrentSNNControllerFake()
    _install_fake_load_snn_model(monkeypatch, fake_model)

    agent = SNNControllerAgent(checkpoint_path="dummy.pt", track_spikes=False)
    assert any(v > 0 for v in agent._recurrent_fanouts)

    out = agent.forward(torch.randn(1, 5))
    assert out.shape == (1, 2)


def test_snn_controller_torch_wrapper_with_fake_model(monkeypatch):
    fake_model = _FakeSNNModel()
    _install_fake_load_snn_model(monkeypatch, fake_model)

    wrapper = SNNControllerTorchAgent(checkpoint_path="dummy.pt")
    out = wrapper(torch.randn(1, 5))
    assert out.shape == (1, 2)
    wrapper.reset()
    stats = wrapper.get_spike_statistics()
    assert isinstance(stats, dict)


def test_snn_controller_inserts_project_root_into_sys_path(monkeypatch):
    fake_model = _FakeSNNModel()
    _install_fake_load_snn_model(monkeypatch, fake_model)

    expected_root = str(Path(agents_mod.__file__).resolve().parents[3])
    while expected_root in sys.path:
        sys.path.remove(expected_root)

    _ = SNNControllerAgent(checkpoint_path="dummy.pt")
    assert expected_root in sys.path


@pytest.mark.parametrize(
    "model_cls,expected_last_fanout",
    [
        (SNNWithPopOut, 7),
        (SNNWithTtfsOut, 8),
        (SNNNoOutputHead, 0),
    ],
)
def test_snn_controller_output_head_fanout_fallbacks(
    monkeypatch, model_cls, expected_last_fanout
):
    fake_model = model_cls()
    _install_fake_load_snn_model(monkeypatch, fake_model)

    agent = SNNControllerAgent(checkpoint_path="dummy.pt", track_spikes=False)
    assert agent._layer_fanouts[-1] == expected_last_fanout
