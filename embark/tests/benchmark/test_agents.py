"""
Unit tests for controller agents.

Tests agent interfaces and basic functionality.
"""

import numpy as np
import pytest
import torch

from embark.benchmark.agents import (
    PIControllerAgent,
    PIControllerTorchAgent,
    SNNControllerAgent,
    SNNControllerTorchAgent,
)
from embark.utils.config import DEFAULT_PMSM
from embark.utils.paths import MODELS_CHECKPOINTS_DIR


class TestPIControllerAgent:
    """Test PI controller agent."""

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
        # Simulate some calls to build up integrator
        state = np.array([0.0, 0.0, 0.1, 0.2, 100.0, 0.0])
        for _ in range(10):
            agent(state)
        agent.reset()
        assert agent.integral_d == 0.0
        assert agent.integral_q == 0.0

    def test_call_returns_action(self):
        """Calling agent returns action array."""
        agent = PIControllerAgent()
        state = np.array([0.0, 0.0, 0.1, 0.2, 100.0, 0.0])
        action = agent(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_action_in_valid_range(self):
        """Agent outputs actions in physical range."""
        agent = PIControllerAgent()
        state = np.array([0.0, 0.0, 0.5, 0.5, 100.0, 0.0])
        action = agent(state)
        assert np.all(action >= -DEFAULT_PMSM.u_max)
        assert np.all(action <= DEFAULT_PMSM.u_max)

    def test_responds_to_error(self):
        """Agent produces non-zero action for non-zero error."""
        agent = PIControllerAgent()
        state = np.array([0.0, 0.0, 1.0, 1.0, 100.0, 0.0])
        action = agent(state)
        assert not np.allclose(action, [0.0, 0.0])

    def test_zero_error_converges(self):
        """Agent produces minimal action when error is zero."""
        agent = PIControllerAgent()
        agent.reset()
        state = np.array([2.0, 2.0, 0.0, 0.0, 100.0, 0.0])
        action = agent(state)
        assert np.all(np.abs(action) < 1.0)


class TestPIControllerTorchAgent:
    """Test PyTorch-wrapped PI controller for NeuroBench compatibility."""

    def test_creates_successfully(self):
        """Agent can be created."""
        agent = PIControllerTorchAgent()
        assert agent is not None

    def test_is_torch_module(self):
        """Agent is a PyTorch module."""
        agent = PIControllerTorchAgent()
        assert isinstance(agent, torch.nn.Module)

    def test_forward_accepts_tensor(self):
        """Forward accepts PyTorch tensor input."""
        agent = PIControllerTorchAgent()
        state = torch.tensor([[0.0, 0.0, 0.1, 0.2, 100.0, 0.0]])
        action = agent(state)
        assert isinstance(action, torch.Tensor)

    def test_forward_returns_correct_shape(self):
        """Forward returns action with correct shape."""
        agent = PIControllerTorchAgent()
        state = torch.tensor([[0.0, 0.0, 0.1, 0.2, 100.0, 0.0]])
        action = agent(state)
        assert action.shape == (1, 2)

    def test_forward_batch(self):
        """Forward handles batch inputs."""
        agent = PIControllerTorchAgent()
        batch_size = 4
        state = torch.randn(batch_size, 6)
        action = agent(state)
        assert action.shape == (batch_size, 2)

    def test_has_reset_method(self):
        """Agent has reset method."""
        agent = PIControllerTorchAgent()
        assert hasattr(agent, "reset")
        agent.reset()


class TestAgentInterface:
    """Test that agents follow expected interface."""

    def test_pi_agent_callable(self):
        """PI agent is callable."""
        agent = PIControllerAgent()
        assert callable(agent)

    def test_torch_agent_callable(self):
        """Torch agent is callable."""
        agent = PIControllerTorchAgent()
        assert callable(agent)

    def test_agents_have_reset(self):
        """All agents have reset method."""
        agents = [PIControllerAgent(), PIControllerTorchAgent()]
        for agent in agents:
            assert hasattr(agent, "reset")


class TestSNNControllerAgent:
    """Test SNN controller agent."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to test checkpoint."""
        return MODELS_CHECKPOINTS_DIR / "best_model.pt"

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

        state = np.array([0.0, 0.0, 0.1, 0.2])
        for _ in range(10):
            agent(state)

        assert agent._snn_state is not None

        agent.reset()
        assert agent._snn_state is None

    def test_call_returns_action(self, checkpoint_path):
        """Calling agent returns action array."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        state = np.array([0.0, 0.0, 0.1, 0.2])
        action = agent(state)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)

    def test_action_in_valid_range(self, checkpoint_path):
        """Agent outputs actions in normalized range."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        state = np.array([0.0, 0.0, 0.5, 0.5])
        action = agent(state)

        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_state_persists_across_calls(self, checkpoint_path):
        """Membrane state persists across timesteps."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        agent.reset()

        state = np.array([0.0, 0.0, 0.1, 0.2])

        agent(state)
        state_after_1 = agent._snn_state

        agent(state)
        state_after_2 = agent._snn_state

        assert state_after_1 is not None
        assert state_after_2 is not None

    def test_get_sparsity(self, checkpoint_path):
        """Agent reports activation sparsity."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        state = np.array([0.0, 0.0, 0.1, 0.2])

        sparsity = agent.get_sparsity(state)

        assert isinstance(sparsity, dict)
        assert len(sparsity) > 0
        for val in sparsity.values():
            assert 0.0 <= val <= 1.0

    def test_no_nan_output(self, checkpoint_path):
        """Agent should not produce NaN outputs."""
        if not checkpoint_path.exists():
            pytest.skip("No SNN checkpoint available")

        agent = SNNControllerAgent(str(checkpoint_path))
        agent.reset()

        for _ in range(100):
            state = np.random.randn(4) * 0.5
            action = agent(state)
            assert not np.isnan(action).any(), "NaN detected in output"


class TestSNNControllerTorchAgent:
    """Test PyTorch-wrapped SNN controller."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to test checkpoint."""
        return MODELS_CHECKPOINTS_DIR / "best_model.pt"

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
