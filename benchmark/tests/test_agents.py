"""
Unit tests for controller agents.

Tests agent interfaces and basic functionality.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import PIControllerAgent, PIControllerTorchAgent


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
        state = np.array([0.0, 0.0, 0.1, 0.2, 100.0, 0.0])  # i_d, i_q, e_d, e_q, omega, epsilon
        action = agent(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
    
    def test_action_in_valid_range(self):
        """Agent outputs actions in normalized range."""
        agent = PIControllerAgent()
        state = np.array([0.0, 0.0, 0.5, 0.5, 100.0, 0.0])
        action = agent(state)
        # Normalized actions should be in [-1, 1]
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    
    def test_responds_to_error(self):
        """Agent produces non-zero action for non-zero error."""
        agent = PIControllerAgent()
        # State with error
        state = np.array([0.0, 0.0, 1.0, 1.0, 100.0, 0.0])
        action = agent(state)
        # Should produce some control action
        assert not np.allclose(action, [0.0, 0.0])
    
    def test_zero_error_converges(self):
        """Agent produces minimal action when error is zero."""
        agent = PIControllerAgent()
        agent.reset()
        # State with zero error
        state = np.array([2.0, 2.0, 0.0, 0.0, 100.0, 0.0])
        action = agent(state)
        # With zero error and reset integrator, action should be small
        # (may not be exactly zero due to feedforward terms)
        assert np.all(np.abs(action) < 0.5)


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
        assert action.shape == (1, 2)  # batch_size=1, action_dim=2
    
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
        assert hasattr(agent, 'reset')
        agent.reset()  # Should not raise


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
            assert hasattr(agent, 'reset')


# Run with: pytest benchmark/tests/test_agents.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

