"""
Integration tests for the full benchmark pipeline.

Tests that all components work together correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "benchmark"))
sys.path.insert(0, str(project_root / "metrics"))


class TestEnvironmentAgentIntegration:
    """Test PMSMEnv + Agent integration."""
    
    def test_pi_agent_with_env(self, pmsm_env, pi_agent):
        """PI agent can control the environment."""
        state, _ = pmsm_env.reset()
        pi_agent.reset()
        
        for _ in range(50):
            action = pi_agent(state)
            state, reward, done, truncated, info = pmsm_env.step(action)
            if done or truncated:
                break
        
        # Should complete some steps
        assert pmsm_env.current_step > 0
    
    def test_episode_completes(self, pmsm_env, pi_agent):
        """Full episode runs to completion."""
        state, _ = pmsm_env.reset()
        pi_agent.reset()
        
        steps = 0
        max_steps = pmsm_env.max_steps
        
        while steps < max_steps:
            action = pi_agent(state)
            state, reward, done, truncated, info = pmsm_env.step(action)
            steps += 1
            if done or truncated:
                break
        
        # Should reach max steps (truncated)
        assert steps == max_steps
    
    def test_episode_data_collected(self, pmsm_env, pi_agent):
        """Episode data is collected during simulation."""
        state, _ = pmsm_env.reset()
        pi_agent.reset()
        
        for _ in range(50):
            action = pi_agent(state)
            state, _, done, truncated, _ = pmsm_env.step(action)
            if done or truncated:
                break
        
        data = pmsm_env.get_episode_data()
        assert len(data) == 50
        assert 'i_d' in data[0]
        assert 'i_q' in data[0]


class TestTrackingPerformance:
    """Test that controllers achieve acceptable tracking."""
    
    def test_pi_tracks_reference(self):
        """PI controller tracks reference within tolerance."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        env = PMSMEnv(
            n_rpm=1000,
            i_d_ref=0.0,
            i_q_ref=2.0,
            max_steps=500,
        )
        agent = PIControllerAgent()
        
        state, _ = env.reset()
        agent.reset()
        
        for _ in range(500):
            action = agent(state)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        
        data = env.get_episode_data()
        env.close()
        
        # Check final tracking error
        final_e_d = abs(data[-1]['e_d'])
        final_e_q = abs(data[-1]['e_q'])
        
        assert final_e_d < 0.1, f"i_d error too high: {final_e_d}"
        assert final_e_q < 0.1, f"i_q error too high: {final_e_q}"
    
    def test_multiple_operating_points(self):
        """PI controller works across operating points."""
        from pmsm_env import PMSMEnv
        from agents import PIControllerAgent
        
        operating_points = [
            (1000, 0.0, 1.0),
            (1000, 0.0, 3.0),
            (1500, 0.0, 2.0),
        ]
        
        agent = PIControllerAgent()
        
        for n_rpm, i_d_ref, i_q_ref in operating_points:
            env = PMSMEnv(
                n_rpm=n_rpm,
                i_d_ref=i_d_ref,
                i_q_ref=i_q_ref,
                max_steps=300,
            )
            
            state, _ = env.reset()
            agent.reset()
            
            for _ in range(300):
                action = agent(state)
                state, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            
            data = env.get_episode_data()
            env.close()
            
            # Check tracking
            final_error = np.sqrt(data[-1]['e_d']**2 + data[-1]['e_q']**2)
            assert final_error < 0.2, f"Failed at {n_rpm} RPM, i_q_ref={i_q_ref}A"


class TestMetricsIntegration:
    """Test metrics computation with environment data."""
    
    def test_compute_metrics_from_episode(self, sample_trajectory_data):
        """Metrics can be computed from episode data."""
        from benchmark_metrics import run_benchmark
        
        result = run_benchmark(
            sample_trajectory_data,
            controller_name="Test",
            operating_point="Test OP",
        )
        
        assert result is not None
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'dynamics')
    
    def test_metrics_reasonable_values(self, sample_trajectory_data):
        """Metric values are in reasonable ranges."""
        from benchmark_metrics import run_benchmark
        
        result = run_benchmark(
            sample_trajectory_data,
            controller_name="Test",
            operating_point="Test OP",
        )
        
        # MAE should be positive
        assert result.accuracy.MAE_iq >= 0
        
        # Rise time should be positive and reasonable
        assert result.dynamics.rise_time_iq > 0
        assert result.dynamics.rise_time_iq < 1.0  # Less than 1 second


class TestNeuroBenchCompatibility:
    """Test NeuroBench framework compatibility."""
    
    def test_env_gymnasium_compatible(self):
        """Environment follows Gymnasium interface."""
        from pmsm_env import PMSMEnv
        import gymnasium as gym
        
        env = PMSMEnv()
        
        # Check interface
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'close')
        
        env.close()
    
    def test_torch_agent_interface(self):
        """Torch agent has required interface."""
        import torch
        from agents import PIControllerTorchAgent
        
        agent = PIControllerTorchAgent()
        
        # Check interface
        assert isinstance(agent, torch.nn.Module)
        assert hasattr(agent, 'forward')
        assert hasattr(agent, 'reset')
        
        # Check callable
        state = torch.randn(1, 6)
        action = agent(state)
        assert action.shape == (1, 2)


# Run with: pytest tests/test_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

