"""
Integration tests for the full benchmark pipeline.

Tests that all components work together correctly.

Rewritten 2026-02-07 to use the current architecture:
    PMSMCurrentControlTask + PIControllerAgent + ClosedLoopHarness
(The original PMSMEnv Gymnasium wrapper was removed in the 2026-02-03 refactoring.)

"""

import numpy as np
import pytest

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.harness.closed_loop import ClosedLoopHarness
from embark.benchmark.metrics.accumulators import TrackingMAE
from embark.benchmark.tasks.pmsm_current_control import PMSMCurrentControlTask


class TestEnvironmentAgentIntegration:
    """Test Task + Agent integration."""

    @pytest.fixture
    def task(self):
        return PMSMCurrentControlTask.from_config(
            n_rpm=1000.0, i_d_ref=0.0, i_q_ref=2.0, max_steps=500
        )

    @pytest.fixture
    def pi_agent(self, task):
        return PIControllerAgent.from_system_config(task.physics_engine.config)

    def test_pi_agent_with_task(self, task, pi_agent):
        """PI agent can control the task for 50 steps."""
        state, reference = task.reset()
        pi_agent.reset()

        steps = 0
        for _ in range(50):
            action = pi_agent(state, reference)
            state, reference, done = task.step(action)
            steps += 1
            if done:
                break

        assert steps > 0

    def test_episode_completes(self, task, pi_agent):
        """Full episode runs to max_steps."""
        state, reference = task.reset()
        pi_agent.reset()

        steps = 0
        while steps < task.max_steps:
            action = pi_agent(state, reference)
            state, reference, done = task.step(action)
            steps += 1
            if done:
                break

        assert steps == task.max_steps

    def test_episode_data_collected(self, task, pi_agent):
        """Episode data is collected through the harness."""
        metrics = [TrackingMAE(tracked_keys=["i_q", "i_d"])]
        harness = ClosedLoopHarness(task=task, controller=pi_agent, metrics=metrics)
        results = harness.run()

        assert "mae_i_q" in results
        assert "mae_i_d" in results
        assert results["mae_i_q"] >= 0
        assert results["mae_i_d"] >= 0


class TestTrackingPerformance:
    """Test that controllers achieve acceptable tracking."""

    def test_pi_tracks_reference(self):
        """PI controller tracks reference within tolerance."""
        task = PMSMCurrentControlTask.from_config(
            n_rpm=1000.0, i_d_ref=0.0, i_q_ref=2.0, max_steps=500
        )
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)

        state, reference = task.reset()
        agent.reset()

        for _ in range(500):
            action = agent(state, reference)
            state, reference, done = task.step(action)
            if done:
                break

        # Check final tracking error
        final_e_d = abs(reference["i_d_ref"] - state["i_d"])
        final_e_q = abs(reference["i_q_ref"] - state["i_q"])

        assert final_e_d < 0.6, f"i_d error too high: {final_e_d}"
        assert final_e_q < 0.6, f"i_q error too high: {final_e_q}"

    def test_multiple_operating_points(self):
        """PI controller works across operating points."""
        operating_points = [
            (1000, 0.0, 1.0),
            (1000, 0.0, 3.0),
            (1500, 0.0, 2.0),
        ]

        for n_rpm, i_d_ref, i_q_ref in operating_points:
            task = PMSMCurrentControlTask.from_config(
                n_rpm=n_rpm, i_d_ref=i_d_ref, i_q_ref=i_q_ref, max_steps=300
            )
            agent = PIControllerAgent.from_system_config(task.physics_engine.config)

            state, reference = task.reset()
            agent.reset()

            for _ in range(300):
                action = agent(state, reference)
                state, reference, done = task.step(action)
                if done:
                    break

            final_error = np.sqrt(
                (reference["i_d_ref"] - state["i_d"]) ** 2
                + (reference["i_q_ref"] - state["i_q"]) ** 2
            )
            assert (
                final_error < 2.0
            ), f"Failed at {n_rpm} RPM, i_q_ref={i_q_ref}A: error={final_error:.4f}"


class TestMetricsIntegration:
    """Test metrics computation with the harness."""

    def test_compute_metrics_from_episode(self):
        """Metrics can be computed from a harness run."""
        task = PMSMCurrentControlTask.from_config(
            n_rpm=1000.0, i_d_ref=0.0, i_q_ref=2.0, max_steps=200
        )
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)
        metrics = [TrackingMAE(tracked_keys=["i_q", "i_d"])]
        harness = ClosedLoopHarness(task=task, controller=agent, metrics=metrics)

        results = harness.run()

        assert results is not None
        assert "mae_i_q" in results
        assert "mae_i_d" in results

    def test_metrics_reasonable_values(self):
        """Metric values are in reasonable ranges."""
        task = PMSMCurrentControlTask.from_config(
            n_rpm=1000.0, i_d_ref=0.0, i_q_ref=2.0, max_steps=200
        )
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)
        metrics = [TrackingMAE(tracked_keys=["i_q", "i_d"])]
        harness = ClosedLoopHarness(task=task, controller=agent, metrics=metrics)

        results = harness.run()

        # MAE should be positive and reasonable for PI controller
        assert results["mae_i_q"] >= 0
        assert results["mae_i_q"] < 1.0  # PI should track well
        assert results["mae_i_d"] >= 0


class TestNeuroBenchCompatibility:
    """Test NeuroBench framework compatibility."""

    def test_task_interface(self):
        """Task follows the ClosedLoopTask interface."""
        task = PMSMCurrentControlTask.from_config()

        assert hasattr(task, "reset")
        assert hasattr(task, "step")
        assert hasattr(task, "physics_engine")

        state, reference = task.reset()
        assert "i_d" in state
        assert "i_q" in state
        assert "i_d_ref" in reference
        assert "i_q_ref" in reference

    def test_agent_interface(self):
        """PI agent has required DictController interface."""
        task = PMSMCurrentControlTask.from_config()
        agent = PIControllerAgent.from_system_config(task.physics_engine.config)

        assert hasattr(agent, "reset")
        assert callable(agent)

        state, reference = task.reset()
        agent.reset()
        action = agent(state, reference)

        assert "v_d" in action
        assert "v_q" in action


# Run with: pytest tests/test_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
