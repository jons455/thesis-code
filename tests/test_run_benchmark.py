"""Tests for run_benchmark using the new harness."""

from __future__ import annotations

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.metrics import TrackingMAE
from embark.benchmark.tasks import PMSMCurrentControlTask


def test_harness_with_pi_controller():
    """Test that the harness works with PIControllerAgent."""
    task = PMSMCurrentControlTask.from_config(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=10,
    )
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    metrics = [TrackingMAE(tracked_keys=["i_q", "i_d"])]

    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()

    assert "steps" in results
    assert results["steps"] == 10
    assert "mae_i_q" in results
    assert "mae_i_d" in results

    task.physics_engine.close()
