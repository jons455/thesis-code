"""Integration tests for the benchmark API."""

from __future__ import annotations

import pytest

pytest.importorskip("gym_electric_motor")

from embark.benchmark import (  # noqa: E402
    ClosedLoopHarness,
    PIControllerAgent,
    PMSMCurrentControlTask,
    TrackingMAE,
)


def test_benchmark_with_pi_controller():
    """Test running a benchmark with PI controller using new harness."""
    task = PMSMCurrentControlTask.from_config(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=100,
    )
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    metrics = [TrackingMAE(tracked_keys=["i_q", "i_d"])]

    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()

    assert "steps" in results
    assert results["steps"] == 100
    assert "mae_i_q" in results
    assert results["mae_i_q"] >= 0

    task.physics_engine.close()


def test_benchmark_api_smoke_test():
    """Smoke test that the benchmark API works end-to-end."""
    task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0, max_steps=50)
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)

    harness = ClosedLoopHarness(task=task, controller=controller)
    results = harness.run()

    assert results["steps"] == 50
    task.physics_engine.close()
