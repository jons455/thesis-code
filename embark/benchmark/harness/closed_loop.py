"""Closed-loop harness for NeuroBench-aligned benchmarks."""

from __future__ import annotations

from typing import Any

from embark.benchmark.interfaces import (
    ClosedLoopTask,
    Controller,
    MetricAccumulator,
)


class ClosedLoopHarness:
    """
    NeuroBench-style harness for closed-loop control benchmarks.

    The harness follows a unified control loop:
        state, ref = task.reset()
        while not done:
            action = controller(state, ref)
            state, ref, done = task.step(action)

    Both classical (PI) and neural (SNN) controllers use the same interface.
    Neural controllers must be wrapped with TensorControllerAdapter first.

    Example (Classical):
        controller = PIControllerAgent.from_system_config(config)
        harness = ClosedLoopHarness(task=task, controller=controller)
        results = harness.run()

    Example (Neural):
        from embark.benchmark.adapters import TensorControllerAdapter

        snn = SNNControllerAgent(...)
        controller = TensorControllerAdapter(
            controller=snn,
            state_processor=MinMaxProcessor(...),
            action_processor=LinearActionProcessor(...),
        )
        controller.configure(task.physics_engine.config, task)
        harness = ClosedLoopHarness(task=task, controller=controller)
        results = harness.run()

    """

    def __init__(
        self,
        task: ClosedLoopTask,
        controller: Controller,
        metrics: list[MetricAccumulator] | None = None,
    ) -> None:
        self.task = task
        self.controller = controller
        self.metrics = metrics or []

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """
        Run one episode of the benchmark.

        Args:
            max_steps: Override task's max_steps if provided.

        Returns:
            Dictionary containing step count and all metric results.

        """
        state, reference = self.task.reset()
        self.controller.reset()
        for metric in self.metrics:
            metric.reset()

        effective_max = max_steps or self.task.max_steps or float("inf")
        step = 0
        done = False

        while not done and step < effective_max:
            # Unified control loop - no if/else based on controller type
            action = self.controller(state, reference)

            # Get controller info (spike stats) if available
            controller_info = getattr(self.controller, "last_info", None)

            # Step the task (physics + reference update + safety check)
            next_state, next_ref, done = self.task.step(action)

            # Update all metric accumulators
            for metric in self.metrics:
                metric.update(state, reference, action, next_state, controller_info)

            state, reference = next_state, next_ref
            step += 1

        # Compute final metrics
        results: dict[str, Any] = {"steps": step}
        for metric in self.metrics:
            result = metric.compute()
            if isinstance(result, dict):
                results.update(result)
            else:
                results[metric.name] = result

        return results
