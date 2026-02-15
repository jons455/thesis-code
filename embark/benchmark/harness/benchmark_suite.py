"""
Multi-scenario benchmark suite for standardised PMSM controller evaluation.

The ``BenchmarkSuite`` runs a controller through multiple operating scenarios
(different speeds, reference profiles, and transients) using the same harness
and metric infrastructure.  Users only need to supply their controller —
the suite handles task creation, metric aggregation, and result formatting.

The suite includes 6 optimal scenarios based on motor control benchmarking
best practices, providing minimum necessary coverage to comprehensively
evaluate controller performance across the full operating envelope.

For detailed information about scenario design, coverage, and interpretation,
see ``docs/BENCHMARK_SCENARIOS.md``.

Usage::

    from embark.benchmark.harness import BenchmarkSuite, STANDARD_SCENARIOS

    suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)
    summary = suite.run(controller=my_controller)
    suite.print_summary(summary)

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embark.benchmark.harness.closed_loop import ClosedLoopHarness
from embark.benchmark.interfaces import Controller, MetricAccumulator
from embark.benchmark.metrics.neurobench_factory import create_metrics
from embark.benchmark.tasks.pmsm_current_control import (
    PMSMCurrentControlTask,
    SafetyLimits,
)
from embark.benchmark.tasks.reference_generators import (
    ConstantReference,
    MultiStepReference,
    ReferenceGenerator,
    SinusoidalReference,
    StepReference,
)


# ============================================================================
# Scenario Definition
# ============================================================================


@dataclass
class ScenarioDefinition:
    """
    Defines a single benchmark scenario (operating point + reference profile).

    Attributes:
        name: Human-readable scenario identifier (e.g. "low_load_step").
        description: Brief explanation of what this scenario tests.
        n_rpm: Motor speed in RPM.
        reference_generator: Reference signal generator for this scenario.
        max_steps: Maximum episode length (steps).
        safety_limits: Optional custom safety limits.

    """

    name: str
    description: str
    n_rpm: float
    reference_generator: ReferenceGenerator
    max_steps: int = 2000
    safety_limits: SafetyLimits | None = None

    def create_task(self) -> PMSMCurrentControlTask:
        """Create a PMSMCurrentControlTask from this scenario definition."""
        from embark.benchmark.physics import PMSMPhysicsEngine

        physics = PMSMPhysicsEngine(n_rpm=self.n_rpm)
        return PMSMCurrentControlTask(
            physics_engine=physics,
            reference_generator=self.reference_generator,
            max_steps=self.max_steps,
            safety_limits=self.safety_limits or SafetyLimits(),
        )


# ============================================================================
# Standard Scenario Presets
# ============================================================================

STANDARD_SCENARIOS: list[ScenarioDefinition] = [
    # Scenario 1: Single-Step Low Speed (500 RPM, 0->2A i_q)
    ScenarioDefinition(
        name="step_low_speed_500rpm_2A",
        description="Step response at low speed (500 RPM, 0->2A i_q)",
        n_rpm=500.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=2.0),
        max_steps=3000,  # 0.3s at 100µs sampling
    ),
    # Scenario 2: Single-Step Mid Speed (1500 RPM, 0->2A i_q) - PRIMARY REFERENCE
    ScenarioDefinition(
        name="step_mid_speed_1500rpm_2A",
        description="Step response at nominal speed (1500 RPM, 0->2A i_q) - primary reference",
        n_rpm=1500.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=2.0),
        max_steps=3000,  # 0.3s at 100µs sampling
    ),
    # Scenario 3: Single-Step High Speed (2500 RPM, 0->2A i_q)
    ScenarioDefinition(
        name="step_high_speed_2500rpm_2A",
        description="Step response at high speed (2500 RPM, 0->2A i_q)",
        n_rpm=2500.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=2.0),
        max_steps=3000,  # 0.3s at 100µs sampling
    ),
    # Scenario 4: Multi-Step Bidirectional (1500 RPM, ±2A i_q, 4 steps)
    ScenarioDefinition(
        name="multi_step_bidirectional_1500rpm",
        description="Multi-step bidirectional tracking (1500 RPM, ±2A i_q, 4 steps)",
        n_rpm=1500.0,
        reference_generator=MultiStepReference(
            steps=[
                (0.0, 0.0, 0.0),     # Initial: 0A
                (0.1, 0.0, 2.0),     # Step 1: +2A (motoring)
                (0.35, 0.0, -2.0),   # Step 2: -2A (generating)
                (0.6, 0.0, 2.0),     # Step 3: +2A (motoring)
                (0.85, 0.0, -2.0),   # Step 4: -2A (generating)
            ]
        ),
        max_steps=10000,  # 1.0s at 100µs sampling
    ),
    # Scenario 5: Four-Quadrant Transition (1500 RPM, +2A -> -2A -> 0)
    ScenarioDefinition(
        name="four_quadrant_transition_1500rpm",
        description="Four-quadrant transition with zero-crossing (1500 RPM, +2A -> -2A -> 0)",
        n_rpm=1500.0,
        reference_generator=MultiStepReference(
            steps=[
                (0.0, 0.0, 0.0),     # Initial: 0A
                (0.1, 0.0, 2.0),     # Motoring: +2A
                (0.4, 0.0, -2.0),    # Regenerative braking: -2A
                (0.7, 0.0, 0.0),     # Zero crossing
            ]
        ),
        max_steps=9000,  # 0.9s at 100µs sampling
    ),
    # Scenario 6: Field-Weakening (2500 RPM, i_d and i_q steps)
    ScenarioDefinition(
        name="field_weakening_2500rpm",
        description="Field-weakening operation (2500 RPM, i_d=-2A, i_q=0->2A)",
        n_rpm=2500.0,
        reference_generator=MultiStepReference(
            steps=[
                (0.0, 0.0, 0.0),     # Initial: no current
                (0.1, -2.0, 0.0),    # Step i_d to -2A (field weakening)
                (0.35, -2.0, 2.0),   # Step i_q to 2A (with active i_d)
            ]
        ),
        max_steps=6000,  # 0.6s at 100µs sampling
    ),
]

#: Minimal scenario set for quick validation (primary reference + one edge case).
QUICK_SCENARIOS: list[ScenarioDefinition] = [
    STANDARD_SCENARIOS[1],  # step_mid_speed_1500rpm_2A (primary reference)
    STANDARD_SCENARIOS[3],  # multi_step_bidirectional_1500rpm
]


# ============================================================================
# Default Metric Factory
# ============================================================================


def default_metric_factory(
    controller: Controller | None = None,
) -> list[MetricAccumulator]:
    """Create metrics for a scenario run, with optional NeuroBench adapters."""
    return create_metrics(controller)


# ============================================================================
# Benchmark Suite
# ============================================================================


@dataclass
class ScenarioResult:
    """Results from a single scenario execution."""

    scenario_name: str
    description: str
    metrics: dict[str, Any]
    safety_terminated: bool = False
    violation_reason: str | None = None


@dataclass
class BenchmarkSummary:
    """Aggregated results across all scenarios."""

    controller_name: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    @property
    def mean_mae_iq(self) -> float:
        """Average MAE i_q across all scenarios."""
        values = [r.metrics.get("mae_i_q", 0.0) for r in self.scenario_results]
        return sum(values) / max(len(values), 1)

    @property
    def worst_max_error_iq(self) -> float:
        """Worst-case max error across all scenarios."""
        values = [r.metrics.get("max_error_i_q", 0.0) for r in self.scenario_results]
        return max(values) if values else 0.0

    @property
    def num_safety_violations(self) -> int:
        """Number of scenarios terminated by safety limits."""
        return sum(1 for r in self.scenario_results if r.safety_terminated)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON export."""
        return {
            "controller_name": self.controller_name,
            "aggregate": {
                "mean_mae_iq": self.mean_mae_iq,
                "worst_max_error_iq": self.worst_max_error_iq,
                "num_safety_violations": self.num_safety_violations,
                "num_scenarios": len(self.scenario_results),
            },
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "description": r.description,
                    "metrics": r.metrics,
                    "safety_terminated": r.safety_terminated,
                    "violation_reason": r.violation_reason,
                }
                for r in self.scenario_results
            ],
        }


class BenchmarkSuite:
    """
    Multi-scenario benchmark runner.

    Runs a controller through a standardised set of operating scenarios
    and aggregates results into a ``BenchmarkSummary``.

    The user just provides a controller (already configured with the
    appropriate adapter/processors); the suite handles everything else.

    Args:
        scenarios: List of scenario definitions to run.  Defaults to
            ``STANDARD_SCENARIOS``.
        metric_factory: Callable returning a fresh list of
            ``MetricAccumulator`` instances.  Called once per scenario.
        verbose: Print progress during execution.

    Example::

        from embark.benchmark.harness import BenchmarkSuite

        suite = BenchmarkSuite()
        summary = suite.run(controller=my_snn_controller, name="MySNN")
        suite.print_summary(summary)
        suite.save_results(summary, "results/benchmark.json")

    """

    def __init__(
        self,
        scenarios: list[ScenarioDefinition] | None = None,
        metric_factory: Any = None,
        verbose: bool = True,
    ) -> None:
        self.scenarios = scenarios or STANDARD_SCENARIOS
        self.metric_factory = metric_factory or default_metric_factory
        self.verbose = verbose

    def run(
        self,
        controller: Controller,
        name: str = "Controller",
    ) -> BenchmarkSummary:
        """
        Run the controller through all scenarios.

        Args:
            controller: A configured controller (DictController or
                TensorControllerAdapter).
            name: Display name for this controller in results.

        Returns:
            BenchmarkSummary with per-scenario and aggregate results.

        """
        summary = BenchmarkSummary(controller_name=name)

        for i, scenario in enumerate(self.scenarios, 1):
            if self.verbose:
                print(
                    f"  [{i}/{len(self.scenarios)}] {scenario.name}: "
                    f"{scenario.description}"
                )

            # Create fresh task and metrics for each scenario
            task = scenario.create_task()
            try:
                metrics = self.metric_factory(controller)
            except TypeError:
                # Backward-compatibility for older metric_factory signatures.
                metrics = self.metric_factory()

            # Re-configure controller with this scenario's task/physics
            # (needed for processors that depend on physics config)
            if hasattr(controller, "configure"):
                controller.configure(task.physics_engine.config, task)

            # Build and run harness
            harness = ClosedLoopHarness(
                task=task,
                controller=controller,
                metrics=metrics,
            )
            results = harness.run()

            # Record result
            scenario_result = ScenarioResult(
                scenario_name=scenario.name,
                description=scenario.description,
                metrics=results,
                safety_terminated=getattr(task, "terminated_by_safety", False),
                violation_reason=getattr(task, "last_violation_reason", None),
            )
            summary.scenario_results.append(scenario_result)

            if self.verbose:
                mae_iq = results.get("mae_i_q", 0.0)
                max_err = results.get("max_error_i_q", 0.0)
                status = "SAFETY VIOLATION" if task.terminated_by_safety else "OK"
                print(
                    f"           MAE={mae_iq:.4f}A  MaxErr={max_err:.4f}A  [{status}]"
                )

        return summary

    @staticmethod
    def print_summary(summary: BenchmarkSummary) -> None:
        """Print a formatted comparison table."""
        print()
        print("=" * 80)
        print(f"  Benchmark Summary: {summary.controller_name}")
        print(
            f"  {len(summary.scenario_results)} scenarios completed, "
            f"{summary.num_safety_violations} safety violations"
        )
        print("=" * 80)
        print()

        # Per-scenario table
        header = (
            f"{'Scenario':<22} {'MAE_iq':>9} {'MaxErr_iq':>10} "
            f"{'Settle[s]':>10} {'Status':>8}"
        )
        print(header)
        print("-" * 80)

        for r in summary.scenario_results:
            m = r.metrics
            status = "FAIL" if r.safety_terminated else "OK"
            settle = m.get("settling_time", float("inf"))
            settle_str = f"{settle:.4f}" if settle < float("inf") else "N/A"
            print(
                f"{r.scenario_name:<22} "
                f"{m.get('mae_i_q', 0.0):>9.4f} "
                f"{m.get('max_error_i_q', 0.0):>10.4f} "
                f"{settle_str:>10} "
                f"{status:>8}"
            )

        # Aggregate
        print("-" * 80)
        print(
            f"{'AGGREGATE':<22} "
            f"{summary.mean_mae_iq:>9.4f} "
            f"{summary.worst_max_error_iq:>10.4f} "
            f"{'':>10} {'':>8}"
        )
        print()

    @staticmethod
    def save_results(summary: BenchmarkSummary, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
