"""
Multi-scenario benchmark suite for standardised PMSM controller evaluation.

The ``BenchmarkSuite`` runs a controller through multiple operating scenarios
(different speeds, load levels, reference profiles) using the same harness
and metric infrastructure.  Users only need to supply their controller —
the suite handles task creation, metric aggregation, and result formatting.

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
    # --- Step Response Scenarios (different load levels) ---
    ScenarioDefinition(
        name="step_low_load",
        description="Step response at low torque (1A i_q) and medium speed",
        n_rpm=1000.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=1.0),
        max_steps=2000,
    ),
    ScenarioDefinition(
        name="step_mid_load",
        description="Step response at medium torque (5A i_q) and medium speed",
        n_rpm=1000.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=5.0),
        max_steps=2000,
    ),
    ScenarioDefinition(
        name="step_high_load",
        description="Step response at high torque (9A i_q) near current limit",
        n_rpm=1000.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=9.0),
        max_steps=2000,
    ),
    # --- Speed Variation Scenarios ---
    ScenarioDefinition(
        name="step_high_speed",
        description="Step response at high speed (2500 RPM) with medium load",
        n_rpm=2500.0,
        reference_generator=StepReference(i_d_ref=0.0, i_q_ref=5.0),
        max_steps=2000,
    ),
    # --- Sinusoidal Tracking (dynamic performance) ---
    ScenarioDefinition(
        name="sinusoidal_tracking",
        description="Sinusoidal i_q reference (2A amplitude, 10Hz) for tracking evaluation",
        n_rpm=1000.0,
        reference_generator=SinusoidalReference(
            i_d_ref=0.0, i_q_amp=2.0, i_q_offset=3.0, frequency_hz=10.0
        ),
        max_steps=3000,
    ),
    # --- Flux Weakening (field-weakening region) ---
    ScenarioDefinition(
        name="flux_weakening",
        description="Negative i_d reference at high speed (field-weakening operation)",
        n_rpm=2500.0,
        reference_generator=StepReference(i_d_ref=-3.0, i_q_ref=3.0),
        max_steps=2000,
    ),
]

#: Minimal scenario set for quick validation (subset of standard).
QUICK_SCENARIOS: list[ScenarioDefinition] = [
    STANDARD_SCENARIOS[0],  # step_low_load
    STANDARD_SCENARIOS[1],  # step_mid_load
    STANDARD_SCENARIOS[4],  # sinusoidal_tracking
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
