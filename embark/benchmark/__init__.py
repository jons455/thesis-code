"""
Closed-loop PMSM Current Control Benchmark.

This module provides a modular closed-loop benchmark framework adapted
from NeuroBench architecture patterns for neuromorphic motor control.

Core Components:
----------------
- ``ClosedLoopHarness``: Main benchmark orchestrator (single scenario)
- ``BenchmarkSuite``: Multi-scenario benchmark runner
- ``PMSMCurrentControlTask``: PMSM current control task with physics engine
- ``PMSMPhysicsEngine``: Pure physics dynamics (wraps GEM)
- ``PIControllerAgent``: Classical PI controller baseline (DictController)
- ``TensorControllerAdapter``: Wraps neural controllers + processors into
  unified Controller interface

Architecture:
-------------
The harness follows a unified control loop without if/else branching::

    state, ref = task.reset()
    while not done:
        action = controller(state, ref)  # Unified interface
        state, ref, done = task.step(action)

Classical controllers implement ``Controller`` directly.
Neural controllers must be wrapped with ``TensorControllerAdapter``.

Multi-Scenario Usage::

    from embark.benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    summary = suite.run(controller=my_controller, name="My SNN")
    suite.print_summary(summary)

Single-Scenario Usage (Classical)::

    from embark.benchmark import (
        PMSMCurrentControlTask,
        PIControllerAgent,
        ClosedLoopHarness,
        TrackingMAE,
    )

    task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    harness = ClosedLoopHarness(task=task, controller=controller)
    results = harness.run()

NeuroBench Integration (experimental):
    See ``embark.benchmark.contrib.neurobench`` for interop utilities.

"""

from .adapters import TensorControllerAdapter
from .agents import (
    PIControllerAgent,
    PIParameters,
    SNNControllerAgent,
    SNNControllerTorchAgent,
)
from .controllers import ANNControllerWrapper, SNNControllerWrapper
from .harness import (
    BenchmarkSuite,
    BenchmarkSummary,
    ClosedLoopHarness,
    QUICK_SCENARIOS,
    STANDARD_SCENARIOS,
    ScenarioDefinition,
)
from .metrics import (
    MaximumError,
    Overshoot,
    SettlingTime,
    TrackingITAE,
    TrackingMAE,
)
from .physics import PMSMConfig, PMSMPhysicsEngine
from .tasks import PMSMCurrentControlTask, SafetyLimits

__all__ = [
    # Harness & Suite
    "BenchmarkSuite",
    "BenchmarkSummary",
    "ClosedLoopHarness",
    "QUICK_SCENARIOS",
    "STANDARD_SCENARIOS",
    "ScenarioDefinition",
    # Tasks
    "PMSMCurrentControlTask",
    "SafetyLimits",
    # Physics
    "PMSMConfig",
    "PMSMPhysicsEngine",
    # Adapters
    "TensorControllerAdapter",
    # Controllers
    "PIControllerAgent",
    "PIParameters",
    "SNNControllerAgent",
    "SNNControllerTorchAgent",
    "ANNControllerWrapper",
    "SNNControllerWrapper",
    # Metrics
    "MaximumError",
    "Overshoot",
    "SettlingTime",
    "TrackingMAE",
    "TrackingITAE",
]
