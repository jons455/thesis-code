"""
NeuroBench-aligned PMSM Current Control Benchmark.

This module provides a modular closed-loop benchmark framework following
NeuroBench architecture patterns.

Components:
-----------
- ClosedLoopHarness: Main benchmark orchestrator
- PMSMCurrentControlTask: PMSM current control task with physics engine
- PMSMPhysicsEngine: Pure physics dynamics (wraps GEM)
- PIControllerAgent: Classical PI controller (DictController)
- TensorControllerAdapter: Wraps TensorController + processors into unified interface

Architecture:
-------------
The harness follows a unified control loop without if/else branching:

    state, ref = task.reset()
    while not done:
        action = controller(state, ref)  # Unified interface
        state, ref, done = task.step(action)

Classical controllers implement Controller directly.
Neural controllers must be wrapped with TensorControllerAdapter.

Usage (Classical):
------------------
    from embark.benchmark import (
        PMSMCurrentControlTask,
        PIControllerAgent,
        ClosedLoopHarness,
        TrackingRMSE,
    )

    task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    harness = ClosedLoopHarness(task=task, controller=controller)
    results = harness.run()

Usage (Neural):
---------------
    from embark.benchmark import (
        PMSMCurrentControlTask,
        TensorControllerAdapter,
        ClosedLoopHarness,
    )
    from embark.benchmark.agents import SNNControllerAgent
    from embark.benchmark.processors import MinMaxProcessor, LinearActionProcessor

    task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)
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

from .adapters import TensorControllerAdapter
from .agents import (
    PIControllerAgent,
    PIParameters,
    SNNControllerAgent,
    SNNControllerTorchAgent,
)
from .controllers import ANNControllerWrapper, SNNControllerWrapper
from .harness import ClosedLoopHarness
from .metrics import (
    ControlEffort,
    Overshoot,
    SettlingTime,
    SyOpsAccumulator,
    TrackingRMSE,
)
from .physics import PMSMConfig, PMSMPhysicsEngine
from .tasks import PMSMCurrentControlTask, SafetyLimits

__all__ = [
    # Harness
    "ClosedLoopHarness",
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
    "ControlEffort",
    "Overshoot",
    "SettlingTime",
    "SyOpsAccumulator",
    "TrackingRMSE",
]
