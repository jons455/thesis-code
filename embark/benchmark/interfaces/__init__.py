"""Benchmark interface protocols and shared types."""

from .controller import Controller, DictController, TensorController
from .metrics import MetricAccumulator
from .physics import PhysicsEngine
from .processors import ActionProcessor, StateProcessor
from .task import ClosedLoopTask
from .types import ActionDict, ControllerInfo, ReferenceDict, StateDict, SystemConfig

__all__ = [
    "ActionDict",
    "ActionProcessor",
    "ClosedLoopTask",
    "Controller",
    "ControllerInfo",
    "DictController",
    "MetricAccumulator",
    "PhysicsEngine",
    "ReferenceDict",
    "StateDict",
    "StateProcessor",
    "SystemConfig",
    "TensorController",
]
