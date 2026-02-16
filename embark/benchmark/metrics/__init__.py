"""Metric accumulators and registry."""

from .accumulators import (
    ActivationSparsity,
    InferenceLatency,
    MaximumError,
    Overshoot,
    SettlingTime,
    SpikeCount,
    SteadyStateRMS,
    SynapticOps,
    TrackingITAE,
    TrackingMAE,  # kept for backward compatibility
)
from .neurobench_factory import create_metrics
from .registry import MetricRegistry

__all__ = [
    "ActivationSparsity",
    "InferenceLatency",
    "MaximumError",
    "MetricRegistry",
    "Overshoot",
    "SettlingTime",
    "SpikeCount",
    "SteadyStateRMS",
    "SynapticOps",
    "TrackingITAE",
    "TrackingMAE",
    "create_metrics",
]
