"""Controller-aware metric factory with optional NeuroBench adapters."""

from __future__ import annotations

from inspect import signature
from typing import Any

import torch

from embark.benchmark.interfaces import MetricAccumulator
from embark.benchmark.metrics.accumulators import (
    ActivationSparsity,
    InferenceLatency,
    MaximumError,
    SpikeCount,
    SynapticOps,
    TrackingITAE,
    TrackingMAE,
)
from embark.benchmark.metrics.accumulators.dynamics import Overshoot, SettlingTime
from embark.benchmark.metrics.accumulators.tracking import SteadyStateRMS


def _supports_noarg_constructor(metric_cls: type) -> bool:
    """Check whether a metric class can be instantiated without arguments."""
    try:
        sig = signature(metric_cls)
    except (TypeError, ValueError):
        return True

    for param in sig.parameters.values():
        if param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        if param.default is param.empty:
            return False
    return True


def _control_metrics() -> list[MetricAccumulator]:
    """
    Create the standard set of control metrics used for all controllers.

    Included metrics:
    - ``TrackingMAE``: mean absolute error over full episode (i_q, i_d)
    - ``TrackingITAE``: transient accuracy over first 50 ms (i_q, i_d)
    - ``MaximumError``: worst-case peak error (i_q, i_d)
    - ``SettlingTime``: time to enter and stay within 2% band for ≥1 ms (i_q)
    - ``Overshoot``: peak overshoot relative to step size (i_q)
    - ``SteadyStateRMS``: RMS of steady-state error after 50 ms (i_q, i_d)
    - ``InferenceLatency``: round-trip and chip-level timing (all controllers)
    - ``SpikeCount``: total spikes emitted by SNN layers (silent for non-SNN)
    - ``SynapticOps``: synaptic operations (silent for non-SNN)
    - ``ActivationSparsity``: fraction of silent neurons (silent for non-SNN)

    """
    return [
        TrackingMAE(tracked_keys=["i_q", "i_d"]),
        TrackingITAE(tracked_keys=["i_q", "i_d"], window_s=0.05),
        MaximumError(tracked_keys=["i_q", "i_d"]),
        SettlingTime(tracked_key="i_q", band_fraction=0.02, dwell_s=0.001),
        Overshoot(tracked_key="i_q"),
        SteadyStateRMS(tracked_keys=["i_q", "i_d"], transient_s=0.05),
        InferenceLatency(),
        # Neuromorphic metrics — safe for non-SNN controllers (return zeros)
        SpikeCount(),
        SynapticOps(),
        ActivationSparsity(),
    ]


def _controller_has_model(controller: Any | None) -> bool:
    """True when the controller exposes a model object for NeuroBench hooks."""
    if controller is None:
        return False
    model = getattr(controller, "model", None)
    if model is None:
        return False
    return isinstance(model, torch.nn.Module)


def _create_neurobench_adapters(controller: Any) -> list[MetricAccumulator]:
    """
    Instantiate all applicable NeuroBench static/workload metric adapters.

    Returns empty list if neurobench is not installed.

    """
    try:
        from embark.benchmark.contrib.neurobench.metric_adapters import (
            NeuroBenchStaticMetricAdapter,
            NeuroBenchWorkloadMetricAdapter,
            discover_neurobench_metric_classes,
        )
    except ImportError:
        return []

    static_classes, workload_classes = discover_neurobench_metric_classes()
    adapters: list[MetricAccumulator] = []

    for metric_cls in static_classes:
        if not _supports_noarg_constructor(metric_cls):
            continue
        adapters.append(
            NeuroBenchStaticMetricAdapter(controller=controller, metric_cls=metric_cls)
        )

    for metric_cls in workload_classes:
        if not _supports_noarg_constructor(metric_cls):
            continue
        adapters.append(
            NeuroBenchWorkloadMetricAdapter(
                controller=controller, metric_cls=metric_cls
            )
        )
    return adapters


def create_metrics(controller: Any | None = None) -> list[MetricAccumulator]:
    """
    Create metrics for a benchmark run.

    Behavior:
    - Always includes control metrics.
    - Uses NeuroBench adapters when controller has `.model`.

    """
    metrics: list[MetricAccumulator] = _control_metrics()
    if _controller_has_model(controller):
        metrics.extend(_create_neurobench_adapters(controller))
    return metrics
