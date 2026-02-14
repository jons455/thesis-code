"""
NeuroBench interoperability utilities (experimental).

This package provides adapters for integrating embark's closed-loop
benchmark framework with the NeuroBench ecosystem. These are **not**
required for running benchmarks — they are convenience wrappers for
users who want to use NeuroBench workload metrics alongside embark's
control metrics.

Status: Experimental. NeuroBench does not yet natively support
closed-loop benchmarks. These adapters will be updated as the upstream
NeuroBench API evolves (targeting the ``2025_GC`` branch).

See ``NEUROBENCH_INTEGRATION_ROADMAP.md`` in the project root for the
full integration plan.

"""

from .model_wrapper import NeuroBenchClosedLoopModel
from .metric_adapters import (
    NeuroBenchStaticMetricAdapter,
    NeuroBenchWorkloadMetricAdapter,
    discover_neurobench_metric_classes,
)
from .result_exporter import ClosedLoopMetricExporter

__all__ = [
    "ClosedLoopMetricExporter",
    "discover_neurobench_metric_classes",
    "NeuroBenchStaticMetricAdapter",
    "NeuroBenchWorkloadMetricAdapter",
    "NeuroBenchClosedLoopModel",
]
