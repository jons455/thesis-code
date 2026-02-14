"""
Export harness results in NeuroBench-compatible reporting format.

This utility reformats the dictionary returned by
``ClosedLoopHarness.run()`` into a structure suitable for NeuroBench
comparison tables, separating control metrics from workload metrics.

.. note::

    Experimental. The exact NeuroBench reporting format may change as
    the upstream project evolves.

Usage::

    from embark.benchmark.contrib.neurobench import ClosedLoopMetricExporter

    results = harness.run()
    exporter = ClosedLoopMetricExporter(
        benchmark_name="pmsm_iq_step_1000rpm",
        model_name="SNN-PI-Imitation",
    )
    report = exporter.to_neurobench_format(results)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ClosedLoopMetricExporter:
    """
    Reformat harness results for NeuroBench-style comparison tables.

    Args:
        benchmark_name: Name of the benchmark scenario.
        model_name: Name of the evaluated controller / model.

    """

    benchmark_name: str = "closed_loop_control"
    model_name: str = "unknown"

    # Keys recognised as control-quality metrics
    _CONTROL_KEYS: tuple[str, ...] = (
        "mae_i_q",
        "mae_i_d",
        "itae_i_q",
        "itae_i_d",
        "max_error_i_q",
        "max_error_i_d",
        "settling_time",
        "overshoot",
        "settling_time_i_q",
        "overshoot_i_q",
    )

    # Keys recognised as neuromorphic workload metrics
    _WORKLOAD_KEYS: tuple[str, ...] = (
        "total_syops",
        "syops_per_step",
    )

    def to_neurobench_format(self, harness_results: dict[str, Any]) -> dict[str, Any]:
        """
        Convert harness results to NeuroBench-compatible dict.

        Args:
            harness_results: Dictionary returned by
                ``ClosedLoopHarness.run()``.

        Returns:
            Dictionary with ``benchmark``, ``model``, ``steps``,
            ``control_metrics``, and ``workload_metrics`` keys.

        """
        control_metrics = {
            k: v for k, v in harness_results.items() if k in self._CONTROL_KEYS
        }
        workload_metrics = {
            k: v for k, v in harness_results.items() if k in self._WORKLOAD_KEYS
        }
        # Also capture any nb_-prefixed keys
        for k, v in harness_results.items():
            if k.startswith("nb_"):
                workload_metrics[k] = v

        return {
            "benchmark": self.benchmark_name,
            "model": self.model_name,
            "steps": harness_results.get("steps", 0),
            "control_metrics": control_metrics,
            "workload_metrics": workload_metrics,
        }
