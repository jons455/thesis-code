"""Inference latency accumulator for hardware-in-the-loop benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


@dataclass
class InferenceLatency(MetricAccumulator):
    """
    Accumulate per-step inference latency from controller info.

    Reads ``inference_latency_s`` (round-trip) and ``chip_inference_time_s``
    (on-chip only) from the ``controller_info`` dict produced by
    :class:`RemoteAkidaPolicy` and computes summary statistics
    (mean, percentiles, jitter).

    Safe to use with controllers that do not report latency -- all
    computed values default to zero when no data is collected.

    """

    _latencies: list[float] = field(default_factory=list, init=False, repr=False)
    _chip_latencies: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def name(self) -> str:
        return "inference_latency"

    def reset(self) -> None:
        self._latencies.clear()
        self._chip_latencies.clear()

    def update(
        self,
        _state: StateDict,
        _reference: ReferenceDict,
        _action: ActionDict,
        _next_state: StateDict,
        controller_info: dict | None = None,
    ) -> None:
        if controller_info and "inference_latency_s" in controller_info:
            self._latencies.append(float(controller_info["inference_latency_s"]))
        if controller_info and "chip_inference_time_s" in controller_info:
            self._chip_latencies.append(float(controller_info["chip_inference_time_s"]))

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}

        # Round-trip latency stats
        if not self._latencies:
            results.update(
                {
                    "mean_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "jitter_ms": 0.0,
                    "total_inference_time_s": 0.0,
                }
            )
        else:
            arr = np.array(self._latencies)
            results.update(
                {
                    "mean_latency_ms": float(np.mean(arr) * 1000.0),
                    "p95_latency_ms": float(np.percentile(arr, 95) * 1000.0),
                    "p99_latency_ms": float(np.percentile(arr, 99) * 1000.0),
                    "max_latency_ms": float(np.max(arr) * 1000.0),
                    "jitter_ms": float(np.std(arr) * 1000.0),
                    "total_inference_time_s": float(np.sum(arr)),
                }
            )

        # On-chip inference time stats
        if not self._chip_latencies:
            results.update(
                {
                    "chip_mean_us": 0.0,
                    "chip_median_us": 0.0,
                    "chip_p95_us": 0.0,
                    "chip_p99_us": 0.0,
                    "chip_max_us": 0.0,
                    "chip_min_us": 0.0,
                }
            )
        else:
            chip = np.array(self._chip_latencies)
            results.update(
                {
                    "chip_mean_us": float(np.mean(chip) * 1e6),
                    "chip_median_us": float(np.median(chip) * 1e6),
                    "chip_p95_us": float(np.percentile(chip, 95) * 1e6),
                    "chip_p99_us": float(np.percentile(chip, 99) * 1e6),
                    "chip_max_us": float(np.max(chip) * 1e6),
                    "chip_min_us": float(np.min(chip) * 1e6),
                }
            )

        return results
