"""Unit tests for InferenceLatency accumulator."""

from __future__ import annotations

import pytest

from embark.benchmark.metrics.accumulators.latency import InferenceLatency

# Helpers ----------------------------------------------------------------

_DUMMY_STATE = {"i_d": 0.0, "i_q": 0.0}
_DUMMY_REF = {"i_d_ref": 0.0, "i_q_ref": 0.0}
_DUMMY_ACTION = {"v_d": 0.0, "v_q": 0.0}


def _step(metric, controller_info=None):
    """Single metric update with dummy state/reference/action."""
    metric.update(
        _DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_STATE, controller_info
    )


# Tests ------------------------------------------------------------------


class TestInferenceLatency:
    def test_no_data_returns_zeros(self):
        m = InferenceLatency()
        m.reset()
        result = m.compute()
        assert result["mean_latency_ms"] == 0.0
        assert result["p95_latency_ms"] == 0.0
        assert result["p99_latency_ms"] == 0.0
        assert result["max_latency_ms"] == 0.0
        assert result["jitter_ms"] == 0.0
        assert result["total_inference_time_s"] == 0.0

    def test_constant_latency(self):
        m = InferenceLatency()
        m.reset()
        for _ in range(100):
            _step(m, {"inference_latency_s": 0.001})
        result = m.compute()
        assert result["mean_latency_ms"] == pytest.approx(1.0)
        assert result["p95_latency_ms"] == pytest.approx(1.0)
        assert result["max_latency_ms"] == pytest.approx(1.0)
        assert result["jitter_ms"] == pytest.approx(0.0)
        assert result["total_inference_time_s"] == pytest.approx(0.1)

    def test_varying_latency_percentiles(self):
        m = InferenceLatency()
        m.reset()
        # 100 samples: 0.001, 0.002, ..., 0.100 s
        for i in range(1, 101):
            _step(m, {"inference_latency_s": i * 0.001})
        result = m.compute()
        # mean = 50.5 ms
        assert result["mean_latency_ms"] == pytest.approx(50.5)
        # p95 should be > mean
        assert result["p95_latency_ms"] > result["mean_latency_ms"]
        # p99 >= p95
        assert result["p99_latency_ms"] >= result["p95_latency_ms"]
        # max >= p99
        assert result["max_latency_ms"] >= result["p99_latency_ms"]
        assert result["max_latency_ms"] == pytest.approx(100.0)
        # jitter should be > 0 for varying data
        assert result["jitter_ms"] > 0.0

    def test_reset_clears_state(self):
        m = InferenceLatency()
        m.reset()
        _step(m, {"inference_latency_s": 0.005})
        assert m.compute()["mean_latency_ms"] > 0.0
        m.reset()
        result = m.compute()
        assert result["mean_latency_ms"] == 0.0

    def test_missing_controller_info(self):
        """None controller_info should not raise."""
        m = InferenceLatency()
        m.reset()
        _step(m, None)
        result = m.compute()
        assert result["mean_latency_ms"] == 0.0

    def test_controller_info_without_latency_key(self):
        """Controller info with other keys but no latency should be safe."""
        m = InferenceLatency()
        m.reset()
        _step(m, {"total_spikes": 42, "syops": 100})
        result = m.compute()
        assert result["mean_latency_ms"] == 0.0

    def test_name_property(self):
        m = InferenceLatency()
        assert m.name == "inference_latency"

    def test_chip_latency_statistics(self):
        """On-chip latency statistics are computed in microseconds."""
        m = InferenceLatency()
        m.reset()

        # Two chip inference times: 1 µs and 3 µs (expressed in seconds)
        _step(m, {"chip_inference_time_s": 1e-6})
        _step(m, {"chip_inference_time_s": 3e-6})

        result = m.compute()

        assert result["chip_mean_us"] == pytest.approx(2.0)
        assert result["chip_median_us"] == pytest.approx(2.0)
        assert result["chip_p95_us"] >= result["chip_median_us"]
        assert result["chip_max_us"] == pytest.approx(3.0)
        assert result["chip_min_us"] == pytest.approx(1.0)
