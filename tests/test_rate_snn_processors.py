"""Unit tests for RateSNNStateProcessor and RateSNNActionProcessor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from embark.benchmark.processors.rate_snn import (
    RateSNNActionProcessor,
    RateSNNStateProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _ConfigStub:
    i_max: float = 10.0
    u_max: float = 12.0
    tau: float = 1e-4
    omega_max: float = 400.0


class _TaskStub:
    class _Physics:
        state_keys = {"i_d", "i_q", "omega"}

    physics_engine = _Physics()
    reference_keys = {"i_d_ref", "i_q_ref"}


def _make_state(i_d: float = 0.0, i_q: float = 0.0, omega: float = 0.0) -> dict:
    return {"i_d": i_d, "i_q": i_q, "omega": omega}


def _make_ref(i_d_ref: float = 0.0, i_q_ref: float = 0.0) -> dict:
    return {"i_d_ref": i_d_ref, "i_q_ref": i_q_ref}


def _rpm_to_omega(rpm: float) -> float:
    return rpm * 2 * math.pi / 60.0


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — output_dim
# ---------------------------------------------------------------------------


class TestRateSNNStateProcessorOutputDim:
    """Verify output_dim matches actual tensor length for each config."""

    def test_basic_output_dim(self):
        """Default config: currents + errors + speed = 5."""
        proc = RateSNNStateProcessor()
        assert proc.output_dim == 5

    def test_with_derivatives_and_ema(self):
        """Currents + errors + speed + derivatives + 2 EMAs = 12."""
        proc = RateSNNStateProcessor(
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        assert proc.output_dim == 12

    def test_with_references_prev_action_ema(self):
        """Currents + refs + errors + speed + prev_action + 2 EMAs = 13."""
        proc = RateSNNStateProcessor(
            include_references=True,
            include_prev_action=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        assert proc.output_dim == 13

    def test_all_features_enabled(self):
        proc = RateSNNStateProcessor(
            include_currents=True,
            include_references=True,
            include_errors=True,
            include_speed=True,
            include_prev_action=True,
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
            include_integral=True,
        )
        # 2+2+2+1+2+3+2+2+2 = 18
        assert proc.output_dim == 18

    def test_no_features_enabled(self):
        proc = RateSNNStateProcessor(
            include_currents=False,
            include_references=False,
            include_errors=False,
            include_speed=False,
        )
        assert proc.output_dim == 0

    @pytest.mark.parametrize(
        "kwargs,expected_dim",
        [
            ({}, 5),
            ({"include_derivatives": True}, 8),
            ({"include_references": True}, 7),
            ({"include_prev_action": True}, 7),
            ({"include_ema_slow": True}, 7),
            ({"include_ema_fast": True}, 7),
            ({"include_integral": True}, 7),
            ({"include_ema_slow": True, "include_ema_fast": True}, 9),
        ],
    )
    def test_dim_matches_tensor_length(self, kwargs, expected_dim):
        """output_dim must match actual tensor length."""
        proc = RateSNNStateProcessor(**kwargs)
        proc.configure(_ConfigStub(), _TaskStub())
        proc.reset()
        tensor = proc(_make_state(1.0, 2.0, _rpm_to_omega(1000)), _make_ref(0.0, 3.0))
        assert proc.output_dim == expected_dim
        assert tensor.shape == (expected_dim,)


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — basic features (currents, errors, speed)
# ---------------------------------------------------------------------------


class TestBasicFeatures:
    """Verify normalization of currents, errors, and speed."""

    @pytest.fixture
    def processor(self):
        proc = RateSNNStateProcessor(error_gain=10.0, n_max=4000.0)
        proc.configure(_ConfigStub(i_max=10.8), _TaskStub())
        proc.reset()
        return proc

    def test_current_normalization(self, processor):
        state = _make_state(i_d=5.4, i_q=3.2)
        ref = _make_ref()
        result = processor(state, ref)
        assert float(result[0]) == pytest.approx(5.4 / 10.8)
        assert float(result[1]) == pytest.approx(3.2 / 10.8)

    def test_error_with_gain_and_clipping(self, processor):
        state = _make_state(i_d=5.0, i_q=3.0)
        ref = _make_ref(i_d_ref=0.0, i_q_ref=5.0)
        result = processor(state, ref)
        # e_d = (0 - 5) / 10.8 * 10 = -4.63 -> clipped to -1
        assert float(result[2]) == pytest.approx(-1.0)
        # e_q = (5 - 3) / 10.8 * 10 = 1.85 -> clipped to 1
        assert float(result[3]) == pytest.approx(1.0)

    def test_error_no_clipping(self, processor):
        state = _make_state(i_d=5.0, i_q=3.0)
        ref = _make_ref(i_d_ref=5.1, i_q_ref=3.1)
        result = processor(state, ref)
        expected_e_d = (5.1 - 5.0) / 10.8 * 10.0
        expected_e_q = (3.1 - 3.0) / 10.8 * 10.0
        assert float(result[2]) == pytest.approx(expected_e_d)
        assert float(result[3]) == pytest.approx(expected_e_q)

    @pytest.mark.parametrize(
        "n_rpm,expected_norm",
        [
            (0, 0.0),
            (1000, 0.25),
            (2000, 0.50),
            (4000, 1.00),
        ],
    )
    def test_speed_normalization(self, processor, n_rpm, expected_norm):
        state = _make_state(omega=_rpm_to_omega(n_rpm))
        ref = _make_ref()
        result = processor(state, ref)
        assert float(result[4]) == pytest.approx(expected_norm, abs=1e-6)

    def test_output_shape_and_dtype(self, processor):
        result = processor(_make_state(1.0, 2.0, 100.0), _make_ref(0.0, 3.0))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)
        assert result.dtype == torch.float32

    def test_feature_order(self, processor):
        """Features must be [i_d, i_q, e_d, e_q, n]."""
        state = _make_state(i_d=2.0, i_q=4.0, omega=_rpm_to_omega(2000))
        ref = _make_ref(i_d_ref=2.0, i_q_ref=4.0)
        result = processor(state, ref)

        assert float(result[0]) == pytest.approx(2.0 / 10.8)  # i_d
        assert float(result[1]) == pytest.approx(4.0 / 10.8)  # i_q
        assert float(result[2]) == pytest.approx(0.0)  # e_d (ref == meas)
        assert float(result[3]) == pytest.approx(0.0)  # e_q (ref == meas)
        assert float(result[4]) == pytest.approx(0.5)  # n = 2000/4000


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — derivative and EMA features
# ---------------------------------------------------------------------------


class TestDerivativeAndEMAFeatures:
    """Test derivative and EMA features."""

    @pytest.fixture
    def processor(self):
        proc = RateSNNStateProcessor(
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        proc.configure(_ConfigStub(i_max=10.0), _TaskStub())
        proc.reset()
        return proc

    def test_first_step_derivatives_are_zero(self, processor):
        result = processor(_make_state(1.0, 2.0, 100.0), _make_ref(1.0, 2.0))
        # Derivatives are at indices 5, 6, 7
        assert float(result[5]) == 0.0  # de_d
        assert float(result[6]) == 0.0  # de_q
        assert float(result[7]) == 0.0  # dn

    def test_derivatives_nonzero_after_second_step(self, processor):
        processor(_make_state(1.0, 2.0, 100.0), _make_ref(2.0, 3.0))
        result = processor(_make_state(1.5, 2.5, 200.0), _make_ref(2.0, 3.0))
        # Errors changed, so derivatives should be nonzero
        assert float(result[5]) != 0.0
        assert float(result[6]) != 0.0

    def test_ema_initialized_on_first_step(self, processor):
        state = _make_state(1.0, 2.0)
        ref = _make_ref(2.0, 3.0)
        result = processor(state, ref)
        # First step: EMA = current error value
        e_d = (2.0 - 1.0) / 10.0 * 10.0  # = 1.0
        e_q = (3.0 - 2.0) / 10.0 * 10.0  # = 1.0
        # EMA slow at indices 8, 9
        assert float(result[8]) == pytest.approx(e_d)
        assert float(result[9]) == pytest.approx(e_q)
        # EMA fast at indices 10, 11
        assert float(result[10]) == pytest.approx(e_d)
        assert float(result[11]) == pytest.approx(e_q)

    def test_ema_smooths_over_time(self, processor):
        # Step 1: error = 1.0
        processor(_make_state(0.0, 0.0), _make_ref(1.0, 1.0))
        # Step 2: error = 0.0 (ref == meas)
        result = processor(_make_state(5.0, 5.0), _make_ref(5.0, 5.0))
        # Slow EMA (alpha=0.98): 0.98 * 1.0 + 0.02 * 0.0 = 0.98
        assert float(result[8]) == pytest.approx(0.98, abs=1e-4)
        # Fast EMA (alpha=0.70): 0.70 * 1.0 + 0.30 * 0.0 = 0.70
        assert float(result[10]) == pytest.approx(0.70, abs=1e-4)

    def test_output_dim_is_12(self, processor):
        assert processor.output_dim == 12


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — reference and prev_action features
# ---------------------------------------------------------------------------


class TestReferenceAndPrevActionFeatures:
    """Test reference and prev_action features."""

    @pytest.fixture
    def processor(self):
        proc = RateSNNStateProcessor(
            include_references=True,
            include_prev_action=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        proc.configure(_ConfigStub(i_max=10.0, u_max=12.0), _TaskStub())
        proc.reset()
        return proc

    def test_includes_references(self, processor):
        state = _make_state(1.0, 2.0)
        ref = _make_ref(3.0, 4.0)
        result = processor(state, ref)
        # References are at indices 2, 3 (after currents)
        assert float(result[2]) == pytest.approx(3.0 / 10.0)
        assert float(result[3]) == pytest.approx(4.0 / 10.0)

    def test_includes_prev_action(self, processor):
        state = _make_state(1.0, 2.0)
        ref = _make_ref(1.0, 2.0)
        # Set previous action
        processor.set_prev_action(6.0, -3.0)
        result = processor(state, ref)
        # Prev action at indices 7, 8 (after i_d, i_q, i_d_ref, i_q_ref, e_d, e_q, n)
        assert float(result[7]) == pytest.approx(6.0 / 12.0)
        assert float(result[8]) == pytest.approx(-3.0 / 12.0)

    def test_output_dim_is_13(self, processor):
        assert processor.output_dim == 13

    def test_feature_order(self, processor):
        """Features: [i_d, i_q, i_d_ref, i_q_ref, e_d, e_q, n, u_d_prev, u_q_prev,
        ema_slow_d, ema_slow_q, ema_fast_d, ema_fast_q]"""
        processor.set_prev_action(1.2, -0.6)
        state = _make_state(2.0, 3.0, _rpm_to_omega(1000))
        ref = _make_ref(2.0, 3.0)
        result = processor(state, ref)
        assert result.shape == (13,)


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — reset
# ---------------------------------------------------------------------------


class TestRateSNNStateProcessorReset:
    """Test that reset() clears all stateful features."""

    def test_reset_clears_derivatives(self):
        proc = RateSNNStateProcessor(
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        proc.configure(_ConfigStub(i_max=10.0), _TaskStub())
        proc.reset()

        # Run two steps to build derivative state
        proc(_make_state(1.0, 2.0), _make_ref(1.0, 2.0))
        proc(_make_state(2.0, 3.0), _make_ref(2.0, 3.0))

        # Reset and verify first step has zero derivatives again
        proc.reset()
        result = proc(_make_state(1.0, 2.0), _make_ref(1.0, 2.0))
        assert float(result[5]) == 0.0  # de_d
        assert float(result[6]) == 0.0  # de_q
        assert float(result[7]) == 0.0  # dn

    def test_reset_clears_ema(self):
        proc = RateSNNStateProcessor(
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        proc.configure(_ConfigStub(i_max=10.0), _TaskStub())
        proc.reset()

        # Build EMA state
        for _ in range(10):
            proc(_make_state(0.0, 0.0), _make_ref(5.0, 5.0))

        # Reset
        proc.reset()
        result = proc(_make_state(0.0, 0.0), _make_ref(0.0, 0.0))
        # EMA should be initialized to current error (0.0)
        assert float(result[8]) == pytest.approx(0.0)

    def test_reset_clears_prev_action(self):
        proc = RateSNNStateProcessor(
            include_references=True,
            include_prev_action=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        proc.configure(_ConfigStub(i_max=10.0, u_max=12.0), _TaskStub())
        proc.reset()
        proc.set_prev_action(6.0, -3.0)

        proc.reset()
        result = proc(_make_state(1.0, 2.0), _make_ref(1.0, 2.0))
        # After reset, prev action should be 0
        assert float(result[7]) == pytest.approx(0.0)
        assert float(result[8]) == pytest.approx(0.0)

    def test_reset_clears_integral(self):
        proc = RateSNNStateProcessor(
            include_currents=False,
            include_errors=False,
            include_speed=False,
            include_integral=True,
        )
        proc.configure(_ConfigStub(i_max=10.0), _TaskStub())
        proc.reset()

        # Accumulate integral
        for _ in range(100):
            proc(_make_state(0.0, 0.0), _make_ref(1.0, 1.0))

        proc.reset()
        result = proc(_make_state(0.0, 0.0), _make_ref(0.0, 0.0))
        assert float(result[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(result[1]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# RateSNNStateProcessor — integral feature
# ---------------------------------------------------------------------------


class TestIntegralFeature:
    """Test error integral accumulation and anti-windup."""

    @pytest.fixture
    def integral_proc(self):
        proc = RateSNNStateProcessor(
            include_currents=False,
            include_errors=False,
            include_speed=False,
            include_integral=True,
            integral_limit=0.5,
        )
        proc.configure(_ConfigStub(i_max=10.0, tau=1e-4), _TaskStub())
        proc.reset()
        return proc

    def test_integral_accumulates(self, integral_proc):
        # Error = (1.0 - 0.0)/10.0 = 0.1 per step
        # Integral += 0.1 * 1e-4 = 1e-5 per step
        for _ in range(10):
            result = integral_proc(_make_state(0.0, 0.0), _make_ref(1.0, 1.0))

        assert float(result[0]) == pytest.approx(10 * 0.1 * 1e-4, abs=1e-8)

    def test_integral_clamps_at_limit(self, integral_proc):
        # Run many steps to saturate
        for _ in range(100_000):
            result = integral_proc(_make_state(0.0, 0.0), _make_ref(10.0, 10.0))

        assert float(result[0]) == pytest.approx(0.5)  # integral_limit
        assert float(result[1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RateSNNActionProcessor — absolute mode
# ---------------------------------------------------------------------------


class TestRateSNNActionProcessorAbsolute:
    """Test absolute (non-incremental) action processing."""

    @pytest.fixture
    def processor(self):
        proc = RateSNNActionProcessor(incremental=False)
        proc.configure(_ConfigStub(u_max=12.0))
        return proc

    def test_scales_to_voltage(self, processor):
        result = processor(torch.tensor([1.0, -1.0]), MagicMock())
        assert result["v_d"] == pytest.approx(12.0)
        assert result["v_q"] == pytest.approx(-12.0)

    def test_zero_action(self, processor):
        result = processor(torch.tensor([0.0, 0.0]), MagicMock())
        assert result["v_d"] == pytest.approx(0.0)
        assert result["v_q"] == pytest.approx(0.0)

    def test_half_action(self, processor):
        result = processor(torch.tensor([0.5, -0.5]), MagicMock())
        assert result["v_d"] == pytest.approx(6.0)
        assert result["v_q"] == pytest.approx(-6.0)

    def test_dimension_mismatch(self, processor):
        with pytest.raises(ValueError, match="smaller than number of output keys"):
            processor(torch.tensor([0.5]), MagicMock())


# ---------------------------------------------------------------------------
# RateSNNActionProcessor — incremental mode
# ---------------------------------------------------------------------------


class TestRateSNNActionProcessorIncremental:
    """Test incremental (delta accumulation) action processing."""

    @pytest.fixture
    def processor(self):
        proc = RateSNNActionProcessor(incremental=True, delta_max=0.2)
        proc.configure(_ConfigStub(u_max=12.0))
        proc.reset()
        return proc

    def test_accumulates_deltas(self, processor):
        # delta = 1.0 * 0.2 = 0.2 per step
        result = processor(torch.tensor([1.0, 0.0]), MagicMock())
        assert result["v_d"] == pytest.approx(0.2)
        assert result["v_q"] == pytest.approx(0.0)

        result = processor(torch.tensor([1.0, 0.0]), MagicMock())
        assert result["v_d"] == pytest.approx(0.4)

    def test_clamps_to_u_max(self, processor):
        # Apply many positive deltas
        for _ in range(1000):
            result = processor(torch.tensor([1.0, 1.0]), MagicMock())
        assert result["v_d"] == pytest.approx(12.0)
        assert result["v_q"] == pytest.approx(12.0)

    def test_clamps_to_negative_u_max(self, processor):
        for _ in range(1000):
            result = processor(torch.tensor([-1.0, -1.0]), MagicMock())
        assert result["v_d"] == pytest.approx(-12.0)
        assert result["v_q"] == pytest.approx(-12.0)

    def test_reset_clears_accumulator(self, processor):
        processor(torch.tensor([1.0, 1.0]), MagicMock())
        processor(torch.tensor([1.0, 1.0]), MagicMock())
        processor.reset()
        result = processor(torch.tensor([0.0, 0.0]), MagicMock())
        assert result["v_d"] == pytest.approx(0.0)
        assert result["v_q"] == pytest.approx(0.0)

    def test_last_accumulated_property(self, processor):
        processor(torch.tensor([1.0, -0.5]), MagicMock())
        accum = processor.last_accumulated
        assert accum[0] == pytest.approx(0.2)
        assert accum[1] == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# TensorControllerAdapter integration
# ---------------------------------------------------------------------------


class TestAdapterIntegration:
    """Test that the adapter properly resets and feeds back to processors."""

    def test_adapter_resets_processors(self):
        from embark.benchmark.adapters.tensor_adapter import TensorControllerAdapter

        state_proc = RateSNNStateProcessor(
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        action_proc = RateSNNActionProcessor(incremental=True, delta_max=0.2)

        controller_mock = MagicMock()
        controller_mock.forward.return_value = torch.tensor([0.5, 0.5])
        controller_mock.reset.return_value = None

        adapter = TensorControllerAdapter(
            controller=controller_mock,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        config = _ConfigStub(i_max=10.0, u_max=12.0)
        adapter.configure(config, _TaskStub())

        # Run a step to build state
        adapter(_make_state(1.0, 2.0), _make_ref(1.0, 2.0))

        # Reset should propagate
        adapter.reset()
        assert state_proc._first_step is True
        assert action_proc._accum == [0.0, 0.0]

    def test_adapter_feeds_back_prev_action(self):
        from embark.benchmark.adapters.tensor_adapter import TensorControllerAdapter

        state_proc = RateSNNStateProcessor(
            include_references=True,
            include_prev_action=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        action_proc = RateSNNActionProcessor(incremental=False)

        controller_mock = MagicMock()
        controller_mock.forward.return_value = torch.tensor([0.5, -0.5])
        controller_mock.reset.return_value = None

        adapter = TensorControllerAdapter(
            controller=controller_mock,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        config = _ConfigStub(i_max=10.0, u_max=12.0)
        adapter.configure(config, _TaskStub())
        adapter.reset()

        # Run a step - action should be fed back
        adapter(_make_state(1.0, 2.0), _make_ref(1.0, 2.0))

        # Prev action should be set (v_d = 0.5 * 12 = 6.0, v_q = -0.5 * 12 = -6.0)
        assert state_proc._prev_u_d == pytest.approx(6.0)
        assert state_proc._prev_u_q == pytest.approx(-6.0)
