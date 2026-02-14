"""Unit tests for processors (normalization/denormalization/PWM)."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from embark.benchmark.physics import PMSMConfig
from embark.benchmark.processors.decoders import (
    LinearActionProcessor,
    PWMActionProcessor,
)
from embark.benchmark.processors.identity import (
    IdentityActionProcessor,
    IdentityStateProcessor,
)
from embark.benchmark.processors.normalizers import (
    MinMaxProcessor,
    SNNStateProcessor,
    StandardScalerProcessor,
)
from embark.benchmark.processors.pwm import PWMConverter


class TestMinMaxProcessor:
    """Test MinMax normalization."""

    def test_normalization_math(self):
        """Verify normalization logic with explicit bounds."""
        processor = MinMaxProcessor(input_keys=["val"], reference_keys=[])
        # Manually set bounds to avoid needing a config object for this specific test
        processor.bounds = {"val": (-10.0, 10.0)}

        # Test -10 -> -1
        res = processor({"val": -10.0}, {})
        assert res[0].item() == pytest.approx(-1.0)

        # Test 10 -> 1
        res = processor({"val": 10.0}, {})
        assert res[0].item() == pytest.approx(1.0)

        # Test 0 -> 0
        res = processor({"val": 0.0}, {})
        assert res[0].item() == pytest.approx(0.0)

        # Test 5 -> 0.5
        res = processor({"val": 5.0}, {})
        assert res[0].item() == pytest.approx(0.5)

    def test_configure_sets_correct_bounds(self):
        """Verify configure method sets bounds based on physics config."""
        config = PMSMConfig(i_max=20.0, u_max=100.0)
        processor = MinMaxProcessor(
            input_keys=["i_d", "omega", "other"], reference_keys=["i_q_ref"]
        )

        processor.configure(config, MagicMock())

        # Current should be +/- i_max
        assert processor.bounds["i_d"] == (-20.0, 20.0)
        assert processor.bounds["i_q_ref"] == (-20.0, 20.0)

        # Other should be default +/- 1.0
        assert processor.bounds["other"] == (-1.0, 1.0)


class TestLinearActionProcessor:
    """Test LinearAction denormalization."""

    def test_denormalization_math(self):
        """Verify denormalization logic with explicit bounds."""
        processor = LinearActionProcessor(
            output_keys=["val"], bounds={"val": (-100.0, 100.0)}
        )

        # Test -1.0 -> -100.0
        tensor = torch.tensor([-1.0])
        res = processor(tensor, MagicMock())
        assert res["val"] == pytest.approx(-100.0)

        # Test 1.0 -> 100.0
        tensor = torch.tensor([1.0])
        res = processor(tensor, MagicMock())
        assert res["val"] == pytest.approx(100.0)

        # Test 0.0 -> 0.0
        tensor = torch.tensor([0.0])
        res = processor(tensor, MagicMock())
        assert res["val"] == pytest.approx(0.0)

        # Test 0.5 -> 50.0
        tensor = torch.tensor([0.5])
        res = processor(tensor, MagicMock())
        assert res["val"] == pytest.approx(50.0)

    def test_dimension_mismatch_raises_error(self):
        """Verify error raised if tensor is too small."""
        processor = LinearActionProcessor(
            output_keys=["v_d", "v_q"], bounds={"v_d": (-1, 1), "v_q": (-1, 1)}
        )

        tensor = torch.tensor([0.5])  # Only 1 value, need 2
        with pytest.raises(ValueError, match="smaller than number of output keys"):
            processor(tensor, MagicMock())


# ---------------------------------------------------------------------------
# PWMConverter
# ---------------------------------------------------------------------------


class TestPWMConverter:
    """Tests for the low-level PWM converter."""

    def test_voltage_to_duty_zero_maps_to_half(self):
        pwm = PWMConverter(v_dc=48.0)
        assert pwm.voltage_to_duty(0.0) == pytest.approx(0.5)

    def test_voltage_to_duty_full_positive(self):
        pwm = PWMConverter(v_dc=48.0)
        assert pwm.voltage_to_duty(48.0) == pytest.approx(1.0)

    def test_voltage_to_duty_full_negative(self):
        pwm = PWMConverter(v_dc=48.0)
        assert pwm.voltage_to_duty(-48.0) == pytest.approx(0.0)

    def test_voltage_to_duty_clamps_above(self):
        """Voltages above v_dc should saturate at duty = 1."""
        pwm = PWMConverter(v_dc=48.0)
        assert pwm.voltage_to_duty(100.0) == pytest.approx(1.0)

    def test_voltage_to_duty_clamps_below(self):
        """Voltages below -v_dc should saturate at duty = 0."""
        pwm = PWMConverter(v_dc=48.0)
        assert pwm.voltage_to_duty(-100.0) == pytest.approx(0.0)

    def test_duty_to_voltage_roundtrip(self):
        """duty_to_voltage(voltage_to_duty(v)) == v for in-range voltages."""
        pwm = PWMConverter(v_dc=48.0, dead_time=0.0)
        for v in [-24.0, 0.0, 12.0, 48.0]:
            d = pwm.voltage_to_duty(v)
            v_back = pwm.duty_to_voltage(d)
            assert v_back == pytest.approx(v, abs=1e-10)

    def test_dead_time_error_positive_current(self):
        """Positive current -> negative dead-time error."""
        pwm = PWMConverter(v_dc=48.0, dead_time=2e-6, pwm_frequency=10_000.0)
        err = pwm.dead_time_error(1.0)
        expected = -48.0 * 2e-6 * 10_000.0  # -0.96 V
        assert err == pytest.approx(expected)

    def test_dead_time_error_negative_current(self):
        """Negative current -> positive dead-time error."""
        pwm = PWMConverter(v_dc=48.0, dead_time=2e-6, pwm_frequency=10_000.0)
        err = pwm.dead_time_error(-1.0)
        assert err == pytest.approx(0.96)

    def test_dead_time_error_zero_current(self):
        """Zero current -> zero dead-time error."""
        pwm = PWMConverter(v_dc=48.0, dead_time=2e-6, pwm_frequency=10_000.0)
        assert pwm.dead_time_error(0.0) == 0.0

    def test_dead_time_disabled(self):
        """dead_time=0 should produce zero error."""
        pwm = PWMConverter(v_dc=48.0, dead_time=0.0)
        assert pwm.dead_time_error(5.0) == 0.0

    def test_convert_dq_returns_all_keys(self):
        pwm = PWMConverter(v_dc=48.0)
        result = pwm.convert_dq(v_d=10.0, v_q=20.0)
        assert set(result.keys()) == {"v_d", "v_q", "duty_d", "duty_q"}

    def test_convert_dq_no_dead_time_identity(self):
        """With dead_time=0 the effective voltage should round-trip exactly."""
        pwm = PWMConverter(v_dc=48.0, dead_time=0.0)
        result = pwm.convert_dq(v_d=12.0, v_q=-24.0)
        assert result["v_d"] == pytest.approx(12.0, abs=1e-10)
        assert result["v_q"] == pytest.approx(-24.0, abs=1e-10)

    def test_convert_dq_dead_time_shifts_voltage(self):
        """Dead-time should shift effective voltage away from the commanded value."""
        pwm = PWMConverter(v_dc=48.0, dead_time=2e-6, pwm_frequency=10_000.0)
        result = pwm.convert_dq(v_d=10.0, v_q=10.0, i_d=5.0, i_q=5.0)
        # Positive current => negative error => effective < commanded
        assert result["v_d"] < 10.0
        assert result["v_q"] < 10.0

    def test_invalid_v_dc_raises(self):
        with pytest.raises(ValueError, match="v_dc must be positive"):
            PWMConverter(v_dc=-10.0)


# ---------------------------------------------------------------------------
# PWMActionProcessor
# ---------------------------------------------------------------------------


class TestPWMActionProcessor:
    """Tests for the PWM action processor pipeline."""

    @pytest.fixture
    def configured_processor(self):
        """A PWMActionProcessor pre-configured with default physics."""
        proc = PWMActionProcessor(output_keys=["v_d", "v_q"])
        config = PMSMConfig()
        proc.configure(config)
        return proc

    def test_output_contains_voltage_and_duty_keys(self, configured_processor):
        action = torch.tensor([0.0, 0.0])
        result = configured_processor(action, MagicMock())
        assert "v_d" in result
        assert "v_q" in result
        assert "duty_d" in result
        assert "duty_q" in result

    def test_zero_action_maps_to_zero_voltage(self, configured_processor):
        """Normalized 0 → physical 0V (centre of symmetric bounds)."""
        action = torch.tensor([0.0, 0.0])
        result = configured_processor(action, MagicMock())
        assert result["v_d"] == pytest.approx(0.0, abs=1.0)  # within dead-time error
        assert result["v_q"] == pytest.approx(0.0, abs=1.0)

    def test_positive_saturation(self, configured_processor):
        """Normalized +1 → u_max, clamped by v_dc."""
        action = torch.tensor([1.0, 1.0])
        result = configured_processor(action, MagicMock())
        # Should be near u_max (48V), possibly slightly lower due to dead-time
        assert result["v_d"] > 0
        assert result["v_q"] > 0

    def test_duty_cycles_in_range(self, configured_processor):
        """Duty cycles must always be in [0, 1]."""
        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            action = torch.tensor([val, val])
            result = configured_processor(action, MagicMock())
            assert 0.0 <= result["duty_d"] <= 1.0
            assert 0.0 <= result["duty_q"] <= 1.0

    def test_set_currents_affects_dead_time(self):
        """set_currents should change dead-time compensation direction."""
        proc = PWMActionProcessor(output_keys=["v_d", "v_q"])
        config = PMSMConfig()
        proc.configure(config)

        action = torch.tensor([0.5, 0.5])

        # With zero current (default) — no dead-time shift direction
        proc(action, MagicMock())

        # With positive current — negative voltage error
        proc.set_currents(i_d=5.0, i_q=5.0)
        result_pos = proc(action, MagicMock())

        # With negative current — positive voltage error
        proc.set_currents(i_d=-5.0, i_q=-5.0)
        result_neg = proc(action, MagicMock())

        # Positive current should give lower voltage, negative should give higher
        assert result_pos["v_d"] < result_neg["v_d"]
        assert result_pos["v_q"] < result_neg["v_q"]

    def test_configure_from_physics_config(self):
        """PWM parameters should be read from physics config."""
        proc = PWMActionProcessor(output_keys=["v_d", "v_q"])
        config = PMSMConfig()
        proc.configure(config)

        assert proc._pwm is not None
        assert proc._pwm.v_dc == config.v_dc
        assert proc._pwm.dead_time == config.dead_time
        assert proc._pwm.pwm_frequency == config.pwm_frequency

    def test_dimension_mismatch_raises_error(self):
        """Verify error raised if tensor is too small."""
        proc = PWMActionProcessor(output_keys=["v_d", "v_q"])
        proc.configure(PMSMConfig())

        tensor = torch.tensor([0.5])  # Only 1 value, need 2
        with pytest.raises(ValueError, match="smaller than number of output keys"):
            proc(tensor, MagicMock())


@dataclass
class _ConfigStub:
    i_max: float = 10.0
    u_max: float = 48.0
    tau: float = 1e-4
    omega_max: float = 400.0


class _TaskStub:
    class _Physics:
        state_keys = {"i_d", "i_q", "omega"}

    physics_engine = _Physics()
    reference_keys = {"i_d_ref", "i_q_ref"}


def test_identity_state_processor_configure_defaults_and_call():
    proc = IdentityStateProcessor()
    proc.configure(_ConfigStub(), _TaskStub())

    out = proc(
        state={"i_d": 1.0, "i_q": 2.0, "omega": 3.0},
        reference={"i_d_ref": 4.0, "i_q_ref": 5.0},
    )

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (5,)
    assert proc.output_dim == 5


def test_identity_state_processor_raises_on_inconsistent_iterables():
    proc = IdentityStateProcessor(
        state_keys=iter(["i_d"]),  # type: ignore[arg-type]
        reference_keys=iter(["i_q_ref"]),  # type: ignore[arg-type]
    )

    with pytest.raises(KeyError, match="missing required keys"):
        proc(
            state={"i_d": 1.0},
            reference={"i_q_ref": 2.0},
        )


def test_identity_action_processor_requires_action_keys_on_configure():
    proc = IdentityActionProcessor()
    with pytest.raises(ValueError, match="requires action_keys"):
        proc.configure(_ConfigStub())


def test_identity_action_processor_raises_on_too_small_tensor():
    proc = IdentityActionProcessor(action_keys=["v_d", "v_q"])
    proc.configure(_ConfigStub())

    with pytest.raises(ValueError, match="smaller than number of action keys"):
        proc(torch.tensor([0.1]), _ConfigStub())


def test_standard_scaler_default_config_and_transform():
    proc = StandardScalerProcessor(
        input_keys=["i_d", "i_q"],
        reference_keys=["i_d_ref", "i_q_ref"],
    )
    proc.configure(_ConfigStub(), _TaskStub())

    out = proc(
        state={"i_d": 1.0, "i_q": -1.0},
        reference={"i_d_ref": 2.0, "i_q_ref": -2.0},
    )

    assert out.tolist() == pytest.approx([1.0, -1.0, 2.0, -2.0])
    assert proc.output_dim == 4


def test_snn_state_processor_configure_reads_i_max():
    proc = SNNStateProcessor(error_gain=10.0, n_max=4000.0)
    proc.configure(_ConfigStub(i_max=12.0), _TaskStub())

    out = proc(
        state={"i_d": 6.0, "i_q": 0.0, "omega": 0.0},
        reference={"i_d_ref": 6.0, "i_q_ref": 0.0},
    )
    assert float(out[0]) == pytest.approx(0.5)


def test_minmax_processor_normalizes_reference_branch_and_output_dim():
    proc = MinMaxProcessor(input_keys=["i_d"], reference_keys=["i_q_ref"])
    proc.configure(_ConfigStub(i_max=10.0), _TaskStub())

    out = proc(
        state={"i_d": 5.0},
        reference={"i_q_ref": -5.0},
    )
    assert out.tolist() == pytest.approx([0.5, -0.5])
    assert proc.output_dim == 2
