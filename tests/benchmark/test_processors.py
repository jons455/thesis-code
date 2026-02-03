"""Unit tests for processors (normalization/denormalization)."""

import pytest
import torch
from unittest.mock import MagicMock

from embark.benchmark.processors.normalizers import MinMaxProcessor
from embark.benchmark.processors.decoders import LinearActionProcessor
from embark.benchmark.physics import PMSMConfig


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
