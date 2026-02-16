"""
Test normalization consistency for rate-SNN processors.

This test verifies that RateSNNStateProcessor produces correct normalization for rate-
encoding SNNs, matching the expected behavior from training datasets.

"""

import math

import numpy as np
import pytest
import torch

from embark.benchmark.processors import RateSNNStateProcessor
from embark.utils.config import DEFAULT_PMSM


class TestNormalizationConsistency:
    """Test that RateSNNStateProcessor normalization is correct."""

    @pytest.fixture
    def processor(self):
        """Create a configured RateSNNStateProcessor with basic features."""
        proc = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            error_gain=10.0,
            n_max=4000.0,
        )
        # Configure with i_max
        from unittest.mock import MagicMock

        config = MagicMock()
        config.i_max = DEFAULT_PMSM.i_max
        task = MagicMock()
        proc.configure(config, task)
        proc.reset()
        return proc

    def test_current_normalization(self, processor):
        """Test that currents are normalized by i_max."""
        state = {"i_d": 5.4, "i_q": 3.2, "omega": 0.0}
        reference = {"i_d_ref": 0.0, "i_q_ref": 0.0}

        result = processor(state, reference)

        expected_i_d = 5.4 / DEFAULT_PMSM.i_max
        expected_i_q = 3.2 / DEFAULT_PMSM.i_max

        # First two elements are i_d, i_q
        assert abs(float(result[0]) - expected_i_d) < 1e-6
        assert abs(float(result[1]) - expected_i_q) < 1e-6

    def test_error_normalization_and_clipping(self, processor):
        """Test that errors are normalized, amplified, and clipped."""
        state = {"i_d": 5.0, "i_q": 3.0, "omega": 0.0}
        reference = {"i_d_ref": 0.0, "i_q_ref": 5.0}

        result = processor(state, reference)

        # e_d = (0.0 - 5.0) / 10.8 * 10.0 = -4.63 → clipped to -1.0
        # e_q = (5.0 - 3.0) / 10.8 * 10.0 = 1.85 → clipped to 1.0
        expected_e_d = -1.0
        expected_e_q = 1.0

        # Elements: [i_d, i_q, e_d, e_q, n] → errors are at indices 2, 3
        assert abs(float(result[2]) - expected_e_d) < 1e-6
        assert abs(float(result[3]) - expected_e_q) < 1e-6

    def test_error_normalization_no_clipping(self, processor):
        """Test error normalization when values are within [-1, 1]."""
        state = {"i_d": 5.0, "i_q": 3.0, "omega": 0.0}
        reference = {"i_d_ref": 5.1, "i_q_ref": 3.1}

        result = processor(state, reference)

        # e_d = (5.1 - 5.0) / 10.8 * 10.0 = 0.0926
        # e_q = (3.1 - 3.0) / 10.8 * 10.0 = 0.0926
        expected_e_d = (5.1 - 5.0) / DEFAULT_PMSM.i_max * 10.0
        expected_e_q = (3.1 - 3.0) / DEFAULT_PMSM.i_max * 10.0

        # Elements: [i_d, i_q, e_d, e_q, n] → errors are at indices 2, 3
        assert abs(float(result[2]) - expected_e_d) < 1e-6
        assert abs(float(result[3]) - expected_e_q) < 1e-6

    @pytest.mark.parametrize(
        "n_rpm,expected_norm",
        [
            (0, 0.0),
            (500, 0.125),
            (1000, 0.250),
            (1500, 0.375),
            (2000, 0.500),
            (2500, 0.625),
            (3000, 0.750),
            (4000, 1.000),
        ],
    )
    def test_speed_normalization_matches_training(
        self, processor, n_rpm, expected_norm
    ):
        """
        Test that speed normalization matches training datasets.

        This is the critical test that verifies the fix for the speed normalization
        mismatch (4000 RPM vs 3000 RPM issue).

        """
        # Convert RPM to rad/s (as physics engine provides)
        omega = n_rpm * 2 * math.pi / 60.0

        state = {"i_d": 0.0, "i_q": 0.0, "omega": omega}
        reference = {"i_d_ref": 0.0, "i_q_ref": 0.0}

        result = processor(state, reference)
        actual_n = float(result[4])

        assert abs(actual_n - expected_norm) < 1e-6, (
            f"Speed normalization mismatch at {n_rpm} RPM:\n"
            f"  Expected: {expected_norm:.6f}\n"
            f"  Actual:   {actual_n:.6f}\n"
            f"  Difference: {abs(actual_n - expected_norm):.6e}"
        )

    def test_output_shape(self, processor):
        """Test that processor returns correct shape."""
        state = {"i_d": 1.0, "i_q": 2.0, "omega": 100.0}
        reference = {"i_d_ref": 0.0, "i_q_ref": 3.0}

        result = processor(state, reference)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)
        assert result.dtype == torch.float32

    def test_output_dim_property(self, processor):
        """Test that output_dim property is correct."""
        assert processor.output_dim == 5

    def test_zero_speed(self, processor):
        """Test normalization with zero speed."""
        state = {"i_d": 5.0, "i_q": 3.0, "omega": 0.0}
        reference = {"i_d_ref": 0.0, "i_q_ref": 5.0}

        result = processor(state, reference)

        # Speed should be 0.0
        assert float(result[4]) == 0.0

    def test_negative_speed(self, processor):
        """Test normalization with negative speed (reverse rotation)."""
        n_rpm = -1500
        omega = n_rpm * 2 * math.pi / 60.0

        state = {"i_d": 0.0, "i_q": 0.0, "omega": omega}
        reference = {"i_d_ref": 0.0, "i_q_ref": 0.0}

        result = processor(state, reference)

        expected_n = n_rpm / 4000.0  # -0.375
        actual_n = float(result[4])

        assert abs(actual_n - expected_n) < 1e-6

    def test_full_normalization_example(self, processor):
        """
        Test complete normalization with realistic values.

        This test simulates a typical operating point and verifies all five normalized
        features match expected values.

        """
        # Operating point: 1500 RPM, 5A q-axis current, tracking 6A
        n_rpm = 1500
        omega = n_rpm * 2 * math.pi / 60.0  # 157.08 rad/s

        state = {"i_d": 0.0, "i_q": 5.0, "omega": omega}
        reference = {"i_d_ref": 0.0, "i_q_ref": 6.0}

        result = processor(state, reference)

        # Expected values
        expected = [
            0.0 / DEFAULT_PMSM.i_max,  # i_d
            5.0 / DEFAULT_PMSM.i_max,  # i_q
            (0.0 - 0.0) / DEFAULT_PMSM.i_max * 10.0,  # e_d (clipped)
            (6.0 - 5.0) / DEFAULT_PMSM.i_max * 10.0,  # e_q
            1500 / 4000.0,  # n
        ]

        for i, (actual, exp) in enumerate(zip(result, expected)):
            assert abs(float(actual) - exp) < 1e-6, (
                f"Feature {i} mismatch:\n"
                f"  Expected: {exp:.6f}\n"
                f"  Actual:   {float(actual):.6f}"
            )


class TestTrainingDatasetCompatibility:
    """
    Test that processor matches training dataset normalization.

    These tests simulate the normalization done in the training datasets and verify that
    the processor produces identical results.

    """

    def test_matches_pytorch_dataset_normalization(self):
        """Test that processor matches PyTorch dataset normalization."""
        # Simulate PyTorch dataset normalization
        # (from evaluation/pytorch_snn/utils/dataset.py)

        i_max = DEFAULT_PMSM.i_max
        n_max = 4000.0
        error_gain = 10.0

        # Raw values
        i_d = 5.0
        i_q = 3.0
        i_d_ref = 0.0
        i_q_ref = 5.0
        n_rpm = 1500

        # PyTorch dataset normalization
        i_d_norm_dataset = i_d / i_max
        i_q_norm_dataset = i_q / i_max
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q
        e_d_norm_dataset = np.clip((e_d / i_max) * error_gain, -1.0, 1.0)
        e_q_norm_dataset = np.clip((e_q / i_max) * error_gain, -1.0, 1.0)
        n_norm_dataset = n_rpm / n_max

        # Processor normalization
        proc = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            error_gain=error_gain,
            n_max=n_max,
        )
        from unittest.mock import MagicMock

        config = MagicMock()
        config.i_max = i_max
        proc.configure(config, MagicMock())
        proc.reset()
        processor = proc

        omega = n_rpm * 2 * math.pi / 60.0
        state = {"i_d": i_d, "i_q": i_q, "omega": omega}
        reference = {"i_d_ref": i_d_ref, "i_q_ref": i_q_ref}

        result = processor(state, reference)

        # Compare
        assert abs(float(result[0]) - i_d_norm_dataset) < 1e-6
        assert abs(float(result[1]) - i_q_norm_dataset) < 1e-6
        assert abs(float(result[2]) - e_d_norm_dataset) < 1e-6
        assert abs(float(result[3]) - e_q_norm_dataset) < 1e-6
        assert abs(float(result[4]) - n_norm_dataset) < 1e-6

    def test_matches_akida_dataset_normalization(self):
        """Test that processor matches Akida dataset normalization."""
        # Simulate Akida dataset normalization
        # (from evaluation/akida/utils/dataset.py)

        i_max = DEFAULT_PMSM.i_max
        n_max = 4000.0
        error_gain = 10.0

        # Raw values
        i_d = 5.0
        i_q = 3.0
        i_d_ref = 0.0
        i_q_ref = 5.0
        n_rpm = 1500

        # Akida dataset normalization (identical to PyTorch)
        i_d_norm_dataset = i_d / i_max
        i_q_norm_dataset = i_q / i_max
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q
        e_d_norm_dataset = np.clip((e_d / i_max) * error_gain, -1.0, 1.0)
        e_q_norm_dataset = np.clip((e_q / i_max) * error_gain, -1.0, 1.0)
        n_norm_dataset = n_rpm / n_max

        # Processor normalization
        proc = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            error_gain=error_gain,
            n_max=n_max,
        )
        from unittest.mock import MagicMock

        config = MagicMock()
        config.i_max = i_max
        proc.configure(config, MagicMock())
        proc.reset()
        processor = proc

        omega = n_rpm * 2 * math.pi / 60.0
        state = {"i_d": i_d, "i_q": i_q, "omega": omega}
        reference = {"i_d_ref": i_d_ref, "i_q_ref": i_q_ref}

        result = processor(state, reference)

        # Compare
        assert abs(float(result[0]) - i_d_norm_dataset) < 1e-6
        assert abs(float(result[1]) - i_q_norm_dataset) < 1e-6
        assert abs(float(result[2]) - e_d_norm_dataset) < 1e-6
        assert abs(float(result[3]) - e_q_norm_dataset) < 1e-6
        assert abs(float(result[4]) - n_norm_dataset) < 1e-6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
