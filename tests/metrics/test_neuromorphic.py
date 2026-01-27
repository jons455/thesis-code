"""
Unit tests for neuromorphic metrics.

Tests spike statistics, energy estimates, sparsity.
"""

import numpy as np
import pytest

from embark.metrics.benchmark_metrics import compute_neuromorphic_metrics_from_spikes


class TestNeuromorphicMetrics:
    """Test neuromorphic metric calculations."""

    @pytest.fixture
    def synthetic_snn_data(self):
        """Synthetic spike data."""
        num_neurons = 10
        num_timesteps = 100
        dt_snn = 1e-3

        # 10% spiking
        spike_trains = np.zeros((num_neurons, num_timesteps))
        spike_trains[:, :10] = 1  # Spike in first 10 steps

        # Weights (dense)
        weights = np.ones((num_neurons, num_neurons))

        return spike_trains, weights, dt_snn

    def test_spike_counting(self, synthetic_snn_data):
        """Total spikes should match sum of spike train."""
        spike_trains, weights, dt_snn = synthetic_snn_data
        metrics = compute_neuromorphic_metrics_from_spikes(
            spike_trains, weights, dt_snn
        )

        expected_spikes = 10 * 10  # 10 neurons * 10 spikes
        assert metrics.total_spikes == expected_spikes

    def test_sparsity_calculation(self, synthetic_snn_data):
        """Sparsity should be calculated correctly."""
        spike_trains, weights, dt_snn = synthetic_snn_data
        metrics = compute_neuromorphic_metrics_from_spikes(
            spike_trains, weights, dt_snn
        )

        # Activity sparsity: All neurons spike at least once -> 0% sparsity (activation)
        # But temporal sparsity: 1000 total slots, 100 spikes -> 90% temporal sparsity
        assert metrics.temporal_sparsity == pytest.approx(0.9)
        assert metrics.activation_sparsity == 0.0  # All neurons active

    def test_syops_calculation(self, synthetic_snn_data):
        """Synaptic operations should scale with fan-out."""
        spike_trains, weights, dt_snn = synthetic_snn_data

        # Weights are 10x10 dense -> fan-out = 10
        metrics = compute_neuromorphic_metrics_from_spikes(
            spike_trains, weights, dt_snn
        )

        total_spikes = 100
        fan_out = 10
        assert metrics.total_syops == total_spikes * fan_out

    def test_sparse_weights(self):
        """Connection sparsity should reflect zero weights."""
        num_neurons = 10
        num_timesteps = 10
        spike_trains = np.zeros((num_neurons, num_timesteps))
        dt_snn = 1e-3

        # 50% sparse weights
        weights = np.zeros((num_neurons, num_neurons))
        weights[:5, :] = 1

        metrics = compute_neuromorphic_metrics_from_spikes(
            spike_trains, weights, dt_snn
        )

        assert metrics.connection_sparsity == 0.5
