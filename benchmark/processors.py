"""Pre/post-processors for PMSM control benchmark.

Processors for spike encoding/decoding and signal transformation
for neuromorphic controllers.

Note:
    These will be expanded when implementing the SNN controller.
    For the PI baseline, no processors are needed.
"""

import numpy as np


def normalize_state(
    state: np.ndarray,
    i_max: float = 10.8,
    u_max: float = 48.0,
) -> np.ndarray:
    """Normalize PMSM state for neural network input.

    Args:
        state: Raw state vector [i_d, i_q, e_d, e_q] or similar.
        i_max: Maximum current for normalization [A].
        u_max: Maximum voltage for normalization [V].

    Returns:
        Normalized state in range [-1, 1] or [0, 1].
    """
    # Simple normalization by limits
    # Currents and errors normalized by i_max
    normalized = state / i_max
    return np.clip(normalized, -1.0, 1.0)


def denormalize_action(
    action: np.ndarray,
    u_max: float = 48.0,
) -> np.ndarray:
    """Denormalize neural network output to voltage commands.

    Args:
        action: Normalized action in range [-1, 1].
        u_max: Maximum voltage [V].

    Returns:
        Voltage command [u_d, u_q] in physical units.
    """
    return action * u_max


# =============================================================================
# Spike Encoding (for future SNN implementation)
# =============================================================================


def rate_encode(
    value: float,
    min_val: float,
    max_val: float,
    num_neurons: int = 10,
    max_rate: float = 100.0,  # Hz
    dt: float = 1e-4,  # 100us timestep
) -> np.ndarray:
    """Rate coding: Convert continuous value to spike probability.

    The value is mapped to a firing rate, and spikes are generated
    stochastically based on that rate.

    Args:
        value: Continuous input value.
        min_val: Minimum of input value range.
        max_val: Maximum of input value range.
        num_neurons: Number of encoding neurons (population coding).
        max_rate: Maximum firing rate [Hz].
        dt: Simulation timestep [s].

    Returns:
        Binary spike vector of shape (num_neurons,).
    """
    # Normalize to [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)

    # Convert to firing rate
    rate = normalized * max_rate

    # Spike probability for this timestep
    prob = rate * dt

    # Generate spikes (population with same rate for now)
    spikes = np.random.random(num_neurons) < prob

    return spikes.astype(np.float32)


def population_decode(
    spikes: np.ndarray,
    min_val: float,
    max_val: float,
) -> float:
    """Decode spike train back to continuous value.

    Simple mean-rate decoding from population activity.

    Args:
        spikes: Spike counts or rates from decoding neurons.
        min_val: Minimum of output value range.
        max_val: Maximum of output value range.

    Returns:
        Decoded continuous value.
    """
    # Mean activity normalized to output range
    mean_activity = np.mean(spikes)
    value = min_val + mean_activity * (max_val - min_val)
    return value
