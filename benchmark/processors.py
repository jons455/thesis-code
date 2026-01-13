"""
Pre/Post-processors for PMSM Control Benchmark
===============================================

Processors for spike encoding/decoding and signal transformation
for neuromorphic controllers.

Note: These will be expanded when implementing the SNN controller.
For the PI baseline, no processors are needed.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def normalize_state(
    state: np.ndarray,
    i_max: float = 10.8,
    u_max: float = 48.0,
) -> np.ndarray:
    """
    Normalize PMSM state for neural network input.
    
    Parameters
    ----------
    state : np.ndarray
        Raw state vector [i_d, i_q, e_d, e_q] or similar
    i_max : float
        Maximum current for normalization [A]
    u_max : float
        Maximum voltage for normalization [V]
        
    Returns
    -------
    np.ndarray
        Normalized state in range [-1, 1] or [0, 1]
    """
    # Simple normalization by limits
    # Currents and errors normalized by i_max
    normalized = state / i_max
    return np.clip(normalized, -1.0, 1.0)


def denormalize_action(
    action: np.ndarray,
    u_max: float = 48.0,
) -> np.ndarray:
    """
    Denormalize neural network output to voltage commands.
    
    Parameters
    ----------
    action : np.ndarray
        Normalized action in range [-1, 1]
    u_max : float
        Maximum voltage [V]
        
    Returns
    -------
    np.ndarray
        Voltage command [u_d, u_q] in physical units
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
    """
    Rate coding: Convert continuous value to spike probability.
    
    The value is mapped to a firing rate, and spikes are generated
    stochastically based on that rate.
    
    Parameters
    ----------
    value : float
        Continuous input value
    min_val, max_val : float
        Range of input values
    num_neurons : int
        Number of encoding neurons (population coding)
    max_rate : float
        Maximum firing rate [Hz]
    dt : float
        Simulation timestep [s]
        
    Returns
    -------
    np.ndarray
        Binary spike vector of shape (num_neurons,)
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
    """
    Decode spike train back to continuous value.
    
    Simple mean-rate decoding from population activity.
    
    Parameters
    ----------
    spikes : np.ndarray
        Spike counts or rates from decoding neurons
    min_val, max_val : float
        Range of output values
        
    Returns
    -------
    float
        Decoded continuous value
    """
    # Mean activity normalized to output range
    mean_activity = np.mean(spikes)
    value = min_val + mean_activity * (max_val - min_val)
    return value

