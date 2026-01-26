"""Pre/post-processors for PMSM control benchmark.

Processors for state normalization, action scaling, and coordinate transforms.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Processor(Protocol):
    """Callable processor interface."""

    def __call__(self, value: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class IdentityProcessor:
    """Pass-through processor."""

    def __call__(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value)


@dataclass(frozen=True)
class StateNormalizationPreprocessor:
    """Normalize [i_d, i_q, e_d, e_q] by i_max."""

    i_max: float

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.size < 4:
            raise ValueError("State must have at least 4 elements.")
        normalized = state.copy()
        normalized[:4] = normalized[:4] / self.i_max
        return np.clip(normalized, -1.0, 1.0)


@dataclass(frozen=True)
class ActionDenormalizer:
    """Convert normalized action to physical voltages."""

    u_max: float

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size < 2:
            raise ValueError("Action must have at least 2 elements.")
        return np.clip(action[:2], -1.0, 1.0) * self.u_max


@dataclass(frozen=True)
class DqToAbcTransformer:
    """Convert dq voltages to GEM-compatible normalized abc action."""

    u_max: float

    def __call__(self, action_dq: np.ndarray, epsilon: float) -> np.ndarray:
        action = np.asarray(action_dq, dtype=np.float32).flatten()
        if action.size < 2:
            raise ValueError("Action must have at least 2 elements.")
        u_d = float(action[0])
        u_q = float(action[1])

        u_d_norm = np.clip(u_d / self.u_max, -1.0, 1.0)
        u_q_norm = np.clip(u_q / self.u_max, -1.0, 1.0)

        c, s = np.cos(epsilon), np.sin(epsilon)
        u_alpha = u_d_norm * c - u_q_norm * s
        u_beta = u_d_norm * s + u_q_norm * c

        u_a = float(u_alpha)
        u_b = float(-0.5 * u_alpha + (np.sqrt(3) / 2) * u_beta)
        u_c = float(-0.5 * u_alpha - (np.sqrt(3) / 2) * u_beta)
        return np.array([u_a, u_b, u_c], dtype=np.float32)


@dataclass(frozen=True)
class GemStateExtractor:
    """Extract physical currents from GEM state."""

    idx_i_d: int
    idx_i_q: int
    limits: dict[str, float]
    i_max: float

    def extract_currents(self, gem_state: np.ndarray) -> tuple[float, float]:
        state = np.asarray(gem_state).flatten()
        i_d = float(state[self.idx_i_d]) * self.limits.get("i_sd", self.i_max)
        i_q = float(state[self.idx_i_q]) * self.limits.get("i_sq", self.i_max)
        return i_d, i_q


@dataclass(frozen=True)
class ProcessorBundle:
    """Pre/post processors for a controller."""

    state_preprocessor: Processor
    action_postprocessor: Processor


def get_default_processors(
    controller: object, i_max: float, u_max: float
) -> ProcessorBundle:
    """Select default processors based on controller space hints."""

    input_space = getattr(controller, "INPUT_SPACE", "normalized")
    output_space = getattr(controller, "OUTPUT_SPACE", "normalized")

    if input_space == "physical":
        state_pre = IdentityProcessor()
    else:
        state_pre = StateNormalizationPreprocessor(i_max=i_max)

    if output_space == "physical":
        action_post = IdentityProcessor()
    else:
        action_post = ActionDenormalizer(u_max=u_max)

    return ProcessorBundle(
        state_preprocessor=state_pre, action_postprocessor=action_post
    )


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
