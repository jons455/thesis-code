"""Shared configuration constants for PMSM benchmark components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PMSMDefaults:
    """Default motor, limits, and timing parameters."""

    p: int = 3
    r_s: float = 0.543
    l_d: float = 0.00113
    l_q: float = 0.00142
    psi_p: float = 0.0169

    i_max: float = 10.8
    u_max: float = 48.0
    omega_max: float = 314.16  # rad/s (~3000 rpm)

    control_frequency: float = 10000.0  # Hz

    @property
    def tau(self) -> float:
        """Control timestep [s]."""
        return 1.0 / self.control_frequency


DEFAULT_PMSM = PMSMDefaults()

# Shared defaults for simulations/benchmarks
DEFAULT_MAX_STEPS = 2000
DEFAULT_EPISODE_DURATION = 1.0
STANDARD_NUM_STEPS = int(DEFAULT_EPISODE_DURATION * DEFAULT_PMSM.control_frequency)
