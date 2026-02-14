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

    # DC bus and PWM parameters
    v_dc: float = 48.0  # DC bus voltage [V]
    pwm_frequency: float = 10000.0  # PWM switching frequency [Hz]
    dead_time: float = (
        2.0e-6  # Inverter dead time [s] (typ. 1-4 µs), used only if use_dead_time=True
    )
    use_dead_time: bool = (
        False  # If False, simulation runs without dead time (simplified PMSM); set True to enable
    )

    control_frequency: float = 10000.0  # Hz

    @property
    def tau(self) -> float:
        """Control timestep [s]."""
        return 1.0 / self.control_frequency

    # --- AUTOMATIC CONTROLLER TUNING ---
    # These properties calculate the "Technical Optimum" gains automatically.
    # Formula: Gain = Parameter / (2 * Time_Delay)

    @property
    def kp_d_optimum(self) -> float:
        """Calculates optimal P-gain for d-axis: Kp = Ld / (2 * tau)."""
        return self.l_d / (2 * self.tau)

    @property
    def kp_q_optimum(self) -> float:
        """Calculates optimal P-gain for q-axis: Kp = Lq / (2 * tau)."""
        return self.l_q / (2 * self.tau)

    @property
    def ki_optimum(self) -> float:
        """Calculates theoretical I-gain: Ki = Rs / (2 * tau).
        Note: Real hardware often detunes this for stability (e.g., divide by 20).
        """
        return self.r_s / (2 * self.tau)

    @property
    def ki_stable(self) -> float:
        """Returns the empirically stable I-gain (Detuned)."""
        # We detune the theoretical value significantly to prevent overshoot.
        # Theoretical is ~2715, we return 100.0 as verified in benchmarking.
        detune_factor = 27.0
        return self.ki_optimum / detune_factor


DEFAULT_PMSM = PMSMDefaults()

# Shared defaults for simulations/benchmarks
DEFAULT_MAX_STEPS = 2000
DEFAULT_EPISODE_DURATION = 1.0
STANDARD_NUM_STEPS = int(DEFAULT_EPISODE_DURATION * DEFAULT_PMSM.control_frequency)
