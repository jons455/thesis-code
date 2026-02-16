"""Configuration dataclasses for PMSM physics."""

from __future__ import annotations

from dataclasses import dataclass

from embark.utils.config import DEFAULT_PMSM


@dataclass
class PMSMConfig:
    """PMSM motor and simulation configuration."""

    p: int = DEFAULT_PMSM.p
    r_s: float = DEFAULT_PMSM.r_s
    l_d: float = DEFAULT_PMSM.l_d
    l_q: float = DEFAULT_PMSM.l_q
    psi_p: float = DEFAULT_PMSM.psi_p

    i_max: float = DEFAULT_PMSM.i_max
    u_max: float = DEFAULT_PMSM.u_max
    omega_max: float = DEFAULT_PMSM.omega_max

    # DC bus and PWM parameters
    v_dc: float = DEFAULT_PMSM.v_dc
    pwm_frequency: float = DEFAULT_PMSM.pwm_frequency
    dead_time: float = DEFAULT_PMSM.dead_time
    use_dead_time: bool = DEFAULT_PMSM.use_dead_time

    # Measurement noise
    noise_current_std: float = DEFAULT_PMSM.noise_current_std
    noise_speed_std: float = DEFAULT_PMSM.noise_speed_std

    tau: float = DEFAULT_PMSM.tau

    @property
    def motor_parameter(self) -> dict:
        return dict(  # noqa: C408
            p=self.p,
            r_s=self.r_s,
            l_d=self.l_d,
            l_q=self.l_q,
            psi_p=self.psi_p,
        )

    @property
    def limit_values(self) -> dict:
        return dict(  # noqa: C408
            i=self.i_max,
            u=self.u_max,
            omega=self.omega_max,
        )
