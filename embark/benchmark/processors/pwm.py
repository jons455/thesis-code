"""
PWM (Pulse Width Modulation) converter for motor control.

Converts dq-frame voltage commands to PWM duty cycles and reconstructs
effective voltages that account for inverter non-idealities (DC bus
limitation, dead-time distortion).

Adding this stage between the neural-network output and the physics
engine makes the simulation match real hardware more closely, where
a 2-level voltage-source inverter converts duty cycles to switched
voltages.

References
----------
- Brainchip Akida IP: https://brainchip.com/ip/
- SVM PWM fundamentals: Bose, *Modern Power Electronics and AC Drives*, Ch. 5

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PWMConverter:
    """
    Space-vector-style PWM converter for dq-frame voltages.

    Converts continuous voltage commands ``v_d``, ``v_q`` (in volts) to
    duty cycles in ``[0, 1]`` and reconstructs the effective motor-terminal
    voltages.  Optional dead-time compensation is applied as a
    current-direction-dependent voltage error.

    Parameters
    ----------
    v_dc : float
        DC bus voltage [V].
    pwm_frequency : float
        Switching frequency [Hz].  Used for dead-time distortion
        calculation.
    dead_time : float
        Inverter dead time [s].  Set to 0.0 to disable dead-time
        compensation.  Typical values are 1–4 µs.

    """

    v_dc: float = 48.0
    pwm_frequency: float = 10_000.0
    dead_time: float = 2.0e-6

    # ---- derived (computed once) -------------------------------------------
    _dead_time_voltage: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.v_dc <= 0:
            raise ValueError(f"v_dc must be positive, got {self.v_dc}")
        # Dead-time voltage distortion per switching period:
        #   ΔV = V_dc · t_dead · f_sw
        self._dead_time_voltage = self.v_dc * self.dead_time * self.pwm_frequency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def voltage_to_duty(self, v: float) -> float:
        """
        Map a single-axis voltage to a duty cycle in [0, 1].

        Uses the standard centre-aligned (SVM-like) mapping::

            duty = (v / v_dc + 1) / 2

        """
        return float(np.clip((v / self.v_dc + 1.0) / 2.0, 0.0, 1.0))

    def duty_to_voltage(self, duty: float) -> float:
        """Reconstruct effective voltage from duty cycle."""
        return (2.0 * duty - 1.0) * self.v_dc

    def dead_time_error(self, current: float) -> float:
        """
        Voltage error introduced by dead time (current-direction-dependent).

        The sign convention is: positive current → negative voltage error.

        """
        if self.dead_time <= 0.0:
            return 0.0
        return -float(np.sign(current)) * self._dead_time_voltage

    def convert_dq(
        self,
        v_d: float,
        v_q: float,
        i_d: float = 0.0,
        i_q: float = 0.0,
    ) -> dict[str, float]:
        """
        Full dq conversion pipeline.

        1. Apply dead-time voltage error (if dead_time > 0).
        2. Convert corrected voltages to duty cycles.
        3. Reconstruct effective voltages from duty cycles.

        Parameters
        ----------
        v_d, v_q : float
            Commanded dq voltages [V].
        i_d, i_q : float
            Measured dq currents [A].  Only used for dead-time
            compensation direction; can be left at 0 if dead time is
            disabled.

        Returns
        -------
        dict with keys:
            ``v_d``  – effective d-axis voltage after PWM [V]
            ``v_q``  – effective q-axis voltage after PWM [V]
            ``duty_d`` – d-axis duty cycle [0, 1]
            ``duty_q`` – q-axis duty cycle [0, 1]

        """
        # Step 1: dead-time distortion
        v_d_corrected = v_d + self.dead_time_error(i_d)
        v_q_corrected = v_q + self.dead_time_error(i_q)

        # Step 2: voltage → duty cycle (clamped to [0, 1])
        duty_d = self.voltage_to_duty(v_d_corrected)
        duty_q = self.voltage_to_duty(v_q_corrected)

        # Step 3: reconstruct effective voltage
        v_d_eff = self.duty_to_voltage(duty_d)
        v_q_eff = self.duty_to_voltage(duty_q)

        return {
            "v_d": v_d_eff,
            "v_q": v_q_eff,
            "duty_d": duty_d,
            "duty_q": duty_q,
        }
