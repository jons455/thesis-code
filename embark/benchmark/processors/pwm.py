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
from typing import Sequence

import numpy as np
import torch

from embark.benchmark.interfaces import ActionDict, ActionProcessor, SystemConfig


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


@dataclass
class PWMActionProcessor(ActionProcessor):
    """
    Scale normalized actions to physical voltages, then apply PWM conversion.

    Extends a linear decode step with a PWMConverter stage that:

    1. Converts commanded voltages to inverter duty cycles.
    2. Applies dead-time compensation (current-direction-dependent).
    3. Reconstructs the effective motor-terminal voltages.

    The *effective* voltages (``v_d``, ``v_q``) are what the physics
    engine sees, making the simulation match real inverter behaviour.
    Duty cycles (``duty_d``, ``duty_q``) are included in the returned
    action dict for logging / hardware deployment.

    Parameters
    ----------
    output_keys : Sequence[str]
        Expected voltage keys, typically ``["v_d", "v_q"]``.
    v_dc : float | None
        DC bus voltage.  If *None*, inferred from ``physics_config.v_dc``
        during :meth:`configure`.
    dead_time : float | None
        Inverter dead time [s].  If *None*, inferred from config.
    pwm_frequency : float | None
        Switching frequency [Hz].  If *None*, inferred from config.
    last_state : dict | None
        Holds the latest physics state so that dead-time compensation
        can read current direction.  Updated automatically by the
        harness's metric-update cycle when the processor is used inside
        a ``TensorControllerAdapter``.

    """

    output_keys: Sequence[str] = ("v_d", "v_q")
    v_dc: float | None = None
    dead_time: float | None = None
    pwm_frequency: float | None = None

    # Internal state
    bounds: dict[str, tuple[float, float]] | None = field(default=None, repr=False)
    _pwm: PWMConverter | None = field(default=None, init=False, repr=False)
    _last_i_d: float = field(default=0.0, init=False, repr=False)
    _last_i_q: float = field(default=0.0, init=False, repr=False)

    def configure(self, physics_config: SystemConfig) -> None:
        """Auto-configure bounds and build the PWM converter from config."""
        u_max = getattr(physics_config, "u_max", 1.0)
        if self.bounds is None:
            self.bounds = {key: (-u_max, u_max) for key in self.output_keys}

        v_dc = self.v_dc or getattr(physics_config, "v_dc", u_max)
        dead_time = (
            self.dead_time
            if self.dead_time is not None
            else getattr(physics_config, "dead_time", 2.0e-6)
        )
        pwm_freq = (
            self.pwm_frequency
            if self.pwm_frequency is not None
            else getattr(physics_config, "pwm_frequency", 10_000.0)
        )
        self._pwm = PWMConverter(
            v_dc=v_dc,
            pwm_frequency=pwm_freq,
            dead_time=dead_time,
        )

    def reset(self) -> None:
        """Reset current measurements."""
        self._last_i_d = 0.0
        self._last_i_q = 0.0

    def set_currents(self, i_d: float, i_q: float) -> None:
        """Feed latest current measurements for dead-time direction."""
        self._last_i_d = i_d
        self._last_i_q = i_q

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:
        action_list = action.detach().cpu().flatten().tolist()
        if len(action_list) < len(self.output_keys):
            raise ValueError("Action tensor smaller than number of output keys.")

        if self.bounds is None or self._pwm is None:
            self.configure(physics_config)

        # Step 1: Decode normalised tensor → physical voltages (linear scaling)
        voltages: dict[str, float] = {}
        for idx, key in enumerate(self.output_keys):
            bounds = self.bounds.get(key, (-1.0, 1.0)) if self.bounds else (-1.0, 1.0)
            low, high = bounds
            voltages[key] = float(low + (action_list[idx] + 1) * (high - low) / 2)

        # Step 2: Apply PWM conversion
        assert self._pwm is not None  # guaranteed by configure
        pwm_result = self._pwm.convert_dq(
            v_d=voltages.get("v_d", 0.0),
            v_q=voltages.get("v_q", 0.0),
            i_d=self._last_i_d,
            i_q=self._last_i_q,
        )

        # Return effective voltages (overwrite raw) + duty cycles
        output: ActionDict = {
            "v_d": pwm_result["v_d"],
            "v_q": pwm_result["v_q"],
            "duty_d": pwm_result["duty_d"],
            "duty_q": pwm_result["duty_q"],
        }
        return output
