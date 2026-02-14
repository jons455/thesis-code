"""Action decoding processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from embark.benchmark.interfaces import ActionDict, ActionProcessor, SystemConfig
from embark.benchmark.processors.pwm import PWMConverter


@dataclass
class LinearActionProcessor(ActionProcessor):
    """Scale normalized action tensor to physical units."""

    output_keys: Sequence[str]
    bounds: dict[str, tuple[float, float]] | None = None

    def configure(self, physics_config: SystemConfig) -> None:
        """Auto-configure bounds from physics config limits."""
        if self.bounds is None:
            # Default assumption: symmetric voltage limits [-u_max, u_max]
            # This covers typical PMSM control (v_d, v_q)
            u_max = getattr(physics_config, "u_max", 1.0)
            self.bounds = {key: (-u_max, u_max) for key in self.output_keys}

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:  # noqa: ARG002
        action = action.detach().cpu().flatten().tolist()
        if len(action) < len(self.output_keys):
            raise ValueError("Action tensor smaller than number of output keys.")

        if self.bounds is None:
            # Fallback if configure wasn't called (shouldn't happen in harness)
            self.configure(physics_config)

        output: ActionDict = {}
        for idx, key in enumerate(self.output_keys):
            # Ensure bounds exist (configure guarantees it, but type checker might complain)
            bounds = self.bounds.get(key, (-1.0, 1.0)) if self.bounds else (-1.0, 1.0)
            low, high = bounds
            output[key] = float(low + (action[idx] + 1) * (high - low) / 2)
        return output


@dataclass
class PWMActionProcessor(ActionProcessor):
    """
    Scale normalized actions to physical voltages, then apply PWM conversion.

    Extends the ``LinearActionProcessor`` decode step with a
    ``PWMConverter`` stage that:

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

        # Step 1: Decode normalised tensor → physical voltages (same as Linear)
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
