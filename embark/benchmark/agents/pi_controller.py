"""
PI controller agent for PMSM current control benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from embark.benchmark.interfaces import ActionDict, DictController, ReferenceDict, StateDict
from embark.benchmark.utils.validation import validate_numeric_dict
from embark.utils.config import DEFAULT_PMSM

DEFAULT_ANTI_WINDUP_DECAY = 0.99


@dataclass
class PIParameters:
    """PI controller parameters using Technical Optimum tuning."""

    L_d: float = DEFAULT_PMSM.l_d
    L_q: float = DEFAULT_PMSM.l_q
    R_s: float = DEFAULT_PMSM.r_s
    psi_pm: float = DEFAULT_PMSM.psi_p
    p: int = DEFAULT_PMSM.p
    i_max: float = DEFAULT_PMSM.i_max
    u_max: float = DEFAULT_PMSM.u_max
    Ts: float = DEFAULT_PMSM.tau

    @property
    def Kp_d(self) -> float:
        return self.L_d / (2 * self.Ts)

    @property
    def Ki_d(self) -> float:
        return self.R_s / (2 * self.Ts)

    @property
    def Kp_q(self) -> float:
        return self.L_q / (2 * self.Ts)

    @property
    def Ki_q(self) -> float:
        return self.R_s / (2 * self.Ts)


class PIControllerAgent(DictController):
    """
    Classical PI controller implementing DictController protocol.

    This serves as the baseline controller for benchmarking. Implements decoupled PI
    control with anti-windup and back-EMF compensation.
    """

    def __init__(
        self,
        params: PIParameters | None = None,
        decoupling: bool = True,
        anti_windup: bool = True,
        anti_windup_decay: float = DEFAULT_ANTI_WINDUP_DECAY,
        kp_d: float | None = None,
        ki_d: float | None = None,
        kp_q: float | None = None,
        ki_q: float | None = None,
    ):
        self.params = params or PIParameters()
        self.decoupling = decoupling
        self.anti_windup = anti_windup
        if not (0.0 < anti_windup_decay <= 1.0):
            raise ValueError("anti_windup_decay must be in (0.0, 1.0].")
        self.anti_windup_decay = anti_windup_decay

        self._kp_d = kp_d
        self._ki_d = ki_d
        self._kp_q = kp_q
        self._ki_q = ki_q

        self.integral_d = 0.0
        self.integral_q = 0.0
        self.prev_e_d = 0.0
        self.prev_e_q = 0.0

    @property
    def kp_d(self) -> float:
        return self._kp_d if self._kp_d is not None else self.params.Kp_d

    @property
    def ki_d(self) -> float:
        return self._ki_d if self._ki_d is not None else self.params.Ki_d

    @property
    def kp_q(self) -> float:
        return self._kp_q if self._kp_q is not None else self.params.Kp_q

    @property
    def ki_q(self) -> float:
        return self._ki_q if self._ki_q is not None else self.params.Ki_q

    def reset(self) -> None:
        """Reset integrator states."""
        self.integral_d = 0.0
        self.integral_q = 0.0
        self.prev_e_d = 0.0
        self.prev_e_q = 0.0

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {
            "integral_d": self.integral_d,
            "integral_q": self.integral_q,
            "prev_e_d": self.prev_e_d,
            "prev_e_q": self.prev_e_q,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        self.integral_d = state.get("integral_d", 0.0)
        self.integral_q = state.get("integral_q", 0.0)
        self.prev_e_d = state.get("prev_e_d", 0.0)
        self.prev_e_q = state.get("prev_e_q", 0.0)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        """Compute PI control action from state and reference dicts."""
        validate_numeric_dict(state, "state", required_keys=("i_d", "i_q"))
        validate_numeric_dict(
            reference, "reference", required_keys=("i_d_ref", "i_q_ref")
        )
        if "omega" in state:
            validate_numeric_dict(state, "state", required_keys=("omega",))

        i_d = state["i_d"]
        i_q = state["i_q"]
        i_d_ref = reference["i_d_ref"]
        i_q_ref = reference["i_q_ref"]

        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # P term
        u_d_p = self.kp_d * e_d
        u_q_p = self.kp_q * e_q

        # I term
        self.integral_d += e_d * self.params.Ts
        self.integral_q += e_q * self.params.Ts
        u_d_i = self.ki_d * self.integral_d
        u_q_i = self.ki_q * self.integral_q

        u_d = u_d_p + u_d_i
        u_q = u_q_p + u_q_i

        # Decoupling
        if self.decoupling and "omega" in state:
            omega_el = state["omega"] * self.params.p
            u_d += -omega_el * self.params.L_q * i_q
            u_q += omega_el * self.params.L_d * i_d + omega_el * self.params.psi_pm

        # Voltage limiting
        u_mag = float(np.sqrt(u_d**2 + u_q**2))
        u_limit = self.params.u_max * 0.95

        if u_mag > u_limit:
            scale = u_limit / u_mag
            u_d *= scale
            u_q *= scale
            if self.anti_windup:
                # Bleed off integrator state when actuator saturation occurs.
                self.integral_d *= self.anti_windup_decay
                self.integral_q *= self.anti_windup_decay

        return {"v_d": float(u_d), "v_q": float(u_q)}

    @classmethod
    def from_system_config(
        cls, config, tuning: str = "technical_optimum"
    ) -> "PIControllerAgent":
        """Factory method for auto-tuning from system config."""
        params = PIParameters(
            L_d=getattr(config, "l_d", DEFAULT_PMSM.l_d),
            L_q=getattr(config, "l_q", DEFAULT_PMSM.l_q),
            R_s=getattr(config, "r_s", DEFAULT_PMSM.r_s),
            psi_pm=getattr(config, "psi_p", DEFAULT_PMSM.psi_p),
            p=getattr(config, "p", DEFAULT_PMSM.p),
            u_max=config.u_max,
            Ts=config.tau,
        )
        return cls(params=params)
