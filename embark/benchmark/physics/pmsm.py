"""PMSM physics engine implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gym_electric_motor as gem
import numpy as np
from gym_electric_motor.physical_systems.electric_motors import (
    PermanentMagnetSynchronousMotor,
)
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad

from embark.benchmark.interfaces import ActionDict, StateDict
from embark.benchmark.physics.config import PMSMConfig


@dataclass
class PMSMPhysicsEngine:
    """Physics engine wrapper around GEM's PMSM environment."""

    n_rpm: float = 1000.0
    config: PMSMConfig = field(default_factory=PMSMConfig)

    def __post_init__(self) -> None:
        self._create_gem_env()
        self._time = 0.0
        self._last_gem_state: np.ndarray | None = None

    @property
    def state_keys(self) -> set[str]:
        return {"i_d", "i_q", "omega", "epsilon", "time"}

    @property
    def action_keys(self) -> set[str]:
        return {"v_alpha", "v_beta"}

    def _create_gem_env(self) -> None:
        omega_fixed = self.n_rpm * 2 * np.pi / 60.0
        motor = PermanentMagnetSynchronousMotor(
            motor_parameter=self.config.motor_parameter,
            limit_values=self.config.limit_values,
        )
        load = ConstantSpeedLoad(omega_fixed=float(omega_fixed))

        self._gem_env = gem.make(
            "Cont-CC-PMSM-v0",
            motor=motor,
            load=load,
            tau=self.config.tau,
            visualization=None,
            render_mode=None,
            constraints=(),
        )

        env_unwrapped = self._gem_env
        while hasattr(env_unwrapped, "env"):
            env_unwrapped = env_unwrapped.env
        ps = env_unwrapped.physical_system
        state_names = list(ps.state_names)
        self._idx_i_d = state_names.index("i_sd")
        self._idx_i_q = state_names.index("i_sq")
        self._idx_omega = state_names.index("omega")
        self._idx_epsilon = state_names.index("epsilon")
        self._limits = {name: ps.limits[i] for i, name in enumerate(state_names)}

    def reset(self, seed: int | None = None) -> StateDict:
        if seed is not None:
            self._gem_env.reset(seed=seed)
        reset_result = self._gem_env.reset()
        gem_state = self._extract_gem_state_from_reset(reset_result)
        self._last_gem_state = gem_state
        self._time = 0.0
        return self._state_from_gem(gem_state)

    def step(self, action: ActionDict) -> tuple[StateDict, dict[str, Any]]:
        action_abc = self._action_to_gem(action)
        obs, reward, terminated, truncated, info = self._gem_env.step(action_abc)
        gem_state = self._extract_gem_state_from_step(obs)

        debug_info = dict(info or {})
        debug_info.update(
            {"reward": reward, "terminated": terminated, "truncated": truncated}
        )

        if terminated:
            reset_result = self._gem_env.reset()
            gem_state = self._extract_gem_state_from_reset(reset_result)
            debug_info["gem_reset"] = True

        self._last_gem_state = gem_state
        self._time += self.config.tau
        return self._state_from_gem(gem_state), debug_info

    def close(self) -> None:
        self._gem_env.close()

    def _extract_gem_state_from_reset(self, reset_result) -> np.ndarray:
        obs = reset_result[0]
        state = obs[0]
        return np.asarray(state).flatten()

    def _extract_gem_state_from_step(self, obs) -> np.ndarray:
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        return np.asarray(state).flatten()

    def _state_from_gem(self, gem_state: np.ndarray) -> StateDict:
        i_d = float(gem_state[self._idx_i_d]) * self._limits.get(
            "i_sd", self.config.i_max
        )
        i_q = float(gem_state[self._idx_i_q]) * self._limits.get(
            "i_sq", self.config.i_max
        )
        omega = float(gem_state[self._idx_omega]) * self._limits.get(
            "omega", self.config.omega_max
        )
        epsilon = float(gem_state[self._idx_epsilon]) * np.pi
        return {
            "i_d": i_d,
            "i_q": i_q,
            "omega": omega,
            "epsilon": epsilon,
            "time": self._time,
        }

    def _action_to_gem(self, action: ActionDict) -> np.ndarray:
        if "v_d" in action and "v_q" in action:
            epsilon = float(self._last_epsilon())
            v_alpha, v_beta = self._dq_to_alpha_beta(
                action["v_d"], action["v_q"], epsilon
            )
        elif "v_alpha" in action and "v_beta" in action:
            v_alpha = float(action["v_alpha"])
            v_beta = float(action["v_beta"])
        else:
            raise KeyError("Action must include v_alpha/v_beta or v_d/v_q.")

        v_alpha_norm = np.clip(v_alpha / self.config.u_max, -1.0, 1.0)
        v_beta_norm = np.clip(v_beta / self.config.u_max, -1.0, 1.0)

        u_a = float(v_alpha_norm)
        u_b = float(-0.5 * v_alpha_norm + (np.sqrt(3) / 2) * v_beta_norm)
        u_c = float(-0.5 * v_alpha_norm - (np.sqrt(3) / 2) * v_beta_norm)
        return np.array([u_a, u_b, u_c], dtype=np.float32)

    def _last_epsilon(self) -> float:
        if self._last_gem_state is None:
            return 0.0
        return float(self._last_gem_state[self._idx_epsilon]) * np.pi

    @staticmethod
    def _dq_to_alpha_beta(
        v_d: float, v_q: float, epsilon: float
    ) -> tuple[float, float]:
        c, s = np.cos(epsilon), np.sin(epsilon)
        v_alpha = v_d * c - v_q * s
        v_beta = v_d * s + v_q * c
        return v_alpha, v_beta
