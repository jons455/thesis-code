"""Gymnasium-compatible wrapper for GEM PMSM simulation.

This module provides the interface layer between gym-electric-motor (GEM)
and the NeuroBench closed-loop benchmark framework. The wrapper creates a
GEM PMSM current control environment, generates current references, and
tracks control quality metrics.

Example:
    Create and run a PMSM current control environment::

        env = PMSMEnv(n_rpm=1000, scenario="step_response")
        state, info = env.reset()

        for _ in range(1000):
            action = agent(state)  # u_d, u_q in V
            state, reward, terminal, truncated, info = env.step(action)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import gym_electric_motor as gem
import gymnasium as gym
import numpy as np
from gym_electric_motor.physical_systems.electric_motors import (
    PermanentMagnetSynchronousMotor,
)
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gymnasium import spaces

from embark.benchmark.processors import DqToAbcTransformer, GemStateExtractor
from embark.utils.config import DEFAULT_MAX_STEPS, DEFAULT_PMSM

# =============================================================================
# Motor Parameters (validated against MATLAB/Simulink)
# =============================================================================


@dataclass
class PMSMConfig:
    """PMSM motor and simulation configuration."""

    # Motor parameters
    p: int = DEFAULT_PMSM.p  # Pole pairs
    r_s: float = DEFAULT_PMSM.r_s  # Stator resistance [Ω]
    l_d: float = DEFAULT_PMSM.l_d  # d-axis inductance [H]
    l_q: float = DEFAULT_PMSM.l_q  # q-axis inductance [H]
    psi_p: float = DEFAULT_PMSM.psi_p  # PM flux linkage [Wb]

    # Limits
    i_max: float = DEFAULT_PMSM.i_max  # Maximum current [A]
    u_max: float = DEFAULT_PMSM.u_max  # DC-link voltage [V]
    omega_max: float = DEFAULT_PMSM.omega_max  # Max angular velocity [rad/s]

    # Simulation
    tau: float = DEFAULT_PMSM.tau  # Control timestep [s] (10 kHz)

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


class BenchmarkScenario(Enum):
    """Predefined benchmark scenarios for PMSM current control."""

    STEP_RESPONSE = "step_response"
    OPERATING_POINT = "operating_point"
    DISTURBANCE = "disturbance"
    NOMINAL = "nominal"
    HIGH_SPEED = "high_speed"
    ROBUSTNESS = "robustness"


# =============================================================================
# Operations Config (for NeuroBench compatibility)
# =============================================================================


@dataclass
class OperationsConfig:
    """Configuration object expected by NeuroBench BenchmarkClosedLoop."""

    time_step: float = DEFAULT_PMSM.tau  # 100 µs


# =============================================================================
# PMSMEnv Gymnasium Wrapper
# =============================================================================


class PMSMEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for GEM PMSM current control.

    This environment wraps the gym-electric-motor PMSM simulation
    and provides a standard interface for NeuroBench closed-loop benchmarks.

    Observation Space:
    - i_d: d-axis current [A]
    - i_q: q-axis current [A]
    - e_d: d-axis current error [A]
    - e_q: q-axis current error [A]

    Action Space:
    - u_d: d-axis voltage command [V]
    - u_q: q-axis voltage command [V]

    Parameters
    ----------
    n_rpm : float
        Fixed mechanical speed [rpm]
    i_d_ref : float
        d-axis current reference [A]
    i_q_ref : float
        q-axis current reference [A]
    scenario : str
        Benchmark scenario type
    max_steps : int
        Maximum steps per episode
    config : PMSMConfig
        Motor and simulation configuration
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_rpm: float = 1000.0,
        i_d_ref: float = 0.0,
        i_q_ref: float = 2.0,
        scenario: str = "step_response",
        step_time: float = 0.0,  # When to apply step (0 = immediate)
        max_steps: int = DEFAULT_MAX_STEPS,
        measurement_noise_std: float = 0.0,  # Std dev [A] for i_d/i_q noise
        settling_threshold: float = 0.02,  # 2% of reference
        config: PMSMConfig | None = None,
    ):
        super().__init__()

        self.config = config or PMSMConfig()
        self.n_rpm = n_rpm
        self.i_d_ref = i_d_ref
        self.i_q_ref = i_q_ref
        self.scenario = scenario
        self.step_time = step_time
        self.max_steps = max_steps
        self.measurement_noise_std = measurement_noise_std
        self.settling_threshold = settling_threshold

        # NeuroBench compatibility
        self.ops = OperationsConfig(time_step=self.config.tau)
        self.min_time_in_target = 0.01  # 10 ms minimum in target

        # State tracking
        self.current_step = 0
        self.time_in_range = 0
        self._episode_data = []
        self._rng = np.random.default_rng()

        # Create GEM environment
        self._create_gem_env()

        # Define spaces (physical units)
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -self.config.i_max,
                    -self.config.i_max,
                    -2 * self.config.i_max,
                    -2 * self.config.i_max,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.config.i_max,
                    self.config.i_max,
                    2 * self.config.i_max,
                    2 * self.config.i_max,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.config.u_max, high=self.config.u_max, shape=(2,), dtype=np.float32
        )

        # State indices are set in _create_gem_env()

    def _create_gem_env(self):
        """Create the underlying GEM environment."""
        omega_fixed = self.n_rpm * 2 * np.pi / 60.0

        motor = PermanentMagnetSynchronousMotor(
            motor_parameter=self.config.motor_parameter,
            limit_values=self.config.limit_values,
        )
        load = ConstantSpeedLoad(omega_fixed=float(omega_fixed))

        self.gem_env = gem.make(
            "Cont-CC-PMSM-v0",
            motor=motor,
            load=load,
            tau=self.config.tau,
            visualization=None,
            render_mode=None,  # Disable rendering completely
            constraints=(),  # No constraints to avoid early termination
        )

        # Unwrap to access physical system
        env_unwrapped = self.gem_env
        while hasattr(env_unwrapped, "env"):
            env_unwrapped = env_unwrapped.env
        self._env_unwrapped = env_unwrapped

        # Get state indices
        ps = env_unwrapped.physical_system
        state_names = list(ps.state_names)
        self._idx_i_d = state_names.index("i_sd")
        self._idx_i_q = state_names.index("i_sq")
        self._idx_omega = state_names.index("omega")
        self._idx_epsilon = state_names.index("epsilon")

        # Get actual limits
        self._limits = {name: ps.limits[i] for i, name in enumerate(state_names)}
        self._init_processors()

    def _init_processors(self):
        """Initialize processors for state extraction and action conversion."""
        self._action_transform = DqToAbcTransformer(u_max=self.config.u_max)
        self._state_extractor = GemStateExtractor(
            idx_i_d=self._idx_i_d,
            idx_i_q=self._idx_i_q,
            limits=self._limits,
            i_max=self.config.i_max,
        )

    def _get_current_reference(self) -> tuple[float, float]:
        """Get current reference based on scenario and time."""
        _t = self.current_step * self.config.tau  # noqa: F841 (kept for future use)
        step_k = int(self.step_time / self.config.tau) if self.step_time > 0 else 0

        if self.scenario == "step_response":
            if self.current_step < step_k:
                return 0.0, 0.0
            else:
                return self.i_d_ref, self.i_q_ref
        else:
            # Default: immediate reference
            return self.i_d_ref, self.i_q_ref

    def _extract_gem_state_from_reset(self, reset_result) -> np.ndarray:
        """
        Extract state from GEM reset result.

        GEM reset returns: ((state_array, reference_array), info)
        """
        obs = reset_result[0]  # (state_array, reference_array)
        state = obs[0]  # state_array
        return np.asarray(state).flatten()

    def _extract_gem_state_from_step(self, obs) -> np.ndarray:
        """
        Extract state from GEM step observation.

        GEM step returns obs as: (state_array, reference_array)
        """
        if isinstance(obs, tuple):
            state = obs[0]  # state_array
        else:
            state = obs
        return np.asarray(state).flatten()

    def _extract_state(self, gem_state: np.ndarray) -> tuple[float, float]:
        """Extract physical currents from GEM state."""
        return self._state_extractor.extract_currents(gem_state)

    def _apply_measurement_noise(
        self, i_d: float, i_q: float
    ) -> tuple[float, float]:
        """Apply optional Gaussian noise to measured currents."""
        if self.measurement_noise_std <= 0:
            return i_d, i_q

        noise = self._rng.normal(0.0, self.measurement_noise_std, size=2)
        return i_d + float(noise[0]), i_q + float(noise[1])

    def _action_to_gem(self, action: np.ndarray, gem_state: np.ndarray) -> np.ndarray:
        """Convert dq voltages [V] to GEM's normalized abc action."""
        gem_state = np.asarray(gem_state).flatten()
        epsilon = float(gem_state[self._idx_epsilon]) * np.pi
        return self._action_transform(action, epsilon)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self.time_in_range = 0
        self._episode_data = []

        # Reset GEM environment
        # GEM returns ((state_array, reference_array), info)
        reset_result = self.gem_env.reset()
        self._gem_state = self._extract_gem_state_from_reset(reset_result)

        # Get initial observation
        i_d, i_q = self._extract_state(self._gem_state)
        i_d_ref, i_q_ref = self._get_current_reference()
        i_d_meas, i_q_meas = self._apply_measurement_noise(i_d, i_q)
        e_d = i_d_ref - i_d_meas
        e_q = i_q_ref - i_q_meas
        obs = np.array([i_d_meas, i_q_meas, e_d, e_q], dtype=np.float32)

        # DEBUG: Check if obs is within space
        if not self.observation_space.contains(obs):
            print(f"DEBUG: Reset obs out of bounds: {obs}")
            print(f"DEBUG: Space low: {self.observation_space.low}")
            print(f"DEBUG: Space high: {self.observation_space.high}")

        info = {
            "i_d": i_d,
            "i_q": i_q,
            "i_d_ref": i_d_ref,
            "i_q_ref": i_q_ref,
            "time": 0.0,
        }
        if self.measurement_noise_std > 0:
            info["i_d_meas"] = i_d_meas
            info["i_q_meas"] = i_q_meas

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one control step.

        Parameters
        ----------
        action : np.ndarray
            Physical voltage command [u_d, u_q] in [V]

        Returns
        -------
        observation : np.ndarray
            Current state [i_d, i_q, e_d, e_q] in physical units
        reward : float
            Control quality reward (negative tracking error)
        terminated : bool
            Episode ended (max steps reached)
        truncated : bool
            Episode truncated (constraint violation)
        info : dict
            Additional information
        """
        # Get current reference
        i_d_ref, i_q_ref = self._get_current_reference()

        # Convert action to GEM format
        action_abc = self._action_to_gem(action, self._gem_state)

        # Step GEM environment
        obs, reward, done, truncated, info = self.gem_env.step(action_abc)

        # Extract state from GEM's observation (step returns (state_array, ref_array))
        self._gem_state = self._extract_gem_state_from_step(obs)

        # If GEM terminated (constraint violation), reset but continue
        if done:
            reset_result = self.gem_env.reset()
            self._gem_state = self._extract_gem_state_from_reset(reset_result)

        # Extract currents
        i_d, i_q = self._extract_state(self._gem_state)

        # Calculate errors
        i_d_meas, i_q_meas = self._apply_measurement_noise(i_d, i_q)
        e_d = i_d_ref - i_d_meas
        e_q = i_q_ref - i_q_meas
        error_magnitude = np.sqrt(e_d**2 + e_q**2)

        # Check if in target (within settling threshold)
        ref_magnitude = np.sqrt(i_d_ref**2 + i_q_ref**2)
        threshold = max(self.settling_threshold * ref_magnitude, 0.01)  # At least 10mA

        if error_magnitude < threshold:
            self.time_in_range += 1

        # Create observation
        observation = np.array([i_d_meas, i_q_meas, e_d, e_q], dtype=np.float32)

        # Reward: negative normalized error (higher is better)
        reward = -error_magnitude / self.config.i_max

        # Update step counter
        self.current_step += 1

        # Terminal condition
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Info dict
        step_info = {
            "i_d": i_d,
            "i_q": i_q,
            "i_d_ref": i_d_ref,
            "i_q_ref": i_q_ref,
            "e_d": e_d,
            "e_q": e_q,
            "u_d": float(action[0]),
            "u_q": float(action[1]),
            "time": self.current_step * self.config.tau,
            "time_in_range": self.time_in_range,
        }
        if self.measurement_noise_std > 0:
            step_info["i_d_meas"] = i_d_meas
            step_info["i_q_meas"] = i_q_meas

        self._episode_data.append(step_info)

        return observation, reward, terminated, truncated, step_info

    def get_episode_data(self) -> list:
        """Return recorded episode data for analysis."""
        return self._episode_data

    def close(self):
        """Clean up resources."""
        if hasattr(self, "gem_env"):
            self.gem_env.close()


# =============================================================================
# Factory Functions
# =============================================================================


def make_pmsm_env(
    scenario: str = "step_response",
    n_rpm: float = 1000.0,
    i_d_ref: float = 0.0,
    i_q_ref: float = 2.0,
    **kwargs,
) -> PMSMEnv:
    """
    Factory function to create PMSM benchmark environment.

    Parameters
    ----------
    scenario : str
        One of: 'step_response', 'operating_point', 'disturbance'
    n_rpm : float
        Mechanical speed [rpm]
    i_d_ref, i_q_ref : float
        Current references [A]

    Returns
    -------
    PMSMEnv
        Configured environment instance
    """
    return PMSMEnv(
        n_rpm=n_rpm, i_d_ref=i_d_ref, i_q_ref=i_q_ref, scenario=scenario, **kwargs
    )
