"""
PMSMEnv: Gymnasium-compatible Wrapper for GEM PMSM Simulation
=============================================================

This module provides the interface layer between gym-electric-motor (GEM)
and the NeuroBench closed-loop benchmark framework.

The wrapper:
- Creates a GEM PMSM current control environment
- Generates current references (step responses, sweeps)
- Normalizes observations for neural network input
- Tracks control quality metrics (time in target, errors)
- Provides standard Gymnasium interface (reset, step)

Example:
--------
    env = PMSMEnv(n_rpm=1000, scenario='step_response')
    state, info = env.reset()
    
    for _ in range(1000):
        action = agent(state)  # u_d, u_q normalized
        state, reward, terminal, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Literal
from enum import Enum

import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad


# =============================================================================
# Motor Parameters (validated against MATLAB/Simulink)
# =============================================================================

@dataclass
class PMSMConfig:
    """PMSM motor and simulation configuration."""
    # Motor parameters
    p: int = 3                    # Pole pairs
    r_s: float = 0.543            # Stator resistance [Ω]
    l_d: float = 0.00113          # d-axis inductance [H]
    l_q: float = 0.00142          # q-axis inductance [H]
    psi_p: float = 0.0169         # PM flux linkage [Wb]
    
    # Limits
    i_max: float = 10.8           # Maximum current [A]
    u_max: float = 48.0           # DC-link voltage [V]
    omega_max: float = 314.16     # Max angular velocity [rad/s] (~3000 rpm)
    
    # Simulation
    tau: float = 1e-4             # Control timestep [s] (10 kHz)
    
    @property
    def motor_parameter(self) -> dict:
        return dict(
            p=self.p,
            r_s=self.r_s,
            l_d=self.l_d,
            l_q=self.l_q,
            psi_p=self.psi_p,
        )
    
    @property
    def limit_values(self) -> dict:
        return dict(
            i=self.i_max,
            u=self.u_max,
            omega=self.omega_max,
        )


class BenchmarkScenario(Enum):
    """Predefined benchmark scenarios for PMSM current control."""
    STEP_RESPONSE = "step_response"
    OPERATING_POINT = "operating_point"
    DISTURBANCE = "disturbance"


# =============================================================================
# Operations Config (for NeuroBench compatibility)
# =============================================================================

@dataclass
class OperationsConfig:
    """Configuration object expected by NeuroBench BenchmarkClosedLoop."""
    time_step: float = 1e-4  # 100 µs


# =============================================================================
# PMSMEnv Gymnasium Wrapper
# =============================================================================

class PMSMEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for GEM PMSM current control.
    
    This environment wraps the gym-electric-motor PMSM simulation
    and provides a standard interface for NeuroBench closed-loop benchmarks.
    
    Observation Space:
    - i_d: d-axis current [normalized]
    - i_q: q-axis current [normalized]
    - e_d: d-axis current error [normalized]
    - e_q: q-axis current error [normalized]
    
    Action Space:
    - u_d: d-axis voltage command [normalized to [-1, 1]]
    - u_q: q-axis voltage command [normalized to [-1, 1]]
    
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
        max_steps: int = 2000,
        settling_threshold: float = 0.02,  # 2% of reference
        config: Optional[PMSMConfig] = None,
    ):
        super().__init__()
        
        self.config = config or PMSMConfig()
        self.n_rpm = n_rpm
        self.i_d_ref = i_d_ref
        self.i_q_ref = i_q_ref
        self.scenario = scenario
        self.step_time = step_time
        self.max_steps = max_steps
        self.settling_threshold = settling_threshold
        
        # NeuroBench compatibility
        self.ops = OperationsConfig(time_step=self.config.tau)
        self.min_time_in_target = 0.01  # 10 ms minimum in target
        
        # State tracking
        self.current_step = 0
        self.time_in_range = 0
        self._episode_data = []
        
        # Create GEM environment
        self._create_gem_env()
        
        # Define spaces (normalized to [-1, 1])
        # Observation: [i_d, i_q, e_d, e_q]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Action: [u_d, u_q] normalized
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
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
            'Cont-CC-PMSM-v0',
            motor=motor,
            load=load,
            tau=self.config.tau,
            visualization=None,
            render_mode=None,  # Disable rendering completely
            constraints=(),  # No constraints to avoid early termination
        )
        
        # Unwrap to access physical system
        env_unwrapped = self.gem_env
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        self._env_unwrapped = env_unwrapped
        
        # Get state indices
        ps = env_unwrapped.physical_system
        state_names = list(ps.state_names)
        self._idx_i_d = state_names.index('i_sd')
        self._idx_i_q = state_names.index('i_sq')
        self._idx_omega = state_names.index('omega')
        self._idx_epsilon = state_names.index('epsilon')
        
        # Get actual limits
        self._limits = {name: ps.limits[i] for i, name in enumerate(state_names)}
        
    def _get_current_reference(self) -> Tuple[float, float]:
        """Get current reference based on scenario and time."""
        t = self.current_step * self.config.tau
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
    
    def _extract_state(self, gem_state: np.ndarray) -> Tuple[float, float]:
        """Extract and denormalize currents from GEM state."""
        i_d = float(gem_state[self._idx_i_d]) * self._limits.get('i_sd', self.config.i_max)
        i_q = float(gem_state[self._idx_i_q]) * self._limits.get('i_sq', self.config.i_max)
        return i_d, i_q
    
    def _normalize_observation(
        self, 
        i_d: float, 
        i_q: float, 
        i_d_ref: float, 
        i_q_ref: float
    ) -> np.ndarray:
        """Create normalized observation vector."""
        # Errors
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q
        
        # Normalize by max current
        obs = np.array([
            i_d / self.config.i_max,
            i_q / self.config.i_max,
            e_d / self.config.i_max,
            e_q / self.config.i_max,
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)
    
    def _action_to_gem(self, action: np.ndarray, gem_state: np.ndarray) -> np.ndarray:
        """
        Convert normalized dq action to GEM's abc action.
        
        GEM expects action in abc coordinates (normalized).
        We need to do inverse Park transform.
        """
        # Ensure action is flat array
        action = np.asarray(action).flatten()
        u_d_norm = float(action[0])
        u_q_norm = float(action[1])
        
        # Ensure gem_state is flat array and get epsilon
        gem_state = np.asarray(gem_state).flatten()
        epsilon = float(gem_state[self._idx_epsilon]) * np.pi
        
        # Inverse Park transform (dq -> αβ)
        c, s = np.cos(epsilon), np.sin(epsilon)
        u_alpha = u_d_norm * c - u_q_norm * s
        u_beta = u_d_norm * s + u_q_norm * c
        
        # Inverse Clarke transform (αβ -> abc)
        u_a = float(u_alpha)
        u_b = float(-0.5 * u_alpha + (np.sqrt(3) / 2) * u_beta)
        u_c = float(-0.5 * u_alpha - (np.sqrt(3) / 2) * u_beta)
        
        return np.array([u_a, u_b, u_c], dtype=np.float32)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
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
        obs = self._normalize_observation(i_d, i_q, i_d_ref, i_q_ref)
        
        info = {
            'i_d': i_d,
            'i_q': i_q,
            'i_d_ref': i_d_ref,
            'i_q_ref': i_q_ref,
            'time': 0.0,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one control step.
        
        Parameters
        ----------
        action : np.ndarray
            Normalized voltage command [u_d, u_q] in [-1, 1]
            
        Returns
        -------
        observation : np.ndarray
            Current state [i_d, i_q, e_d, e_q] normalized
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
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q
        error_magnitude = np.sqrt(e_d**2 + e_q**2)
        
        # Check if in target (within settling threshold)
        ref_magnitude = np.sqrt(i_d_ref**2 + i_q_ref**2)
        threshold = max(self.settling_threshold * ref_magnitude, 0.01)  # At least 10mA
        
        if error_magnitude < threshold:
            self.time_in_range += 1
        
        # Create observation
        observation = self._normalize_observation(i_d, i_q, i_d_ref, i_q_ref)
        
        # Reward: negative normalized error (higher is better)
        reward = -error_magnitude / self.config.i_max
        
        # Update step counter
        self.current_step += 1
        
        # Terminal condition
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Info dict
        step_info = {
            'i_d': i_d,
            'i_q': i_q,
            'i_d_ref': i_d_ref,
            'i_q_ref': i_q_ref,
            'e_d': e_d,
            'e_q': e_q,
            'u_d': action[0] * self.config.u_max,
            'u_q': action[1] * self.config.u_max,
            'time': self.current_step * self.config.tau,
            'time_in_range': self.time_in_range,
        }
        
        self._episode_data.append(step_info)
        
        return observation, reward, terminated, truncated, step_info
    
    def get_episode_data(self) -> list:
        """Return recorded episode data for analysis."""
        return self._episode_data
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'gem_env'):
            self.gem_env.close()


# =============================================================================
# Factory Functions
# =============================================================================

def make_pmsm_env(
    scenario: str = "step_response",
    n_rpm: float = 1000.0,
    i_d_ref: float = 0.0,
    i_q_ref: float = 2.0,
    **kwargs
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
        n_rpm=n_rpm,
        i_d_ref=i_d_ref,
        i_q_ref=i_q_ref,
        scenario=scenario,
        **kwargs
    )

