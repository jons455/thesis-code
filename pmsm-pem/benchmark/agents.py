"""
Controller Agents for PMSM Benchmark
====================================

NeuroBench-compatible agents for PMSM current control benchmark.

Agents:
-------
- PIControllerAgent: Classical PI controller (baseline)
- SNNControllerAgent: Spiking neural network controller (future)

All agents follow the NeuroBench agent interface:
- __call__(state) -> action
- reset() for stateful agents
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# PI Controller Parameters (Technical Optimum)
# =============================================================================

@dataclass
class PIParameters:
    """
    PI controller parameters using Technical Optimum tuning.
    
    Kp = L / (2 * Ts)
    Ki = R / (2 * Ts)
    
    where Ts is the control sampling period.
    """
    # Motor parameters
    L_d: float = 0.00113     # d-axis inductance [H]
    L_q: float = 0.00142     # q-axis inductance [H]
    R_s: float = 0.543       # Stator resistance [Ω]
    psi_pm: float = 0.0169   # PM flux linkage [Wb]
    p: int = 3               # Pole pairs
    
    # Limits
    i_max: float = 10.8      # Maximum current [A]
    u_max: float = 48.0      # Maximum voltage [V]
    
    # Sampling
    Ts: float = 1e-4         # Control period [s] (10 kHz)
    
    @property
    def Kp_d(self) -> float:
        """Proportional gain for d-axis."""
        return self.L_d / (2 * self.Ts)
    
    @property
    def Ki_d(self) -> float:
        """Integral gain for d-axis."""
        return self.R_s / (2 * self.Ts)
    
    @property
    def Kp_q(self) -> float:
        """Proportional gain for q-axis."""
        return self.L_q / (2 * self.Ts)
    
    @property
    def Ki_q(self) -> float:
        """Integral gain for q-axis."""
        return self.R_s / (2 * self.Ts)


# =============================================================================
# PI Controller Agent
# =============================================================================

class PIControllerAgent:
    """
    Classical PI controller for PMSM current control.
    
    This serves as the baseline controller for benchmarking.
    Implements decoupled PI control with anti-windup and
    back-EMF compensation.
    
    The agent interface matches NeuroBench expectations:
    - __call__(state) returns action
    - reset() clears integrator states
    
    Parameters
    ----------
    params : PIParameters
        Controller tuning parameters
    decoupling : bool
        Enable cross-coupling compensation
    anti_windup : bool
        Enable anti-windup on integrators
        
    Example
    -------
        agent = PIControllerAgent()
        state, _ = env.reset()
        action = agent(state)  # Returns normalized [u_d, u_q]
    """
    
    def __init__(
        self,
        params: Optional[PIParameters] = None,
        decoupling: bool = True,
        anti_windup: bool = True,
    ):
        self.params = params or PIParameters()
        self.decoupling = decoupling
        self.anti_windup = anti_windup
        
        # Integrator states
        self.integral_d = 0.0
        self.integral_q = 0.0
        
        # Previous errors for derivative (if needed)
        self.prev_e_d = 0.0
        self.prev_e_q = 0.0
        
        # Omega for decoupling (estimated from environment)
        self.omega_el = 0.0
        
    def reset(self):
        """Reset integrator states."""
        self.integral_d = 0.0
        self.integral_q = 0.0
        self.prev_e_d = 0.0
        self.prev_e_q = 0.0
        
    def set_omega(self, omega_el: float):
        """Set electrical angular velocity for decoupling."""
        self.omega_el = omega_el
        
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Compute PI control action.
        
        Parameters
        ----------
        state : np.ndarray
            Normalized state [i_d, i_q, e_d, e_q] from PMSMEnv
            - i_d, i_q: normalized currents
            - e_d, e_q: normalized current errors
            
        Returns
        -------
        np.ndarray
            Normalized voltage command [u_d, u_q] in [-1, 1]
        """
        # Handle torch tensor input
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()
        
        # Ensure state is a flat numpy array
        state = np.asarray(state).flatten()
        
        # Extract from normalized state
        # state = [i_d_norm, i_q_norm, e_d_norm, e_q_norm]
        i_d_norm = float(state[0])
        i_q_norm = float(state[1])
        e_d_norm = float(state[2])  # Already (i_d_ref - i_d) / i_max
        e_q_norm = float(state[3])
        
        # Denormalize for PI calculation
        e_d = e_d_norm * self.params.i_max  # [A]
        e_q = e_q_norm * self.params.i_max  # [A]
        i_d = i_d_norm * self.params.i_max  # [A]
        i_q = i_q_norm * self.params.i_max  # [A]
        
        # PI control
        # P term
        u_d_p = self.params.Kp_d * e_d
        u_q_p = self.params.Kp_q * e_q
        
        # I term (with Ts multiplication for discrete integration)
        self.integral_d += e_d * self.params.Ts
        self.integral_q += e_q * self.params.Ts
        
        u_d_i = self.params.Ki_d * self.integral_d
        u_q_i = self.params.Ki_q * self.integral_q
        
        # Total PI output
        u_d = u_d_p + u_d_i
        u_q = u_q_p + u_q_i
        
        # Decoupling compensation
        if self.decoupling:
            # Cross-coupling terms
            u_d_dec = -self.omega_el * self.params.L_q * i_q
            u_q_dec = self.omega_el * self.params.L_d * i_d + self.omega_el * self.params.psi_pm
            
            u_d += u_d_dec
            u_q += u_q_dec
        
        # Voltage limiting
        u_mag = float(np.sqrt(u_d**2 + u_q**2))
        u_limit = self.params.u_max * 0.95  # 95% to have margin
        
        if u_mag > u_limit:
            scale = u_limit / u_mag
            u_d = float(u_d * scale)
            u_q = float(u_q * scale)
            
            # Anti-windup: limit integrator growth
            if self.anti_windup:
                self.integral_d *= 0.99
                self.integral_q *= 0.99
        
        # Normalize output to [-1, 1]
        u_d_norm = np.clip(u_d / self.params.u_max, -1.0, 1.0)
        u_q_norm = np.clip(u_q / self.params.u_max, -1.0, 1.0)
        
        return np.array([u_d_norm, u_q_norm], dtype=np.float32)
    
    def reset_hooks(self):
        """NeuroBench compatibility: reset any registered hooks."""
        pass


# =============================================================================
# PyTorch Wrapper for NeuroBench Compatibility
# =============================================================================

class PIControllerTorchAgent(nn.Module):
    """
    PyTorch wrapper around PI controller for NeuroBench TorchAgent compatibility.
    
    This wraps the PI controller as a PyTorch module so it can be used
    with NeuroBench's TorchAgent interface for metrics computation.
    """
    
    def __init__(self, params: Optional[PIParameters] = None):
        super().__init__()
        self.pi_controller = PIControllerAgent(params)
        
        # Dummy parameter so PyTorch recognizes this as a module
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute control action."""
        # Convert to numpy, compute, convert back
        if state.dim() > 1:
            state = state.squeeze()
        
        action = self.pi_controller(state.cpu().numpy())
        return torch.tensor(action, dtype=torch.float32)
    
    def reset(self):
        """Reset controller state."""
        self.pi_controller.reset()


# =============================================================================
# Placeholder for Future SNN Controller
# =============================================================================

class SNNControllerAgent:
    """
    Placeholder for SNN controller.
    
    This will be implemented in WP3 using snnTorch with LIF neurons.
    The architecture will be:
    - Input: [i_d, i_q, e_d, e_q] (rate-encoded or direct)
    - Hidden: 1-2 layers of LIF neurons
    - Output: [u_d, u_q] (from membrane potentials)
    
    Training: Imitation learning from PI controller trajectories.
    """
    
    def __init__(self):
        raise NotImplementedError(
            "SNNControllerAgent will be implemented in WP3. "
            "Use PIControllerAgent for baseline testing."
        )

