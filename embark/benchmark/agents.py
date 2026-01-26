"""NeuroBench-compatible controller agents for PMSM current control benchmark.

This module provides controller agents that follow the NeuroBench agent
interface for closed-loop control benchmarks.

Available agents:
    - PIControllerAgent: Classical PI controller (baseline)
    - SNNControllerAgent: Spiking neural network controller

All agents implement the NeuroBench agent interface with ``__call__(state)``
returning an action and ``reset()`` for stateful agents.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from embark.utils.config import DEFAULT_PMSM

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
    L_d: float = DEFAULT_PMSM.l_d  # d-axis inductance [H]
    L_q: float = DEFAULT_PMSM.l_q  # q-axis inductance [H]
    R_s: float = DEFAULT_PMSM.r_s  # Stator resistance [Ω]
    psi_pm: float = DEFAULT_PMSM.psi_p  # PM flux linkage [Wb]
    p: int = DEFAULT_PMSM.p  # Pole pairs

    # Limits
    i_max: float = DEFAULT_PMSM.i_max  # Maximum current [A]
    u_max: float = DEFAULT_PMSM.u_max  # Maximum voltage [V]

    # Sampling
    Ts: float = DEFAULT_PMSM.tau  # Control period [s] (10 kHz)

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

    INPUT_SPACE = "physical"
    OUTPUT_SPACE = "physical"

    def __init__(
        self,
        params: PIParameters | None = None,
        decoupling: bool = True,
        anti_windup: bool = True,
        kp_d: float | None = None,
        ki_d: float | None = None,
        kp_q: float | None = None,
        ki_q: float | None = None,
    ):
        self.params = params or PIParameters()
        self.decoupling = decoupling
        self.anti_windup = anti_windup

        # Override gains if provided directly
        self._kp_d = kp_d
        self._ki_d = ki_d
        self._kp_q = kp_q
        self._ki_q = ki_q

        # Integrator states
        self.integral_d = 0.0
        self.integral_q = 0.0

        # Previous errors for derivative (if needed)
        self.prev_e_d = 0.0
        self.prev_e_q = 0.0

        # Omega for decoupling (estimated from environment)
        self.omega_el = 0.0

    @property
    def kp_d(self) -> float:
        """Proportional gain for d-axis."""
        return self._kp_d if self._kp_d is not None else self.params.Kp_d

    @property
    def ki_d(self) -> float:
        """Integral gain for d-axis."""
        return self._ki_d if self._ki_d is not None else self.params.Ki_d

    @property
    def kp_q(self) -> float:
        """Proportional gain for q-axis."""
        return self._kp_q if self._kp_q is not None else self.params.Kp_q

    @property
    def ki_q(self) -> float:
        """Integral gain for q-axis."""
        return self._ki_q if self._ki_q is not None else self.params.Ki_q

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
            Physical state [i_d, i_q, e_d, e_q] from PMSMEnv

        Returns
        -------
        np.ndarray
            Physical voltage command [u_d, u_q] in [V]
        """
        # Handle torch tensor input
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().flatten()

        # Ensure state is a flat numpy array
        state = np.asarray(state).flatten()

        # Extract from physical state
        # state = [i_d, i_q, e_d, e_q]
        i_d = float(state[0])
        i_q = float(state[1])
        e_d = float(state[2])
        e_q = float(state[3])

        # PI control
        # P term
        u_d_p = self.kp_d * e_d
        u_q_p = self.kp_q * e_q

        # I term (with Ts multiplication for discrete integration)
        self.integral_d += e_d * self.params.Ts
        self.integral_q += e_q * self.params.Ts

        u_d_i = self.ki_d * self.integral_d
        u_q_i = self.ki_q * self.integral_q

        # Total PI output
        u_d = u_d_p + u_d_i
        u_q = u_q_p + u_q_i

        # Decoupling compensation
        if self.decoupling:
            # Cross-coupling terms
            u_d_dec = -self.omega_el * self.params.L_q * i_q
            u_q_dec = (
                self.omega_el * self.params.L_d * i_d
                + self.omega_el * self.params.psi_pm
            )

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

        return np.array([u_d, u_q], dtype=np.float32)

    def reset_hooks(self):
        """NeuroBench compatibility: reset any registered hooks."""


# =============================================================================
# PyTorch Wrapper for NeuroBench Compatibility
# =============================================================================


class PIControllerTorchAgent(nn.Module):
    """
    PyTorch wrapper around PI controller for NeuroBench TorchAgent compatibility.

    This wraps the PI controller as a PyTorch module so it can be used
    with NeuroBench's TorchAgent interface for metrics computation.
    """

    def __init__(self, params: PIParameters | None = None):
        super().__init__()
        self.pi_controller = PIControllerAgent(params)

        self.INPUT_SPACE = self.pi_controller.INPUT_SPACE
        self.OUTPUT_SPACE = self.pi_controller.OUTPUT_SPACE

        # Dummy parameter so PyTorch recognizes this as a module
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute control action."""
        # Handle batch dimension
        if state.dim() == 1:
            # Single sample without batch dim
            action = self.pi_controller(state.cpu().numpy())
            return torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        elif state.dim() == 2:
            # Batched input - process each sample
            batch_size = state.shape[0]
            actions = []
            for i in range(batch_size):
                action = self.pi_controller(state[i].cpu().numpy())
                actions.append(action)
            return torch.tensor(np.stack(actions), dtype=torch.float32)
        else:
            raise ValueError(f"Expected 1D or 2D input, got {state.dim()}D")

    def reset(self):
        """Reset controller state."""
        self.pi_controller.reset()


# =============================================================================
# SNN Controller Agent
# =============================================================================


class SNNControllerAgent:
    """
    Spiking Neural Network controller for PMSM current control.

    Uses a trained SimpleSNNController model from evaluation/snn/models.py.
    The SNN uses slow-leak LIF output neurons whose membrane potential
    directly encodes the voltage command (no external integrator needed).

    Following NeuroBench/literature recommendations, this agent supports
    multiple internal SNN timesteps per control step for proper spike
    integration (Option B from literature review).

    Parameters
    ----------
    checkpoint_path : str
        Path to trained model checkpoint (.pt file)
    device : str
        Device for inference ('cpu' or 'cuda')
    track_spikes : bool
        Whether to track spike activity for neuromorphic metrics
    num_inference_steps : int
        Number of SNN timesteps per control step (default 1).
        Higher values allow better spike integration but increase latency.
        Literature recommends 5-20 for control tasks.

    Example
    -------
        agent = SNNControllerAgent("snn/checkpoints/best_model.pt", num_inference_steps=10)
        state, _ = env.reset()
        agent.reset()  # Reset neuron states for new episode
        action = agent(state)  # Returns normalized [u_d, u_q]

        # After episode, get spike statistics
        spike_stats = agent.get_spike_statistics()
    """

    INPUT_SPACE = "normalized"
    OUTPUT_SPACE = "normalized"

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pt",
        device: str = "cpu",
        track_spikes: bool = True,
        num_inference_steps: int = 1,
    ):
        # Import SNN model here to avoid circular imports
        import sys
        from pathlib import Path

        # Add project root to path if needed
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from evaluation.snn.models import SimpleSNNController

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.track_spikes = track_spikes
        self.num_inference_steps = num_inference_steps

        # Load trained model
        self.model = SimpleSNNController.load(checkpoint_path, device=device)
        self.model.eval()

        # SNN state (membrane potentials) - persists across timesteps
        self._snn_state: tuple | None = None

        # Spike tracking for neuromorphic metrics
        self._spike_counts_per_step: list = []  # List of spike counts per timestep
        self._sparsities_per_step: list = []  # List of sparsities per timestep
        self._inference_times: list = []  # Inference latencies
        self._total_spikes: int = 0
        self._total_control_steps: int = 0  # For spikes per control step

        # Network stats (cached)
        self._network_stats = self.model.get_network_stats()

    def reset(self):
        """Reset neuron membrane potentials and spike tracking for new episode."""
        self._snn_state = None
        self._spike_counts_per_step = []
        self._sparsities_per_step = []
        self._inference_times = []
        self._total_spikes = 0
        self._total_control_steps = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Compute SNN control action.

        Runs the SNN for num_inference_steps internal timesteps per control step.
        This allows proper spike integration following NeuroBench recommendations.

        Parameters
        ----------
        state : np.ndarray
            Normalized state [i_d, i_q, e_d, e_q] from PMSMEnv

        Returns
        -------
        np.ndarray
            Normalized voltage command [u_d, u_q] in [-1, 1]
        """
        import time

        # Handle torch tensor input
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        # Ensure shape is [batch, features]
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Forward pass through SNN with timing
        # Run multiple internal timesteps per control step
        t_start = time.perf_counter()
        step_spikes = 0
        step_sparsities = []

        with torch.no_grad():
            for _ in range(self.num_inference_steps):
                voltage, self._snn_state, spike_info = self.model(
                    state_tensor, self._snn_state, return_spikes=self.track_spikes
                )

                # Aggregate spike info across internal steps
                if self.track_spikes and spike_info is not None:
                    step_spikes += spike_info["total_spikes"]
                    step_sparsities.append(spike_info["layer_sparsities"])

        t_end = time.perf_counter()

        # Track spike activity (aggregated per control step)
        if self.track_spikes and spike_info is not None:
            self._spike_counts_per_step.append([step_spikes])
            if step_sparsities:
                # Average sparsity across internal steps
                avg_sparsity = np.mean(step_sparsities, axis=0).tolist()
                self._sparsities_per_step.append(avg_sparsity)
            self._total_spikes += step_spikes
            self._inference_times.append(t_end - t_start)
            self._total_control_steps += 1

        # Convert to numpy and ensure shape
        action = voltage.cpu().numpy().flatten()

        # Clip to valid range (should already be in [-1, 1] due to tanh)
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    def get_info(self) -> dict[str, Any]:
        """Return controller metadata for benchmark reporting.

        Returns
        -------
        dict
            Controller metadata including:
            - name: Controller identifier
            - type: Controller type ('snn')
            - checkpoint: Path to model checkpoint
            - hidden_size: Number of neurons per hidden layer
            - num_layers: Total number of layers
            - parameters: Total trainable parameters
        """
        return {
            "name": "SNN-PI-Imitation",
            "type": "snn",
            "checkpoint": str(self.checkpoint_path),
            "hidden_size": self._network_stats["hidden_size"],
            "num_layers": self._network_stats["num_layers"],
            "parameters": self.model.count_parameters(),
        }

    def get_sparsity(self, state: np.ndarray) -> dict:
        """
        Get activation sparsity for neuromorphic metrics.

        Returns fraction of neurons that did NOT spike (higher = more efficient).
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return self.model.get_sparsity(state_tensor, self._snn_state)

    def get_spike_statistics(self) -> dict:
        """
        Get aggregated spike statistics for neuromorphic metrics.

        Returns
        -------
        dict
            Contains:
            - total_spikes: total spikes across all timesteps
            - spikes_per_control_step: average spikes per control step
            - mean_sparsity: average activation sparsity
            - inference_latency_mean/max/std: timing statistics
            - network_stats: neuron/synapse counts
            - num_inference_steps: internal SNN steps per control step
        """
        if not self._spike_counts_per_step:
            return {"error": "No spike data collected. Enable track_spikes=True"}

        _spike_counts = np.array(self._spike_counts_per_step)  # noqa: F841
        sparsities = (
            np.array(self._sparsities_per_step)
            if self._sparsities_per_step
            else np.array([[0.0]])
        )

        stats = {
            # Spike counts
            "total_spikes": int(self._total_spikes),
            "num_control_steps": self._total_control_steps,
            "num_inference_steps_per_control": self.num_inference_steps,
            "spikes_per_control_step": float(
                self._total_spikes / max(1, self._total_control_steps)
            ),
            # Sparsity
            "mean_sparsity": float(sparsities.mean()) if sparsities.size > 0 else 0.0,
            "sparsity_per_layer": sparsities.mean(axis=0).tolist()
            if sparsities.size > 0
            else [],
            # Network architecture
            **self._network_stats,
        }

        # Timing statistics
        if self._inference_times:
            times = np.array(self._inference_times)
            stats.update(
                {
                    "inference_latency_mean_s": float(times.mean()),
                    "inference_latency_max_s": float(times.max()),
                    "inference_latency_std_s": float(times.std()),
                    "inference_latency_p99_s": float(np.percentile(times, 99)),
                    "control_frequency_hz": float(1.0 / times.mean())
                    if times.mean() > 0
                    else 0.0,
                }
            )

        return stats

    def get_weight_matrix(self) -> np.ndarray:
        """Get weight matrix for neuromorphic metrics calculation."""
        return self.model.get_weight_matrix()

    def reset_hooks(self):
        """NeuroBench compatibility: reset any registered hooks."""
        pass


class SNNControllerTorchAgent(nn.Module):
    """
    PyTorch wrapper around SNN controller for NeuroBench TorchAgent compatibility.

    Parameters
    ----------
    checkpoint_path : str
        Path to trained model checkpoint
    device : str
        Device for inference
    num_inference_steps : int
        Number of internal SNN timesteps per control step
    """

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pt",
        device: str = "cpu",
        num_inference_steps: int = 1,
    ):
        super().__init__()
        self.snn_controller = SNNControllerAgent(
            checkpoint_path,
            device,
            track_spikes=True,
            num_inference_steps=num_inference_steps,
        )

        self.INPUT_SPACE = self.snn_controller.INPUT_SPACE
        self.OUTPUT_SPACE = self.snn_controller.OUTPUT_SPACE

        # Dummy parameter so PyTorch recognizes this as a module
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute control action."""
        # Handle batch dimension
        if state.dim() == 1:
            action = self.snn_controller(state.cpu().numpy())
            return torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        elif state.dim() == 2:
            # Batched input - process each sample
            batch_size = state.shape[0]
            actions = []
            for i in range(batch_size):
                action = self.snn_controller(state[i].cpu().numpy())
                actions.append(action)
            return torch.tensor(np.stack(actions), dtype=torch.float32)
        else:
            raise ValueError(f"Expected 1D or 2D input, got {state.dim()}D")

    def reset(self):
        """Reset controller state."""
        self.snn_controller.reset()

    def get_spike_statistics(self) -> dict:
        """Get spike statistics from underlying SNN controller."""
        return self.snn_controller.get_spike_statistics()
