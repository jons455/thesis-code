"""NeuroBench-aligned controller agents for PMSM current control benchmark.

This module provides controller agents that follow the NeuroBench-aligned
interfaces for closed-loop control benchmarks.

Available agents:
    - PIControllerAgent: Classical PI controller implementing DictController
    - SNNControllerAgent: Spiking neural network implementing TensorController

All agents follow the protocols defined in embark.benchmark.interfaces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from embark.benchmark.interfaces import (
    ActionDict,
    DictController,
    ReferenceDict,
    StateDict,
    TensorController,
)
from embark.utils.config import DEFAULT_PMSM


# =============================================================================
# PI Controller Parameters (Technical Optimum)
# =============================================================================


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


# =============================================================================
# PI Controller Agent (DictController Protocol)
# =============================================================================


class PIControllerAgent(DictController):
    """Classical PI controller implementing DictController protocol.

    This serves as the baseline controller for benchmarking.
    Implements decoupled PI control with anti-windup and back-EMF compensation.
    """

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
                self.integral_d *= 0.99
                self.integral_q *= 0.99

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


# =============================================================================
# SNN Controller Agent (TensorController Protocol)
# =============================================================================


class SNNControllerAgent(TensorController):
    """Spiking Neural Network controller implementing TensorController protocol.

    Uses a trained SNN model from evaluation.snn.models. Handles normalization
    internally and tracks spike statistics for neuromorphic metrics.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pt",
        device: str = "cpu",
        track_spikes: bool = True,
        num_inference_steps: int = 1,
        i_max: float = DEFAULT_PMSM.i_max,
        u_max: float = DEFAULT_PMSM.u_max,
    ):
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from evaluation.snn.models import load_snn_model

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.track_spikes = track_spikes
        self.num_inference_steps = num_inference_steps
        self.i_max = i_max
        self.u_max = u_max

        self.model = load_snn_model(checkpoint_path, device=device)
        self.model.eval()

        self._snn_state: tuple | None = None
        self._spike_counts_per_step: list = []
        self._layer_spike_counts: np.ndarray | None = None
        self._sparsities_per_step: list = []
        self._inference_times: list = []
        self._total_spikes: int = 0
        self._total_control_steps: int = 0
        self._network_stats = self.model.get_network_stats()

        # Last inference info for metrics
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        """Reset neuron membrane potentials and spike tracking."""
        self._snn_state = None
        self._spike_counts_per_step = []
        self._layer_spike_counts = None
        self._sparsities_per_step = []
        self._inference_times = []
        self._total_spikes = 0
        self._total_control_steps = 0
        self.last_info = None

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {"snn_state": self._snn_state}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        self._snn_state = state.get("snn_state")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute SNN control action from observation tensor.

        Args:
            observation: Normalized observation tensor [i_d, i_q, e_d, e_q, n]

        Returns:
            Normalized action tensor [v_d, v_q] in [-1, 1]
        """
        t_start = time.perf_counter()
        
        # Initialize accumulation variables
        step_spikes = 0
        step_sparsities = []
        voltage_normalized = torch.zeros(
            observation.shape[0], 2, device=observation.device
        )
        spike_info = None  # Will hold last info if available

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            for _ in range(self.num_inference_steps):
                voltage_normalized, self._snn_state, current_spike_info = self.model(
                    observation, self._snn_state, return_spikes=self.track_spikes
                )

                if self.track_spikes and current_spike_info is not None:
                    # Accumulate stats from this step
                    step_spikes += current_spike_info["total_spikes"]
                    step_sparsities.append(current_spike_info["layer_sparsities"])

                    current_layer_counts = np.array(current_spike_info["spike_counts"])
                    if self._layer_spike_counts is None:
                        self._layer_spike_counts = np.zeros_like(current_layer_counts)
                    if self._layer_spike_counts.shape == current_layer_counts.shape:
                        self._layer_spike_counts += current_layer_counts
                    
                    # Keep last valid info structure for updating last_info later
                    spike_info = current_spike_info

        t_end = time.perf_counter()

        # Update statistics if we tracked any spikes (even if last step returned None)
        if self.track_spikes and (step_spikes > 0 or spike_info is not None):
            self._spike_counts_per_step.append([step_spikes])
            if step_sparsities:
                avg_sparsity = np.mean(step_sparsities, axis=0).tolist()
                self._sparsities_per_step.append(avg_sparsity)
            self._total_spikes += step_spikes
            self._inference_times.append(t_end - t_start)
            self._total_control_steps += 1

            self.last_info = {
                "total_spikes": step_spikes,
                "sparsity": np.mean(step_sparsities) if step_sparsities else 0.0,
                "latency_s": t_end - t_start,
            }

        return torch.clamp(voltage_normalized, -1.0, 1.0)

    def get_info(self) -> dict[str, Any]:
        """Return controller metadata for benchmark reporting."""
        return {
            "name": "SNN-PI-Imitation",
            "type": "snn",
            "checkpoint": str(self.checkpoint_path),
            "hidden_size": self._network_stats["hidden_size"],
            "num_layers": self._network_stats["num_layers"],
            "parameters": self.model.count_parameters(),
            "model_class": self.model.__class__.__name__,
        }

    def get_spike_statistics(self) -> dict:
        """Get aggregated spike statistics for neuromorphic metrics."""
        if not self._spike_counts_per_step:
            return {"error": "No spike data collected. Enable track_spikes=True"}

        sparsities = (
            np.array(self._sparsities_per_step)
            if self._sparsities_per_step
            else np.array([[0.0]])
        )

        total_syops = 0
        if self._layer_spike_counts is not None:

            def get_out_features(layer_idx):
                if layer_idx < len(self.model.layers):
                    return self.model.layers[layer_idx].out_features
                if hasattr(self.model, "fc_out"):
                    return self.model.fc_out.out_features
                elif hasattr(self.model, "pop_out") and hasattr(
                    self.model.pop_out, "fc"
                ):
                    return self.model.pop_out.fc.out_features
                elif hasattr(self.model, "ttfs_out") and hasattr(
                    self.model.ttfs_out, "fc"
                ):
                    return self.model.ttfs_out.fc.out_features
                return 0

            num_hidden = len(self.model.layers)
            is_recurrent = "RecurrentSNNController" in self.model.__class__.__name__

            for i, count in enumerate(self._layer_spike_counts):
                if i >= num_hidden:
                    break
                next_layer_width = get_out_features(i + 1)
                total_syops += int(count * next_layer_width)
                if is_recurrent:
                    current_layer_width = get_out_features(i)
                    total_syops += int(count * current_layer_width)

        stats = {
            "total_spikes": int(self._total_spikes),
            "num_control_steps": self._total_control_steps,
            "num_inference_steps_per_control": self.num_inference_steps,
            "spikes_per_control_step": float(
                self._total_spikes / max(1, self._total_control_steps)
            ),
            "mean_sparsity": float(sparsities.mean()) if sparsities.size > 0 else 0.0,
            "sparsity_per_layer": sparsities.mean(axis=0).tolist()
            if sparsities.size > 0
            else [],
            "total_syops": total_syops,
            "syops_per_timestep": float(
                total_syops / max(1, self._total_control_steps)
            ),
            **self._network_stats,
        }

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


# =============================================================================
# Wrapper for SNN as nn.Module (for NeuroBench TorchAgent compatibility)
# =============================================================================


class SNNControllerTorchAgent(nn.Module):
    """PyTorch wrapper around SNN controller for NeuroBench TorchAgent compatibility."""

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pt",
        device: str = "cpu",
        num_inference_steps: int = 1,
        i_max: float = DEFAULT_PMSM.i_max,
        u_max: float = DEFAULT_PMSM.u_max,
    ):
        super().__init__()
        self.snn_controller = SNNControllerAgent(
            checkpoint_path,
            device,
            track_spikes=True,
            num_inference_steps=num_inference_steps,
            i_max=i_max,
            u_max=u_max,
        )
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute control action."""
        return self.snn_controller.forward(state)

    def reset(self):
        """Reset controller state."""
        self.snn_controller.reset()

    def get_spike_statistics(self) -> dict:
        """Get spike statistics from underlying SNN controller."""
        return self.snn_controller.get_spike_statistics()
