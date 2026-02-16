"""
SNN controller agents for PMSM current control benchmark.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from embark.benchmark.interfaces import TensorController
from embark.utils.config import DEFAULT_PMSM


class SNNControllerAgent(TensorController):
    """
    Spiking Neural Network controller implementing TensorController protocol.

    Uses a trained SNN model from evaluation.pytorch_snn.models. Handles normalization
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

        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from evaluation.pytorch_snn.models import load_snn_model

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

        # Pre-calculate layer fan-outs for efficient SyOps tracking
        self._layer_fanouts: list[int] = []
        self._recurrent_fanouts: list[int] = []
        self._cache_layer_fanouts()

    def _cache_layer_fanouts(self) -> None:
        """Cache fan-out values for SyOps calculation."""
        self._layer_fanouts = []
        self._recurrent_fanouts = []

        # Check if recurrent
        is_recurrent = "RecurrentSNNController" in self.model.__class__.__name__
        is_learned_linear = (
            "LearnedLinearSNNController" in self.model.__class__.__name__
        )

        def get_out_features(layer_idx):
            if layer_idx < len(self.model.layers):
                return self.model.layers[layer_idx].out_features
            if hasattr(self.model, "fc_out"):
                return self.model.fc_out.out_features
            elif hasattr(self.model, "pop_out") and hasattr(self.model.pop_out, "fc"):
                return self.model.pop_out.fc.out_features
            elif hasattr(self.model, "ttfs_out") and hasattr(self.model.ttfs_out, "fc"):
                return self.model.ttfs_out.fc.out_features
            elif hasattr(self.model, "delta_out") and hasattr(self.model.delta_out, "fc"):
                return self.model.delta_out.fc.out_features
            return 0

        def get_linear_out_input_size():
            """Get input size for LearnedLinearOutput (fanout from last hidden layer)."""
            if hasattr(self.model, "linear_out") and hasattr(
                self.model.linear_out, "spike_fc"
            ):
                return self.model.linear_out.spike_fc.in_features
            return 0

        num_hidden = len(self.model.layers)

        for i in range(num_hidden):
            # Feedforward connections to next layer (or output)
            if i + 1 < num_hidden:
                # Next hidden layer
                next_width = get_out_features(i + 1)
            elif is_learned_linear:
                # Last hidden layer -> LearnedLinearOutput input
                next_width = get_linear_out_input_size()
            else:
                # Last hidden layer -> output layer
                next_width = get_out_features(num_hidden)
            self._layer_fanouts.append(next_width)

            # Recurrent connections (self-feedback)
            if is_recurrent:
                current_width = get_out_features(i)
                self._recurrent_fanouts.append(current_width)
            else:
                self._recurrent_fanouts.append(0)

        # For LearnedLinearSNNController, add fanout for output layer
        # Output layer spikes don't propagate (decoded from membrane), so fanout = 0
        if is_learned_linear:
            self._layer_fanouts.append(0)  # Output layer has no fanout
            self._recurrent_fanouts.append(0)

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
        # Reset EMA state for spike-counter models
        if hasattr(self.model, "reset_buffer"):
            self.model.reset_buffer()
        if hasattr(self.model, "_ema_rate"):
            self.model._ema_rate = None

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {"snn_state": self._snn_state}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        self._snn_state = state.get("snn_state")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute SNN control action from observation tensor.

        Args:
            observation: Normalized observation tensor [i_d, i_q, e_d, e_q, n]

        Returns:
            Normalized action tensor [v_d, v_q] in [-1, 1]
        """
        if not isinstance(observation, torch.Tensor):
            raise TypeError(
                f"observation must be a torch.Tensor, got {type(observation).__name__}."
            )
        if observation.dim() not in (1, 2):
            raise ValueError(
                "observation must be 1D ([features]) or 2D ([batch, features]); "
                f"got shape {tuple(observation.shape)}."
            )
        if observation.numel() == 0:
            raise ValueError("observation must not be empty.")
        if not torch.is_floating_point(observation):
            raise TypeError(
                "observation must be a floating-point tensor for SNN inference."
            )
        if not torch.isfinite(observation).all().item():
            raise ValueError("observation contains NaN or Inf values.")

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

            # Calculate SyOps for this step
            step_syops = 0
            if spike_info and "spike_counts" in spike_info:
                counts = spike_info["spike_counts"]
                for i, count in enumerate(counts):
                    if i < len(self._layer_fanouts):
                        step_syops += count * self._layer_fanouts[i]
                        step_syops += count * self._recurrent_fanouts[i]

            # Extract sparsity from spike_info if available (preferred), otherwise compute from layer_sparsities
            overall_sparsity = 0.0
            if spike_info and "sparsity" in spike_info:
                # Model provides overall sparsity directly
                overall_sparsity = float(spike_info["sparsity"])
            elif step_sparsities:
                # Fallback: compute mean across all layers and inference steps
                overall_sparsity = float(
                    np.mean([s for layer_sparsities in step_sparsities for s in layer_sparsities])
                )
            elif spike_info and "layer_sparsities" in spike_info:
                # Fallback: use mean of layer sparsities from last inference step
                overall_sparsity = float(np.mean(spike_info["layer_sparsities"]))

            self.last_info = {
                "total_spikes": step_spikes,
                "sparsity": overall_sparsity,
                "latency_s": t_end - t_start,
                "syops": step_syops,  # Per-step SyOps for accumulators
                "hidden_size": self._network_stats["hidden_size"],
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
            # Re-calculate total SyOps using cached fanouts
            for i, count in enumerate(self._layer_spike_counts):
                if i < len(self._layer_fanouts):
                    total_syops += int(count * self._layer_fanouts[i])
                    total_syops += int(count * self._recurrent_fanouts[i])

        stats = {
            "total_spikes": int(self._total_spikes),
            "num_control_steps": self._total_control_steps,
            "num_inference_steps_per_control": self.num_inference_steps,
            "spikes_per_control_step": float(
                self._total_spikes / max(1, self._total_control_steps)
            ),
            "mean_sparsity": float(sparsities.mean()) if sparsities.size > 0 else 0.0,
            "sparsity_per_layer": (
                sparsities.mean(axis=0).tolist() if sparsities.size > 0 else []
            ),
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
                    "control_frequency_hz": (
                        float(1.0 / times.mean()) if times.mean() > 0 else 0.0
                    ),
                }
            )

        return stats


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
