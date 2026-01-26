"""SNN controller models for PMSM current control.

This module provides spiking neural network architectures for motor control,
designed for imitation learning from PI controller trajectories, direct
voltage output, and compatibility with neuromorphic hardware (Akida, Loihi).

Available models:
    - MembraneSNNController: Pure SNN with slow-leak output neurons (formerly SimpleSNNController).
      Uses membrane potential readout for continuous voltage output.
    - PopulationSNNController: Population-coded output layer with fixed tuning curves.
    - LearnedLinearSNNController: Population spikes decoded by a learned linear readout.
    - DeltaSNNController: Incremental (up/down) spikes integrated to voltage (Akida-friendly).

Example:
    Create and run an SNN controller::

        # Original membrane-readout model
        model = MembraneSNNController(hidden_size=64)

        # Akida-compatible population model
        model = PopulationSNNController(hidden_size=64, neurons_per_output=50)

        # Single timestep inference
        state = torch.tensor([[i_d, i_q, e_d, e_q]])
        voltage, snn_state = model(state)
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate

    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    snn = None

from evaluation.snn.output_layers import (  # noqa: E402
    DeltaCodingOutput,
    LearnedLinearOutput,
    PopulationCodingOutput,
)


@dataclass
class SNNConfig:
    """Configuration for SNN controller."""

    # Architecture
    input_size: int = 4  # [i_d, i_q, e_d, e_q]
    hidden_size: int = 64  # Neurons per hidden layer
    num_hidden_layers: int = 2  # Number of hidden layers
    output_size: int = 2  # [u_d, u_q]

    # Neuron dynamics
    beta_hidden: float = 0.9  # Decay rate for hidden layers (fast)
    beta_output: float = 0.995  # Decay rate for output layer (slow, acts as integrator)

    # Population coding specific
    neurons_per_output: int = 50  # Number of neurons per output dimension (for PopulationSNN)
    output_range: tuple[float, float] = (-1.0, 1.0)

    # Delta coding specific
    delta_scale: float = 0.01  # Voltage increment per net spike
    delta_beta: float = 0.8  # Output neuron decay for delta spikes

    # Training
    spike_grad: str = "fast_sigmoid"  # Surrogate gradient function
    slope: float = 25.0  # Surrogate gradient slope

    # Output scaling
    output_scale: float = 0.1  # Scale factor for membrane → voltage


class MembraneSNNController(nn.Module):
    """Pure SNN controller with built-in integration.

    The output layer uses slow-leak LIF neurons (high beta) whose membrane
    potential directly encodes the voltage command. This eliminates the need
    for an external integrator.

    Architecture::

        Input [4] → Dense → LIF (β=0.9) → Dense → LIF (β=0.9) → Dense → LIF (β=0.995)
                                                                          ↓
                                                                    Membrane = [u_d, u_q]

    The slow-leak output neurons act as integrators: they accumulate input
    over time, high beta (0.995) means minimal decay, and at steady state
    the membrane holds the required voltage.

    Args:
        config: Model configuration. Uses defaults if not provided.
        hidden_size: Override hidden layer size (convenience parameter).
    """

    def __init__(
        self,
        config: SNNConfig | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()

        if not SNNTORCH_AVAILABLE:
            raise ImportError(
                "snnTorch is required for SNN models. "
                "Install with: pip install snntorch"
            )

        # Configuration
        self.config = config or SNNConfig()
        if hidden_size is not None:
            self.config.hidden_size = hidden_size

        # Surrogate gradient for training
        spike_grad = surrogate.fast_sigmoid(slope=self.config.slope)

        # Build layers
        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        # Input → first hidden
        self.layers.append(nn.Linear(self.config.input_size, self.config.hidden_size))
        self.neurons.append(
            snn.Leaky(
                beta=self.config.beta_hidden,
                spike_grad=spike_grad,
                learn_beta=False,
            )
        )

        # Additional hidden layers
        for _ in range(self.config.num_hidden_layers - 1):
            self.layers.append(
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
            self.neurons.append(
                snn.Leaky(
                    beta=self.config.beta_hidden,
                    spike_grad=spike_grad,
                    learn_beta=False,
                )
            )

        # Output layer - SLOW leak (built-in integration)
        self.fc_out = nn.Linear(self.config.hidden_size, self.config.output_size)
        self.lif_out = snn.Leaky(
            beta=self.config.beta_output,
            spike_grad=spike_grad,
            learn_beta=False,
            reset_mechanism="none",  # Don't reset - accumulate!
        )

        # Output scaling layer (learnable)
        self.output_scale = nn.Parameter(
            torch.tensor(self.config.output_scale), requires_grad=True
        )

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Initialize membrane potentials for all layers."""
        if device is None:
            device = next(self.parameters()).device

        states = []

        # Hidden layer states
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )

        # Output layer state
        states.append(torch.zeros(batch_size, self.config.output_size, device=device))

        return tuple(states)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict | None]:
        """Forward pass for a single timestep."""
        batch_size = x.shape[0]

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, x.device)

        # Unpack state
        *hidden_mems, mem_out = state
        new_mems = []

        # Track spikes if requested
        spike_tensors = [] if return_spikes else None

        # Process through hidden layers
        spk = x
        for i, (layer, neuron) in enumerate(
            zip(self.layers, self.neurons, strict=False)
        ):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            new_mems.append(mem)

            if return_spikes:
                spike_tensors.append(spk.detach())

        # Output layer - read membrane potential
        cur_out = self.fc_out(spk)
        _, mem_out = self.lif_out(cur_out, mem_out)
        new_mems.append(mem_out)

        # Scale and clip output to [-1, 1]
        voltage = torch.tanh(mem_out * self.output_scale)

        # Build spike info dict if requested
        if return_spikes:
            spike_counts = [s.sum().item() for s in spike_tensors]
            spike_info = {
                "spikes": spike_tensors,
                "spike_counts": spike_counts,
                "total_spikes": sum(spike_counts),
                "layer_sparsities": [1.0 - s.mean().item() for s in spike_tensors],
            }
            return voltage, tuple(new_mems), spike_info

        return voltage, tuple(new_mems), None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict | None]:
        """Forward pass for a sequence of timesteps."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        voltages = []
        all_spike_counts = []
        all_sparsities = []

        for t in range(seq_len):
            voltage, state, spike_info = self.forward(x[:, t, :], state, return_spikes)
            voltages.append(voltage)

            if return_spikes and spike_info is not None:
                all_spike_counts.append(spike_info["spike_counts"])
                all_sparsities.append(spike_info["layer_sparsities"])

        # Aggregate spike stats
        aggregated_spike_info = None
        if return_spikes and all_spike_counts:
            import numpy as np

            spike_counts_array = np.array(all_spike_counts)  # [time, layers]
            sparsities_array = np.array(all_sparsities)

            aggregated_spike_info = {
                "total_spikes": int(spike_counts_array.sum()),
                "spikes_per_timestep": spike_counts_array.sum(axis=1).tolist(),
                "spikes_per_layer": spike_counts_array.sum(axis=0).tolist(),
                "mean_sparsity_per_layer": sparsities_array.mean(axis=0).tolist(),
                "overall_sparsity": float(sparsities_array.mean()),
            }

        return torch.stack(voltages, dim=1), state, aggregated_spike_info

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MembraneSNNController":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_sparsity(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> dict:
        """Compute activation sparsity for a given input."""
        batch_size = x.shape[0]

        if state is None:
            state = self.init_state(batch_size, x.device)

        hidden_mems = list(state[:-1])
        sparsities = {}

        spk = x
        for i, (layer, neuron) in enumerate(
            zip(self.layers, self.neurons, strict=False)
        ):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])

            # Sparsity = fraction of neurons that did NOT spike
            sparsity = 1.0 - spk.mean().item()
            sparsities[f"hidden_{i}"] = sparsity

        return sparsities

    def get_weight_matrix(self) -> torch.Tensor:
        """Get concatenated weight matrix for neuromorphic metrics."""
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        weights.append(self.fc_out.weight.data.cpu().numpy())

        # Return flattened for simple analysis
        import numpy as np

        return np.concatenate([w.flatten() for w in weights])

    def get_network_stats(self) -> dict:
        """Get network architecture statistics."""
        total_neurons = 0
        total_synapses = 0

        # Hidden layers
        for layer in self.layers:
            total_neurons += layer.out_features
            total_synapses += layer.weight.numel()

        # Output layer
        total_neurons += self.fc_out.out_features
        total_synapses += self.fc_out.weight.numel()

        return {
            "num_neurons": total_neurons,
            "num_synapses": total_synapses,
            "num_layers": len(self.layers) + 1,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }


class PopulationSNNController(nn.Module):
    """SNN controller with population-coded output (Akida compatible)."""

    def __init__(
        self,
        config: SNNConfig | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()

        if not SNNTORCH_AVAILABLE:
            raise ImportError("snnTorch is required.")

        self.config = config or SNNConfig()
        if hidden_size is not None:
            self.config.hidden_size = hidden_size

        spike_grad = surrogate.fast_sigmoid(slope=self.config.slope)

        # Build hidden layers (same as MembraneSNN)
        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        # Input → first hidden
        self.layers.append(nn.Linear(self.config.input_size, self.config.hidden_size))
        self.neurons.append(
            snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
        )

        # Additional hidden layers
        for _ in range(self.config.num_hidden_layers - 1):
            self.layers.append(
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
            self.neurons.append(
                snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
            )

        # Output layer - Population Coding
        self.pop_out = PopulationCodingOutput(
            input_size=self.config.hidden_size,
            output_size=self.config.output_size,
            neurons_per_output=self.config.neurons_per_output,
            value_range=self.config.output_range,
            spike_grad=spike_grad,
        )

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = []
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )
        # Output state
        states.append(
            torch.zeros(
                batch_size,
                self.config.output_size * self.config.neurons_per_output,
                device=device,
            )
        )
        return tuple(states)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ):
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        *hidden_mems, mem_out = state
        new_mems = []
        spike_tensors = [] if return_spikes else None

        spk = x
        for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            new_mems.append(mem)
            if return_spikes:
                spike_tensors.append(spk.detach())

        # Population output
        voltage, mem_out, spk_out = self.pop_out(spk, mem_out)
        new_mems.append(mem_out)

        if return_spikes:
            spike_tensors.append(spk_out.detach())
            spike_counts = [s.sum().item() for s in spike_tensors]
            spike_info = {
                "spikes": spike_tensors,
                "spike_counts": spike_counts,
                "total_spikes": sum(spike_counts),
                "layer_sparsities": [1.0 - s.mean().item() for s in spike_tensors],
            }
            return voltage, tuple(new_mems), spike_info

        return voltage, tuple(new_mems), None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict | None]:
        """Forward pass for a sequence of timesteps."""
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        voltages = []
        all_spike_counts = []
        all_sparsities = []

        for t in range(seq_len):
            voltage, state, spike_info = self.forward(x[:, t, :], state, return_spikes)
            voltages.append(voltage)

            if return_spikes and spike_info is not None:
                all_spike_counts.append(spike_info["spike_counts"])
                all_sparsities.append(spike_info["layer_sparsities"])

        # Aggregate spike stats
        aggregated_spike_info = None
        if return_spikes and all_spike_counts:
            import numpy as np

            spike_counts_array = np.array(all_spike_counts)  # [time, layers]
            sparsities_array = np.array(all_sparsities)

            aggregated_spike_info = {
                "total_spikes": int(spike_counts_array.sum()),
                "spikes_per_timestep": spike_counts_array.sum(axis=1).tolist(),
                "spikes_per_layer": spike_counts_array.sum(axis=0).tolist(),
                "mean_sparsity_per_layer": sparsities_array.mean(axis=0).tolist(),
                "overall_sparsity": float(sparsities_array.mean()),
            }

        return torch.stack(voltages, dim=1), state, aggregated_spike_info

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PopulationSNNController":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_sparsity(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> dict:
        """Compute activation sparsity for a given input."""
        batch_size = x.shape[0]

        if state is None:
            state = self.init_state(batch_size, x.device)

        hidden_mems = list(state[:-1])
        sparsities = {}

        spk = x
        for i, (layer, neuron) in enumerate(
            zip(self.layers, self.neurons, strict=False)
        ):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])

            # Sparsity = fraction of neurons that did NOT spike
            sparsity = 1.0 - spk.mean().item()
            sparsities[f"hidden_{i}"] = sparsity
        
        # Output sparsity (from population layer)
        # Note: population layer runs LIF inside forward(), so we don't have direct access
        # to its state here unless we re-run logic. 
        # But PopulationSNNController.forward returns spike_info which has sparsities.
        # This method is used for static analysis, so it might be tricky for the output layer
        # without running forward.
        # For now, just return hidden sparsities.
        return sparsities

    def get_weight_matrix(self) -> torch.Tensor:
        """Get concatenated weight matrix for neuromorphic metrics."""
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        weights.append(self.pop_out.fc.weight.data.cpu().numpy())

        # Return flattened for simple analysis
        import numpy as np

        return np.concatenate([w.flatten() for w in weights])

    def get_network_stats(self) -> dict:
        """Get network architecture statistics."""
        total_neurons = 0
        total_synapses = 0

        # Hidden layers
        for layer in self.layers:
            total_neurons += layer.out_features
            total_synapses += layer.weight.numel()

        # Output layer
        total_neurons += self.pop_out.fc.out_features
        total_synapses += self.pop_out.fc.weight.numel()

        return {
            "num_neurons": total_neurons,
            "num_synapses": total_synapses,
            "num_layers": len(self.layers) + 1,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }


class LearnedLinearSNNController(nn.Module):
    """SNN controller with learned linear decoding (Akida compatible)."""

    def __init__(
        self,
        config: SNNConfig | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()

        if not SNNTORCH_AVAILABLE:
            raise ImportError("snnTorch is required.")

        self.config = config or SNNConfig()
        if hidden_size is not None:
            self.config.hidden_size = hidden_size

        spike_grad = surrogate.fast_sigmoid(slope=self.config.slope)

        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        self.layers.append(nn.Linear(self.config.input_size, self.config.hidden_size))
        self.neurons.append(
            snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
        )

        for _ in range(self.config.num_hidden_layers - 1):
            self.layers.append(
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
            self.neurons.append(
                snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
            )

        self.linear_out = LearnedLinearOutput(
            input_size=self.config.hidden_size,
            output_size=self.config.output_size,
            neurons_per_output=self.config.neurons_per_output,
            spike_grad=spike_grad,
            beta=self.config.beta_hidden,
            output_scale=self.config.output_scale,
        )

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = []
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )
        states.append(
            torch.zeros(
                batch_size,
                self.config.output_size * self.config.neurons_per_output,
                device=device,
            )
        )
        return tuple(states)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ):
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        *hidden_mems, mem_out = state
        new_mems = []
        spike_tensors = [] if return_spikes else None

        spk = x
        for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            new_mems.append(mem)
            if return_spikes:
                spike_tensors.append(spk.detach())

        voltage, mem_out, spk_out = self.linear_out(spk, mem_out)
        new_mems.append(mem_out)

        if return_spikes:
            spike_tensors.append(spk_out.detach())
            spike_counts = [s.sum().item() for s in spike_tensors]
            spike_info = {
                "spikes": spike_tensors,
                "spike_counts": spike_counts,
                "total_spikes": sum(spike_counts),
                "layer_sparsities": [1.0 - s.mean().item() for s in spike_tensors],
            }
            return voltage, tuple(new_mems), spike_info

        return voltage, tuple(new_mems), None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict | None]:
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        voltages = []
        all_spike_counts = []
        all_sparsities = []

        for t in range(seq_len):
            voltage, state, spike_info = self.forward(x[:, t, :], state, return_spikes)
            voltages.append(voltage)

            if return_spikes and spike_info is not None:
                all_spike_counts.append(spike_info["spike_counts"])
                all_sparsities.append(spike_info["layer_sparsities"])

        aggregated_spike_info = None
        if return_spikes and all_spike_counts:
            import numpy as np

            spike_counts_array = np.array(all_spike_counts)
            sparsities_array = np.array(all_sparsities)

            aggregated_spike_info = {
                "total_spikes": int(spike_counts_array.sum()),
                "spikes_per_timestep": spike_counts_array.sum(axis=1).tolist(),
                "spikes_per_layer": spike_counts_array.sum(axis=0).tolist(),
                "mean_sparsity_per_layer": sparsities_array.mean(axis=0).tolist(),
                "overall_sparsity": float(sparsities_array.mean()),
            }

        return torch.stack(voltages, dim=1), state, aggregated_spike_info

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LearnedLinearSNNController":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_sparsity(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...] | None = None
    ) -> dict:
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        hidden_mems = list(state[:-1])
        sparsities = {}

        spk = x
        for i, (layer, neuron) in enumerate(
            zip(self.layers, self.neurons, strict=False)
        ):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            sparsity = 1.0 - spk.mean().item()
            sparsities[f"hidden_{i}"] = sparsity

        return sparsities

    def get_weight_matrix(self) -> torch.Tensor:
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        weights.append(self.linear_out.spike_fc.weight.data.cpu().numpy())
        weights.append(self.linear_out.decoder.weight.data.cpu().numpy())

        import numpy as np

        return np.concatenate([w.flatten() for w in weights])

    def get_network_stats(self) -> dict:
        total_neurons = 0
        total_synapses = 0

        for layer in self.layers:
            total_neurons += layer.out_features
            total_synapses += layer.weight.numel()

        total_neurons += self.linear_out.spike_fc.out_features
        total_synapses += self.linear_out.spike_fc.weight.numel()

        return {
            "num_neurons": total_neurons,
            "num_synapses": total_synapses,
            "num_layers": len(self.layers) + 1,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }


class DeltaSNNController(nn.Module):
    """SNN controller with delta (incremental) output coding."""

    def __init__(
        self,
        config: SNNConfig | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()

        if not SNNTORCH_AVAILABLE:
            raise ImportError("snnTorch is required.")

        self.config = config or SNNConfig()
        if hidden_size is not None:
            self.config.hidden_size = hidden_size

        spike_grad = surrogate.fast_sigmoid(slope=self.config.slope)

        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        self.layers.append(nn.Linear(self.config.input_size, self.config.hidden_size))
        self.neurons.append(
            snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
        )

        for _ in range(self.config.num_hidden_layers - 1):
            self.layers.append(
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )
            self.neurons.append(
                snn.Leaky(beta=self.config.beta_hidden, spike_grad=spike_grad)
            )

        self.delta_out = DeltaCodingOutput(
            input_size=self.config.hidden_size,
            output_size=self.config.output_size,
            spike_grad=spike_grad,
            beta=self.config.delta_beta,
        )

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = []
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )
        states.append(torch.zeros(batch_size, self.config.output_size * 2, device=device))
        states.append(torch.zeros(batch_size, self.config.output_size, device=device))
        return tuple(states)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ):
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        *hidden_mems, mem_out, voltage_state = state
        new_mems = []
        spike_tensors = [] if return_spikes else None

        spk = x
        for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            new_mems.append(mem)
            if return_spikes:
                spike_tensors.append(spk.detach())

        spk_out, mem_out = self.delta_out(spk, mem_out)
        new_mems.append(mem_out)

        spk_reshaped = spk_out.view(batch_size, self.config.output_size, 2)
        net_spikes = spk_reshaped[..., 0] - spk_reshaped[..., 1]

        voltage = voltage_state + net_spikes * self.config.delta_scale
        voltage = torch.clamp(voltage, -1.0, 1.0)
        new_mems.append(voltage)

        if return_spikes:
            spike_tensors.append(spk_out.detach())
            spike_counts = [s.sum().item() for s in spike_tensors]
            spike_info = {
                "spikes": spike_tensors,
                "spike_counts": spike_counts,
                "total_spikes": sum(spike_counts),
                "layer_sparsities": [1.0 - s.mean().item() for s in spike_tensors],
            }
            return voltage, tuple(new_mems), spike_info

        return voltage, tuple(new_mems), None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict | None]:
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        voltages = []
        all_spike_counts = []
        all_sparsities = []

        for t in range(seq_len):
            voltage, state, spike_info = self.forward(x[:, t, :], state, return_spikes)
            voltages.append(voltage)

            if return_spikes and spike_info is not None:
                all_spike_counts.append(spike_info["spike_counts"])
                all_sparsities.append(spike_info["layer_sparsities"])

        aggregated_spike_info = None
        if return_spikes and all_spike_counts:
            import numpy as np

            spike_counts_array = np.array(all_spike_counts)
            sparsities_array = np.array(all_sparsities)

            aggregated_spike_info = {
                "total_spikes": int(spike_counts_array.sum()),
                "spikes_per_timestep": spike_counts_array.sum(axis=1).tolist(),
                "spikes_per_layer": spike_counts_array.sum(axis=0).tolist(),
                "mean_sparsity_per_layer": sparsities_array.mean(axis=0).tolist(),
                "overall_sparsity": float(sparsities_array.mean()),
            }

        return torch.stack(voltages, dim=1), state, aggregated_spike_info

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DeltaSNNController":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_sparsity(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...] | None = None
    ) -> dict:
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        hidden_mems = list(state[:-2])
        sparsities = {}

        spk = x
        for i, (layer, neuron) in enumerate(
            zip(self.layers, self.neurons, strict=False)
        ):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            sparsity = 1.0 - spk.mean().item()
            sparsities[f"hidden_{i}"] = sparsity

        return sparsities

    def get_weight_matrix(self) -> torch.Tensor:
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        weights.append(self.delta_out.fc.weight.data.cpu().numpy())

        import numpy as np

        return np.concatenate([w.flatten() for w in weights])

    def get_network_stats(self) -> dict:
        total_neurons = 0
        total_synapses = 0

        for layer in self.layers:
            total_neurons += layer.out_features
            total_synapses += layer.weight.numel()

        total_neurons += self.delta_out.fc.out_features
        total_synapses += self.delta_out.fc.weight.numel()

        return {
            "num_neurons": total_neurons,
            "num_synapses": total_synapses,
            "num_layers": len(self.layers) + 1,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }

def load_snn_model(path: str, device: str = "cpu") -> nn.Module:
    """Load SNN model from checkpoint, automatically detecting type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Detect model type based on keys
    if "delta_out.fc.weight" in state_dict:
        model_cls = DeltaSNNController
    elif "linear_out.decoder.weight" in state_dict:
        model_cls = LearnedLinearSNNController
    elif "pop_out.fc.weight" in state_dict:
        model_cls = PopulationSNNController
    elif "fc_out.weight" in state_dict:
        model_cls = MembraneSNNController
    else:
        # Fallback or unknown
        print("Warning: Unknown model structure, trying MembraneSNNController")
        model_cls = MembraneSNNController

    model = model_cls(config=checkpoint.get("config"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

# Aliases for compatibility
SimpleSNNController = MembraneSNNController
SNN = MembraneSNNController
