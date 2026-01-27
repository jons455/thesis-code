import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate

    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    snn = None

from evaluation.snn.utils.output_layers import PopulationCodingOutput
from .config import SNNConfig


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
        from pathlib import Path

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
