import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate

    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    snn = None

from .config import SNNConfig


class MembraneSNNController(nn.Module):
    """
    Pure SNN controller with built-in integration.

    The output layer uses slow-leak LIF neurons (high beta) whose membrane potential
    directly encodes the voltage command. This eliminates the need for an external
    integrator.

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
        from pathlib import Path

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
