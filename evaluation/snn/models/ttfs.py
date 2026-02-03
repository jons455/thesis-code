import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate

    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    snn = None

from evaluation.snn.utils.output_layers import TTFSOutput
from .config import SNNConfig


class TTFSSNNController(nn.Module):
    """SNN controller with time-to-first-spike (TTFS) output coding."""

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

        self.ttfs_out = TTFSOutput(
            input_size=self.config.hidden_size,
            output_size=self.config.output_size,
            spike_grad=spike_grad,
            time_window=self.config.ttfs_time_window,
            output_range=self.config.output_range,
            beta=self.config.ttfs_beta_output,
            learn_beta=self.config.ttfs_learn_beta,
        )

        # Initialize weights with higher gain to ensure propagation in sparse TTFS regime
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with higher gain for sparse TTFS activity."""
        for layer in self.layers:
            # Boosting gain because inputs are very sparse (1 spike per neuron)
            nn.init.kaiming_uniform_(
                layer.weight, a=0, mode="fan_in", nonlinearity="linear"
            )
            with torch.no_grad():
                layer.weight.data *= 4.0  # Boost weights to encourage spiking
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Output layer
        nn.init.kaiming_uniform_(
            self.ttfs_out.fc.weight, a=0, mode="fan_in", nonlinearity="linear"
        )
        with torch.no_grad():
            self.ttfs_out.fc.weight.data *= 4.0
            if self.ttfs_out.fc.bias is not None:
                nn.init.zeros_(self.ttfs_out.fc.bias)

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = []
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )
        return tuple(states)

    def _encode_ttfs(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs as TTFS latencies (integer timesteps)."""
        x_min, x_max = self.config.input_range
        x_clamped = torch.clamp(x, x_min, x_max)
        if x_max == x_min:
            raise ValueError("input_range must have non-zero span.")

        # Map input range to [0, time_window - 1] (larger value -> earlier spike)
        time_span = self.config.ttfs_time_window - 1
        latencies = (x_max - x_clamped) / (x_max - x_min) * time_span
        latencies = torch.round(latencies).to(torch.long)
        return torch.clamp(latencies, 0, time_span)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        return_spikes: bool = False,
    ):
        batch_size = x.shape[0]
        if state is None:
            state = self.init_state(batch_size, x.device)

        hidden_mems = list(state)
        time_window = self.config.ttfs_time_window
        input_latencies = self._encode_ttfs(x)

        mem_out = torch.zeros(batch_size, self.config.output_size, device=x.device)
        spk_history = []

        layer_spike_sum = [torch.zeros_like(mem) for mem in hidden_mems]
        total_spikes = 0.0

        for t in range(time_window):
            spk = (input_latencies == t).float()
            new_mems = []

            for i, (layer, neuron) in enumerate(
                zip(self.layers, self.neurons, strict=False)
            ):
                cur = layer(spk)
                spk, mem = neuron(cur, hidden_mems[i])
                new_mems.append(mem)
                layer_spike_sum[i] += spk
                total_spikes += spk.sum().item()

            spk_out, mem_out = self.ttfs_out.forward_step(spk, mem_out)
            spk_history.append(spk_out)
            total_spikes += spk_out.sum().item()

            hidden_mems = new_mems

        spk_history_tensor = torch.stack(spk_history, dim=0)
        voltage = self.ttfs_out.decode(spk_history_tensor)

        spike_info = None
        if return_spikes:
            layer_sparsities = []
            for spk_sum in layer_spike_sum:
                spk_mean = spk_sum.mean().item() / time_window
                layer_sparsities.append(1.0 - spk_mean)
            out_sparsity = 1.0 - spk_history_tensor.mean().item()
            spike_info = {
                "spikes": [spk_history_tensor.detach()],
                "spike_counts": [s.sum().item() for s in layer_spike_sum]
                + [spk_history_tensor.sum().item()],
                "total_spikes": total_spikes,
                "layer_sparsities": layer_sparsities + [out_sparsity],
            }

        return voltage, tuple(hidden_mems), spike_info

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
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TTFSSNNController":
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

        hidden_mems = list(state)
        input_latencies = self._encode_ttfs(x)
        time_window = self.config.ttfs_time_window
        sparsities = {}

        for t in range(time_window):
            spk = (input_latencies == t).float()
            new_mems = []

            for i, (layer, neuron) in enumerate(
                zip(self.layers, self.neurons, strict=False)
            ):
                cur = layer(spk)
                spk, mem = neuron(cur, hidden_mems[i])
                new_mems.append(mem)
                sparsity = 1.0 - spk.mean().item()
                sparsities[f"hidden_{i}"] = sparsity

            hidden_mems = new_mems

        return sparsities

    def get_weight_matrix(self) -> torch.Tensor:
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        weights.append(self.ttfs_out.fc.weight.data.cpu().numpy())

        import numpy as np

        return np.concatenate([w.flatten() for w in weights])

    def get_network_stats(self) -> dict:
        total_neurons = 0
        total_synapses = 0

        for layer in self.layers:
            total_neurons += layer.out_features
            total_synapses += layer.weight.numel()

        total_neurons += self.ttfs_out.fc.out_features
        total_synapses += self.ttfs_out.fc.weight.numel()

        return {
            "num_neurons": total_neurons,
            "num_synapses": total_synapses,
            "num_layers": len(self.layers) + 1,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }
