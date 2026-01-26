"""Output-layer building blocks for SNN controllers.

These layers encapsulate different continuous-output decoding strategies
so model architectures can stay focused on core dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import snntorch as snn
except ImportError:  # pragma: no cover - handled by caller
    snn = None


class PopulationCodingOutput(nn.Module):
    """Population coding output layer.

    Multiple neurons per output dimension, each tuned to a preferred value.
    Decoding uses a weighted average of preferred values.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        neurons_per_output: int,
        value_range: tuple[float, float],
        spike_grad,
        beta: float = 0.9,
    ):
        super().__init__()
        self.output_size = output_size
        self.neurons_per_output = neurons_per_output

        total_neurons = output_size * neurons_per_output
        self.fc = nn.Linear(input_size, total_neurons)

        values = torch.linspace(value_range[0], value_range[1], neurons_per_output)
        self.register_buffer("preferred_values", values)

        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        if mem is None:
            mem = torch.zeros(
                batch_size, self.output_size * self.neurons_per_output, device=device
            )

        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)

        spk_reshaped = spk.view(batch_size, self.output_size, self.neurons_per_output)
        spk_sum = spk_reshaped.sum(dim=-1, keepdim=True) + 1e-8
        weights = spk_reshaped / spk_sum

        output = (weights * self.preferred_values).sum(dim=-1)
        return output, mem, spk


class LearnedLinearOutput(nn.Module):
    """Learned linear decoding from output spikes.

    Produces a dense, trainable mapping from spikes to continuous outputs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        neurons_per_output: int,
        spike_grad,
        beta: float = 0.9,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.neurons_per_output = neurons_per_output

        total_neurons = output_size * neurons_per_output
        self.spike_fc = nn.Linear(input_size, total_neurons)
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.decoder = nn.Linear(total_neurons, output_size, bias=True)
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        if mem is None:
            mem = torch.zeros(
                batch_size, self.output_size * self.neurons_per_output, device=device
            )

        cur = self.spike_fc(x)
        spk, mem = self.lif(cur, mem)

        output = self.decoder(spk) * self.output_scale
        return output, mem, spk


class DeltaCodingOutput(nn.Module):
    """Delta coding output layer (up/down spikes per axis)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        spike_grad,
        beta: float = 0.8,
    ):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size * 2)
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        if mem is None:
            mem = torch.zeros(batch_size, self.output_size * 2, device=device)

        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem
