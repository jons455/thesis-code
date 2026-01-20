"""
SNN Controller Models for PMSM Current Control
==============================================

This module provides spiking neural network architectures for motor control.

Models:
- SimpleSNNController: Pure SNN with slow-leak output neurons (primary)
  Uses membrane potential readout for continuous voltage output.

The architecture is designed for:
- Imitation learning from PI controller trajectories
- Direct voltage output (no external integrator needed)
- Compatibility with neuromorphic hardware (Akida, Loihi)

Example:
    model = SimpleSNNController(hidden_size=64)
    
    # Single timestep inference
    state = torch.tensor([[i_d, i_q, e_d, e_q]])
    voltage, snn_state = model(state)
    
    # Continue with state persistence
    voltage, snn_state = model(next_state, snn_state)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    snn = None


@dataclass
class SNNConfig:
    """Configuration for SNN controller."""
    
    # Architecture
    input_size: int = 4          # [i_d, i_q, e_d, e_q]
    hidden_size: int = 64        # Neurons per hidden layer
    num_hidden_layers: int = 2   # Number of hidden layers
    output_size: int = 2         # [u_d, u_q]
    
    # Neuron dynamics
    beta_hidden: float = 0.9     # Decay rate for hidden layers (fast)
    beta_output: float = 0.995   # Decay rate for output layer (slow, acts as integrator)
    
    # Training
    spike_grad: str = "fast_sigmoid"  # Surrogate gradient function
    slope: float = 25.0          # Surrogate gradient slope
    
    # Output scaling
    output_scale: float = 0.1    # Scale factor for membrane → voltage


class SimpleSNNController(nn.Module):
    """
    Pure SNN controller with built-in integration.
    
    The output layer uses slow-leak LIF neurons (high beta) whose membrane
    potential directly encodes the voltage command. This eliminates the need
    for an external integrator.
    
    Architecture:
        Input [4] → Dense → LIF (β=0.9) → Dense → LIF (β=0.9) → Dense → LIF (β=0.995)
                                                                          ↓
                                                                    Membrane = [u_d, u_q]
    
    The slow-leak output neurons act as integrators:
    - They accumulate input over time
    - High beta (0.995) means minimal decay
    - At steady state, membrane holds the required voltage
    
    Parameters
    ----------
    config : SNNConfig, optional
        Model configuration. Uses defaults if not provided.
    hidden_size : int, optional
        Override hidden layer size (convenience parameter)
    
    Example
    -------
        model = SimpleSNNController(hidden_size=64)
        
        # Reset state for new episode
        state = None
        
        for obs in observations:
            voltage, state = model(obs.unsqueeze(0), state)
            # voltage is [batch, 2] tensor with [u_d, u_q]
    """
    
    def __init__(
        self,
        config: Optional[SNNConfig] = None,
        hidden_size: Optional[int] = None,
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
        self.layers.append(
            nn.Linear(self.config.input_size, self.config.hidden_size)
        )
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
            torch.tensor(self.config.output_scale),
            requires_grad=True
        )
    
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Initialize membrane potentials for all layers.
        
        Parameters
        ----------
        batch_size : int
            Number of samples in batch
        device : torch.device, optional
            Device for tensors. Uses model device if not specified.
        
        Returns
        -------
        tuple of tensors
            Initial membrane states for all layers
        """
        if device is None:
            device = next(self.parameters()).device
        
        states = []
        
        # Hidden layer states
        for _ in range(len(self.neurons)):
            states.append(
                torch.zeros(batch_size, self.config.hidden_size, device=device)
            )
        
        # Output layer state
        states.append(
            torch.zeros(batch_size, self.config.output_size, device=device)
        )
        
        return tuple(states)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for a single timestep.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, 4] containing [i_d, i_q, e_d, e_q]
            Values should be normalized to approximately [-1, 1]
        state : tuple of tensors, optional
            Previous membrane states. If None, initializes to zeros.
        
        Returns
        -------
        voltage : torch.Tensor
            Output voltage command [batch, 2] containing [u_d, u_q]
            Normalized to [-1, 1] via tanh
        new_state : tuple of tensors
            Updated membrane states for next timestep
        """
        batch_size = x.shape[0]
        
        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, x.device)
        
        # Unpack state
        *hidden_mems, mem_out = state
        new_mems = []
        
        # Process through hidden layers
        spk = x
        for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            new_mems.append(mem)
        
        # Output layer - read membrane potential
        cur_out = self.fc_out(spk)
        _, mem_out = self.lif_out(cur_out, mem_out)
        new_mems.append(mem_out)
        
        # Scale and clip output to [-1, 1]
        voltage = torch.tanh(mem_out * self.output_scale)
        
        return voltage, tuple(new_mems)
    
    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for a sequence of timesteps.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, time, 4]
        state : tuple of tensors, optional
            Initial membrane states
        
        Returns
        -------
        voltages : torch.Tensor
            Output voltages [batch, time, 2]
        final_state : tuple of tensors
            Final membrane states
        """
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            state = self.init_state(batch_size, x.device)
        
        voltages = []
        for t in range(seq_len):
            voltage, state = self.forward(x[:, t, :], state)
            voltages.append(voltage)
        
        return torch.stack(voltages, dim=1), state
    
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
    def load(cls, path: str, device: str = "cpu") -> "SimpleSNNController":
        """Load model from checkpoint."""
        # PyTorch 2.6+ requires weights_only=False for custom classes
        # or adding them to safe globals
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
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> dict:
        """
        Compute activation sparsity for a given input.
        
        Returns dict with sparsity per layer (fraction of non-spiking neurons).
        """
        batch_size = x.shape[0]
        
        if state is None:
            state = self.init_state(batch_size, x.device)
        
        hidden_mems = list(state[:-1])
        sparsities = {}
        
        spk = x
        for i, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
            
            # Sparsity = fraction of neurons that did NOT spike
            sparsity = 1.0 - spk.mean().item()
            sparsities[f"hidden_{i}"] = sparsity
        
        return sparsities


# Alias for convenience
SNN = SimpleSNNController
