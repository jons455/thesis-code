"""Configuration for Akida-compatible Keras models.

This mirrors the PyTorch SNNConfig but with Akida constraints.
"""

from dataclasses import dataclass, field


@dataclass
class AkidaConfig:
    """Configuration for Akida-compatible PMSM controller.
    
    Akida 1.0 Constraints:
    - Only ReLU activations in hidden layers
    - No recurrent connections (feed-forward only)
    - 4-bit quantized weights and activations
    - Linear output for regression (scaled post-inference)
    
    Attributes:
        input_size: Number of input features [i_d, i_q, e_d, e_q, n]
        hidden_sizes: List of hidden layer sizes
        output_size: Number of outputs [u_d, u_q]
        output_scale: Scale factor for output regression
        use_batch_norm: Whether to use batch normalization (can help quantization)
        dropout_rate: Dropout rate during training (0.0 to disable)
    """
    
    # Architecture
    input_size: int = 5  # [i_d, i_q, e_d, e_q, n]
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    output_size: int = 2  # [u_d, u_q]
    
    # Output scaling (for regression)
    # Akida outputs quantized integers; multiply by this to get float values
    output_scale: float = 1.0
    
    # Regularization
    use_batch_norm: bool = False  # BN can help quantization stability
    dropout_rate: float = 0.0  # Dropout during training
    
    # Quantization settings
    weight_bits: int = 4  # Akida 1.0 uses 4-bit weights
    activation_bits: int = 4  # 4-bit activations
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration for Akida compatibility."""
        if self.weight_bits not in [4, 8]:
            raise ValueError(
                f"Akida 1.0 supports 4 or 8 bit weights, got {self.weight_bits}"
            )
        if len(self.hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer")
    
    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers."""
        return len(self.hidden_sizes)
    
    @property
    def hidden_size(self) -> int:
        """Size of first hidden layer (for compatibility)."""
        return self.hidden_sizes[0]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "output_scale": self.output_scale,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "AkidaConfig":
        """Create from dictionary."""
        return cls(**d)
