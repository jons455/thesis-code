from dataclasses import dataclass


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
    neurons_per_output: int = (
        50  # Number of neurons per output dimension (for PopulationSNN)
    )
    output_range: tuple[float, float] = (-1.0, 1.0)
    input_range: tuple[float, float] = (-1.0, 1.0)

    # Delta coding specific
    delta_scale: float = 0.01  # Voltage increment per net spike
    delta_beta: float = 0.8  # Output neuron decay for delta spikes

    # Training
    spike_grad: str = "fast_sigmoid"  # Surrogate gradient function
    slope: float = 25.0  # Surrogate gradient slope

    # Output scaling
    output_scale: float = 0.1  # Scale factor for membrane → voltage

    # TTFS coding specific
    ttfs_time_window: int = 20  # Internal steps per control cycle
    ttfs_beta_output: float = 0.9  # Output neuron decay for TTFS
    ttfs_learn_beta: bool = True  # Learnable decay for TTFS output layer
