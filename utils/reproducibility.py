"""
Reproducibility Utilities
=========================

Functions and classes for ensuring reproducible experiments.

This module provides:
- Seed management for NumPy, PyTorch, and Python's random module
- Experiment configuration dataclass for tracking hyperparameters
- File checksum verification for data integrity

Usage:
------
    from utils import set_seed, ExperimentConfig
    
    # Set all random seeds
    set_seed(42)
    
    # Create experiment configuration
    config = ExperimentConfig(
        seed=42,
        model_name="snn_lif_64",
        description="Baseline SNN with 64 hidden neurons"
    )
    config.save("experiments/exp_001.yaml")

References:
-----------
[1] PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
[2] NumPy Random Generator: https://numpy.org/doc/stable/reference/random/generator.html
"""

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Try to import torch - it's optional for some utilities
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch (if available)
    
    Parameters
    ----------
    seed : int
        The seed value to use. Default is 42.
    deterministic : bool
        If True and PyTorch is available, enables deterministic algorithms.
        This may impact performance. Default is True.
        
    Example
    -------
        >>> set_seed(42)
        >>> np.random.rand()  # Will always produce the same value
        0.3745401188473625
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # PyTorch 1.8+ deterministic algorithms
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    # Some operations don't have deterministic implementations
                    pass


def get_random_state() -> dict[str, Any]:
    """
    Capture the current random state of all random number generators.
    
    Returns
    -------
    dict
        Dictionary containing the random states that can be restored later.
        
    Example
    -------
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    
    if TORCH_AVAILABLE:
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    """
    Restore random state from a previously captured state.
    
    Parameters
    ----------
    state : dict
        State dictionary from get_random_state().
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    
    if TORCH_AVAILABLE and "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])


def compute_file_checksum(filepath: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file for data integrity verification.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the file to checksum.
    algorithm : str
        Hash algorithm to use. Default is 'sha256'.
        Options: 'md5', 'sha1', 'sha256', 'sha512'
        
    Returns
    -------
    str
        Hexadecimal checksum string.
        
    Example
    -------
        >>> checksum = compute_file_checksum("data/train.csv")
        >>> print(checksum[:16])  # First 16 chars
        'a1b2c3d4e5f67890'
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    hasher = hashlib.new(algorithm)
    
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


@dataclass
class ExperimentConfig:
    """
    Configuration dataclass for tracking experiment parameters.
    
    This class helps ensure reproducibility by capturing all relevant
    experiment parameters in a structured format that can be saved
    and loaded.
    
    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    model_name : str
        Name/identifier of the model being trained.
    description : str
        Human-readable description of the experiment.
    timestamp : str
        ISO format timestamp when config was created.
    hyperparameters : dict
        Model hyperparameters (learning rate, hidden size, etc.).
    data_config : dict
        Data-related configuration (paths, splits, etc.).
    environment : dict
        Environment information (Python version, GPU, etc.).
        
    Example
    -------
        >>> config = ExperimentConfig(
        ...     seed=42,
        ...     model_name="snn_lif",
        ...     description="LIF network with 64 hidden neurons",
        ...     hyperparameters={
        ...         "hidden_size": 64,
        ...         "learning_rate": 1e-3,
        ...         "beta": 0.9,
        ...     }
        ... )
        >>> config.save("experiments/exp_001.yaml")
    """
    
    seed: int = 42
    model_name: str = "unnamed"
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    data_config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Populate environment info if not provided."""
        if not self.environment:
            self.environment = self._get_environment_info()
    
    def _get_environment_info(self) -> dict[str, Any]:
        """Gather environment information."""
        import platform
        import sys
        
        info: dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        }
        
        if TORCH_AVAILABLE:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
        
        return info
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str | Path) -> None:
        """
        Save configuration to a YAML or JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path. Extension determines format (.yaml/.yml or .json).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if filepath.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(filepath, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback to JSON if yaml not available
                filepath = filepath.with_suffix(".json")
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str | Path) -> "ExperimentConfig":
        """
        Load configuration from a YAML or JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to configuration file.
            
        Returns
        -------
        ExperimentConfig
            Loaded configuration object.
        """
        filepath = Path(filepath)
        
        if filepath.suffix in (".yaml", ".yml"):
            import yaml
            with open(filepath) as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath) as f:
                data = json.load(f)
        
        return cls(**data)
    
    def apply_seed(self) -> None:
        """Apply the seed from this configuration."""
        set_seed(self.seed)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_experiment_dir(
    base_dir: str | Path = "experiments",
    experiment_name: str | None = None,
) -> Path:
    """
    Create a timestamped experiment directory.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory for experiments.
    experiment_name : str, optional
        Name prefix for the experiment directory.
        
    Returns
    -------
    Path
        Path to the created experiment directory.
        
    Example
    -------
        >>> exp_dir = create_experiment_dir("experiments", "snn_training")
        >>> print(exp_dir)
        experiments/snn_training_2026-01-13_143025
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    exp_dir = base_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return exp_dir

