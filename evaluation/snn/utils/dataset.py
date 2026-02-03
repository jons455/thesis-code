"""Dataset for PMSM SNN training.

Loads PI controller trajectories and prepares them for SNN training.
The dataset provides (input, target) pairs where input is [i_d, i_q, e_d, e_q]
normalized state and target is [u_d, u_q] normalized voltage command.

Example:
    Load and iterate over training data::

        dataset = PMSMDataset(
            data_dir="data/raw/train",
            window_size=100,
        )

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for inputs, targets in dataloader:
            # inputs: [batch, time, 4]
            # targets: [batch, time, 2]
            outputs, _ = model.forward_sequence(inputs)
            loss = F.mse_loss(outputs, targets)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from embark.utils.config import DEFAULT_PMSM


@dataclass
class DataConfig:
    """Configuration for data loading and normalization."""

    # Motor limits for normalization
    i_max: float = DEFAULT_PMSM.i_max  # Maximum current [A]
    u_max: float = DEFAULT_PMSM.u_max  # Maximum voltage [V]

    # Data columns (may vary by CSV format)
    current_cols: tuple[str, str] = ("i_sd", "i_sq")  # d-q currents
    voltage_cols: tuple[str, str] = ("u_sd", "u_sq")  # d-q voltages
    reference_cols: tuple[str, str] = ("i_sd_ref", "i_sq_ref")  # Reference currents

    # Alternative column names (for compatibility)
    alt_current_cols: tuple[str, str] = ("i_d", "i_q")
    alt_voltage_cols: tuple[str, str] = ("u_d", "u_q")
    alt_reference_cols: tuple[str, str] = ("i_d_ref", "i_q_ref")

    # Sequence handling
    window_size: int = 100  # Timesteps per training window
    stride: int = 50  # Step between windows (for overlap)

    # Skip initial transient
    skip_initial: int = 10  # Skip first N timesteps (initial conditions)


class PMSMDataset(Dataset):
    """Dataset of PI controller trajectories for imitation learning.

    Loads CSV files from a directory and extracts windows of input
    [i_d, i_q, e_d, e_q] and target [u_d, u_q] voltage commands.
    All values are normalized to approximately [-1, 1] using motor limits.

    Args:
        data_dir: Directory containing CSV trajectory files.
        config: Data loading configuration.
        window_size: Override config window size.
        stride: Override config stride.
        file_pattern: Glob pattern for CSV files (default: "*.csv").
        max_files: Maximum number of files to load (for debugging).

    Example:
        Load trajectories and get samples::

            dataset = PMSMDataset("data/raw/train", window_size=200)

            inputs, targets = dataset[0]
            # inputs: [window_size, 4] tensor
            # targets: [window_size, 2] tensor
    """

    def __init__(
        self,
        data_dir: str,
        config: DataConfig | None = None,
        window_size: int | None = None,
        stride: int | None = None,
        file_pattern: str = "*.csv",
        max_files: int | None = None,
        error_gain: float = 10.0,
    ):
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()
        self.error_gain = error_gain

        # Override config if provided
        if window_size is not None:
            self.config.window_size = window_size
        if stride is not None:
            self.config.stride = stride

        # Find all CSV files
        self.files = sorted(self.data_dir.glob(file_pattern))

        # Filter out merged files (they're aggregates, not individual trajectories)
        self.files = [f for f in self.files if "merged" not in f.name.lower()]

        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        # Load and preprocess all data
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        self._load_all_files()

    def _find_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """Find the correct column names in the dataframe."""
        cols = df.columns.tolist()

        # Try primary column names
        i_cols = list(self.config.current_cols)
        u_cols = list(self.config.voltage_cols)
        ref_cols = list(self.config.reference_cols)

        # Check if primary columns exist
        if not all(c in cols for c in i_cols):
            # Try alternative names
            i_cols = list(self.config.alt_current_cols)

        if not all(c in cols for c in u_cols):
            u_cols = list(self.config.alt_voltage_cols)

        if not all(c in cols for c in ref_cols):
            ref_cols = list(self.config.alt_reference_cols)

        # Validate
        for c in i_cols + u_cols:
            if c not in cols:
                raise ValueError(
                    f"Column '{c}' not found in data. " f"Available: {cols}"
                )

        return i_cols, u_cols, ref_cols

    def _load_file(self, filepath: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """Load a single CSV file and extract input/target arrays.

        Args:
            filepath: Path to CSV file.

        Returns:
            Tuple of (inputs, targets) arrays, or None if file has issues.
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

        # Skip if too short
        min_length = self.config.window_size + self.config.skip_initial
        if len(df) < min_length:
            return None

        # Find column names
        try:
            i_cols, u_cols, ref_cols = self._find_columns(df)
        except ValueError:
            i_cols, u_cols, _ = self._find_columns(df)
            ref_cols = None

        # --- EXTRACT DATA ---
        i_d = df[i_cols[0]].values
        i_q = df[i_cols[1]].values
        u_d = df[u_cols[0]].values
        u_q = df[u_cols[1]].values

        # NEW: Extract Speed (n)
        # Check if 'n' or 'n_rpm' exists (standard names in your generator)
        if "n" in df.columns:
            n_rpm = df["n"].values
        elif "n_rpm" in df.columns:
            n_rpm = df["n_rpm"].values
        else:
            # Fallback for old files (assume 0 if missing, but better to fail)
            # print(f"Warning: No speed in {filepath.name}")
            return None

        # --- SAFETY CLAMPING ---
        limit = self.config.u_max * 1.2
        u_d = np.clip(u_d, -limit, limit)
        u_q = np.clip(u_q, -limit, limit)

        # Smooth initial transient
        for i in range(min(5, len(u_d))):
            u_d[i] = np.clip(u_d[i], -self.config.u_max, self.config.u_max)
            u_q[i] = np.clip(u_q[i], -self.config.u_max, self.config.u_max)

        # --- COMPUTE INPUT FEATURES ---

        # 1. Reference
        if ref_cols and all(c in df.columns for c in ref_cols):
            i_d_ref = df[ref_cols[0]].values
            i_q_ref = df[ref_cols[1]].values
        else:
            i_d_ref = np.full_like(i_d, i_d[-1])
            i_q_ref = np.full_like(i_q, i_q[-1])

        # 2. Errors
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # 3. Normalization & Amplification
        GAIN = self.error_gain

        # State Normalization
        i_d_norm = i_d / self.config.i_max
        i_q_norm = i_q / self.config.i_max

        # Error Amplification
        e_d_norm = np.clip((e_d / self.config.i_max) * GAIN, -1.0, 1.0)
        e_q_norm = np.clip((e_q / self.config.i_max) * GAIN, -1.0, 1.0)

        # NEW: Speed Normalization
        # We assume Max RPM is around 6000 for standard PMSM,
        # or we can use a safe upper bound like 4000.
        # Let's use 4000 to keep it in [-1, 1] for your 3000 RPM tests.
        N_MAX = 4000.0
        n_norm = n_rpm / N_MAX

        # 4. Target Normalization
        u_d_norm = u_d / self.config.u_max
        u_q_norm = u_q / self.config.u_max

        # --- STACK INPUTS (5 Features Now) ---
        # [i_d, i_q, e_d, e_q, n]
        inputs = np.stack([i_d_norm, i_q_norm, e_d_norm, e_q_norm, n_norm], axis=1)
        targets = np.stack([u_d_norm, u_q_norm], axis=1)

        # Skip initial transient
        inputs = inputs[self.config.skip_initial :]
        targets = targets[self.config.skip_initial :]

        return inputs.astype(np.float32), targets.astype(np.float32)

    def _load_all_files(self) -> None:
        """Load all files and extract windows."""

        n_files = len(self.files)
        print(f"Loading {n_files} trajectory files...", flush=True)

        total_windows = 0
        loaded_files = 0

        for i, filepath in enumerate(self.files):
            result = self._load_file(filepath)
            if result is None:
                continue

            inputs, targets = result
            loaded_files += 1

            # Extract windows
            num_steps = len(inputs)
            window_size = self.config.window_size
            stride = self.config.stride

            for start in range(0, num_steps - window_size + 1, stride):
                end = start + window_size
                window_in = inputs[start:end]
                window_out = targets[start:end]
                self.windows.append((window_in, window_out))
                total_windows += 1

            # Progress update every 100 files
            if (i + 1) % 100 == 0 or i == n_files - 1:
                print(
                    f"  [{i+1}/{n_files}] Loaded {loaded_files} files, {total_windows} windows",
                    flush=True,
                )

        print(
            f"Done! {total_windows} training windows from {loaded_files} files",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.windows[idx]
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
        )

    def get_statistics(self) -> dict:
        """Compute dataset statistics for debugging."""
        all_inputs = np.concatenate([w[0] for w in self.windows], axis=0)
        all_targets = np.concatenate([w[1] for w in self.windows], axis=0)

        return {
            "num_windows": len(self.windows),
            "window_size": self.config.window_size,
            "total_timesteps": len(all_inputs),
            "input_mean": all_inputs.mean(axis=0).tolist(),
            "input_std": all_inputs.std(axis=0).tolist(),
            "target_mean": all_targets.mean(axis=0).tolist(),
            "target_std": all_targets.std(axis=0).tolist(),
            "input_min": all_inputs.min(axis=0).tolist(),
            "input_max": all_inputs.max(axis=0).tolist(),
            "target_min": all_targets.min(axis=0).tolist(),
            "target_max": all_targets.max(axis=0).tolist(),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 100,
    stride: int = 50,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Path to directory with CSV files.
        batch_size: Batch size for training.
        window_size: Timesteps per window.
        stride: Overlap between windows.
        val_split: Fraction of data for validation.
        num_workers: DataLoader workers.
        seed: Random seed for split.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from torch.utils.data import DataLoader, random_split

    # Load full dataset
    dataset = PMSMDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
    )

    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")

    return train_loader, val_loader
