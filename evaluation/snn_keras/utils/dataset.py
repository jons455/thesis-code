"""
TensorFlow/Keras dataset for PMSM controller training.

This module provides data loading utilities compatible with TensorFlow/Keras,
mirroring the PyTorch dataset but using tf.data for efficient training.

The data format is the same as the PyTorch version:
- Input: [i_d, i_q, e_d, e_q, n] normalized features
- Target: [u_d, u_q] normalized voltage commands

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Try to import the config from embark if available
try:
    from embark.utils.config import DEFAULT_PMSM

    I_MAX_DEFAULT = DEFAULT_PMSM.i_max
    U_MAX_DEFAULT = DEFAULT_PMSM.u_max
except ImportError:
    # Fallback values
    I_MAX_DEFAULT = 230.0  # Maximum current [A]
    U_MAX_DEFAULT = 350.0  # Maximum voltage [V]


@dataclass
class DataConfig:
    """Configuration for data loading and normalization."""

    # Motor limits for normalization
    i_max: float = I_MAX_DEFAULT
    u_max: float = U_MAX_DEFAULT
    n_max: float = 4000.0  # Maximum RPM for normalization

    # Data columns
    current_cols: tuple[str, str] = ("i_sd", "i_sq")
    voltage_cols: tuple[str, str] = ("u_sd", "u_sq")
    reference_cols: tuple[str, str] = ("i_sd_ref", "i_sq_ref")

    # Alternative column names
    alt_current_cols: tuple[str, str] = ("i_d", "i_q")
    alt_voltage_cols: tuple[str, str] = ("u_d", "u_q")
    alt_reference_cols: tuple[str, str] = ("i_d_ref", "i_q_ref")

    # Sequence handling
    window_size: int = 100
    stride: int = 50
    skip_initial: int = 10

    # Error amplification
    error_gain: float = 10.0


class PMSMKerasDataset:
    """
    Dataset for PMSM controller training with TensorFlow.

    This class loads CSV trajectory files and prepares them for
    training Keras models. It provides both numpy arrays and
    tf.data.Dataset objects.

    Args:
        data_dir: Directory containing CSV trajectory files.
        config: Data configuration.
        window_size: Override config window size.
        stride: Override config stride.
        error_gain: Error signal amplification factor.
        max_files: Maximum files to load (for debugging).

    Example:
        >>> dataset = PMSMKerasDataset("data/raw/train")
        >>> x_train, y_train = dataset.get_arrays()
        >>> tf_dataset = dataset.get_tf_dataset(batch_size=32)

    """

    def __init__(
        self,
        data_dir: str,
        config: DataConfig | None = None,
        window_size: int | None = None,
        stride: int | None = None,
        error_gain: float | None = None,
        max_files: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()

        # Override config if provided
        if window_size is not None:
            self.config.window_size = window_size
        if stride is not None:
            self.config.stride = stride
        if error_gain is not None:
            self.config.error_gain = error_gain

        # Find CSV files
        self.files = sorted(self.data_dir.glob("*.csv"))
        self.files = [f for f in self.files if "merged" not in f.name.lower()]

        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        # Load all data
        self.windows_x: list[np.ndarray] = []
        self.windows_y: list[np.ndarray] = []
        self._load_all_files()

        # Stack into arrays
        self._x = np.array(self.windows_x, dtype=np.float32)
        self._y = np.array(self.windows_y, dtype=np.float32)

    def _find_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """Find the correct column names in the dataframe."""
        cols = df.columns.tolist()

        i_cols = list(self.config.current_cols)
        u_cols = list(self.config.voltage_cols)
        ref_cols = list(self.config.reference_cols)

        if not all(c in cols for c in i_cols):
            i_cols = list(self.config.alt_current_cols)
        if not all(c in cols for c in u_cols):
            u_cols = list(self.config.alt_voltage_cols)
        if not all(c in cols for c in ref_cols):
            ref_cols = list(self.config.alt_reference_cols)

        for c in i_cols + u_cols:
            if c not in cols:
                raise ValueError(f"Column '{c}' not found. Available: {cols}")

        return i_cols, u_cols, ref_cols

    def _load_file(self, filepath: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """Load a single CSV file and extract input/target arrays."""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

        min_length = self.config.window_size + self.config.skip_initial
        if len(df) < min_length:
            return None

        try:
            i_cols, u_cols, ref_cols = self._find_columns(df)
        except ValueError:
            i_cols, u_cols, _ = self._find_columns(df)
            ref_cols = None

        # Extract data
        i_d = df[i_cols[0]].values
        i_q = df[i_cols[1]].values
        u_d = df[u_cols[0]].values
        u_q = df[u_cols[1]].values

        # Extract speed
        if "n" in df.columns:
            n_rpm = df["n"].values
        elif "n_rpm" in df.columns:
            n_rpm = df["n_rpm"].values
        else:
            return None

        # Safety clamping
        limit = self.config.u_max * 1.2
        u_d = np.clip(u_d, -limit, limit)
        u_q = np.clip(u_q, -limit, limit)

        for i in range(min(5, len(u_d))):
            u_d[i] = np.clip(u_d[i], -self.config.u_max, self.config.u_max)
            u_q[i] = np.clip(u_q[i], -self.config.u_max, self.config.u_max)

        # Compute references and errors
        if ref_cols and all(c in df.columns for c in ref_cols):
            i_d_ref = df[ref_cols[0]].values
            i_q_ref = df[ref_cols[1]].values
        else:
            i_d_ref = np.full_like(i_d, i_d[-1])
            i_q_ref = np.full_like(i_q, i_q[-1])

        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # Normalize
        GAIN = self.config.error_gain

        i_d_norm = i_d / self.config.i_max
        i_q_norm = i_q / self.config.i_max
        e_d_norm = np.clip((e_d / self.config.i_max) * GAIN, -1.0, 1.0)
        e_q_norm = np.clip((e_q / self.config.i_max) * GAIN, -1.0, 1.0)
        n_norm = n_rpm / self.config.n_max

        u_d_norm = u_d / self.config.u_max
        u_q_norm = u_q / self.config.u_max

        # Stack inputs: [i_d, i_q, e_d, e_q, n]
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
                self.windows_x.append(inputs[start:end])
                self.windows_y.append(targets[start:end])

            if (i + 1) % 100 == 0 or i == n_files - 1:
                print(
                    f"  [{i+1}/{n_files}] Loaded {loaded_files} files, "
                    f"{len(self.windows_x)} windows",
                    flush=True,
                )

        print(
            f"Done! {len(self.windows_x)} training windows from {loaded_files} files",
            flush=True,
        )

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get input/target arrays.

        Returns:
            Tuple of (inputs, targets) arrays with shapes:
            - inputs: (num_windows, window_size, 5)
            - targets: (num_windows, window_size, 2)

        """
        return self._x, self._y

    def get_flattened_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get flattened input/target arrays for non-sequential training.

        This is useful for training feed-forward models that process
        single timesteps rather than sequences.

        Returns:
            Tuple of (inputs, targets) arrays with shapes:
            - inputs: (num_samples, 5)
            - targets: (num_samples, 2)

        """
        x_flat = self._x.reshape(-1, self._x.shape[-1])
        y_flat = self._y.reshape(-1, self._y.shape[-1])
        return x_flat, y_flat

    def get_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        flatten: bool = True,
    ) -> tf.data.Dataset:
        """
        Get TensorFlow Dataset.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            flatten: If True, flatten windows to individual samples.
                    If False, keep window structure for sequence models.

        Returns:
            tf.data.Dataset ready for training.

        """
        if flatten:
            x, y = self.get_flattened_arrays()
        else:
            x, y = self.get_arrays()

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(x)))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def train_val_split(
        self,
        val_split: float = 0.2,
        seed: int = 42,
    ) -> tuple["PMSMKerasDataset", "PMSMKerasDataset"]:
        """
        Split dataset into train and validation.

        Note: Returns views that share the underlying data loading
        but with different window indices.

        Args:
            val_split: Fraction for validation.
            seed: Random seed.

        Returns:
            Tuple of (train_dataset, val_dataset).

        """
        np.random.seed(seed)
        indices = np.random.permutation(len(self._x))

        val_size = int(len(indices) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # Create shallow copies with subset of data
        train_ds = PMSMKerasDataset.__new__(PMSMKerasDataset)
        train_ds.config = self.config
        train_ds._x = self._x[train_indices]
        train_ds._y = self._y[train_indices]

        val_ds = PMSMKerasDataset.__new__(PMSMKerasDataset)
        val_ds.config = self.config
        val_ds._x = self._x[val_indices]
        val_ds._y = self._y[val_indices]

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self._x)

    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        x_flat, y_flat = self.get_flattened_arrays()

        return {
            "num_windows": len(self._x),
            "window_size": self.config.window_size,
            "total_timesteps": len(x_flat),
            "input_mean": x_flat.mean(axis=0).tolist(),
            "input_std": x_flat.std(axis=0).tolist(),
            "target_mean": y_flat.mean(axis=0).tolist(),
            "target_std": y_flat.std(axis=0).tolist(),
            "input_min": x_flat.min(axis=0).tolist(),
            "input_max": x_flat.max(axis=0).tolist(),
            "target_min": y_flat.min(axis=0).tolist(),
            "target_max": y_flat.max(axis=0).tolist(),
        }


def create_tf_dataset(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 100,
    stride: int = 50,
    val_split: float = 0.2,
    error_gain: float = 10.0,
    flatten: bool = True,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create train and validation TensorFlow datasets.

    Args:
        data_dir: Directory with CSV trajectory files.
        batch_size: Batch size.
        window_size: Timesteps per window.
        stride: Overlap between windows.
        val_split: Fraction for validation.
        error_gain: Error signal amplification.
        flatten: Whether to flatten windows for feed-forward training.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, val_dataset).

    """
    dataset = PMSMKerasDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        error_gain=error_gain,
    )

    train_ds, val_ds = dataset.train_val_split(val_split=val_split, seed=seed)

    train_tf = train_ds.get_tf_dataset(
        batch_size=batch_size,
        shuffle=True,
        flatten=flatten,
    )

    val_tf = val_ds.get_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        flatten=flatten,
    )

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    return train_tf, val_tf
