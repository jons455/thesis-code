"""Validation script for Akida PMSM models.

This script loads a trained model (Keras .keras or Akida .fbz) and runs it
against a specific trajectory file to visualize performance.

Usage:
    # Validate Float32 Keras model
    python -m evaluation.snn_keras.validate --model trained_models/akida/final_model.keras --data data/raw/train/trajectory_0.csv

    # Validate Akida .fbz model
    python -m evaluation.snn_keras.validate --model trained_models/akida/akida_model.fbz --data data/raw/train/trajectory_0.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from evaluation.snn_keras.utils.dataset import DataConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_trajectory(
    filepath: str, config: DataConfig
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and normalize a single trajectory file."""
    df = pd.read_csv(filepath)

    # 1. Identify columns
    cols = df.columns.tolist()
    i_cols = ["i_sd", "i_sq"] if "i_sd" in cols else ["i_d", "i_q"]
    u_cols = ["u_sd", "u_sq"] if "u_sd" in cols else ["u_d", "u_q"]
    ref_cols = (
        ["i_sd_ref", "i_sq_ref"] if "i_sd_ref" in cols else ["i_d_ref", "i_q_ref"]
    )

    # Speed
    if "n" in cols:
        n_rpm = df["n"].values
    elif "n_rpm" in cols:
        n_rpm = df["n_rpm"].values
    else:
        raise ValueError("Speed column 'n' or 'n_rpm' not found.")

    # 2. Extract raw data
    i_d = df[i_cols[0]].values
    i_q = df[i_cols[1]].values
    u_d = df[u_cols[0]].values
    u_q = df[u_cols[1]].values

    if all(c in cols for c in ref_cols):
        i_d_ref = df[ref_cols[0]].values
        i_q_ref = df[ref_cols[1]].values
    else:
        # Fallback if no ref (unlikely for training data)
        i_d_ref = np.zeros_like(i_d)
        i_q_ref = np.zeros_like(i_q)

    # 3. Calculate features
    e_d = i_d_ref - i_d
    e_q = i_q_ref - i_q

    # 4. Normalize
    i_d_norm = i_d / config.i_max
    i_q_norm = i_q / config.i_max
    e_d_norm = np.clip((e_d / config.i_max) * config.error_gain, -1.0, 1.0)
    e_q_norm = np.clip((e_q / config.i_max) * config.error_gain, -1.0, 1.0)
    n_norm = n_rpm / config.n_max

    u_d_norm = u_d / config.u_max
    u_q_norm = u_q / config.u_max

    # 5. Stack inputs [i_d, i_q, e_d, e_q, n]
    inputs = np.stack([i_d_norm, i_q_norm, e_d_norm, e_q_norm, n_norm], axis=1).astype(
        np.float32
    )
    targets = np.stack([u_d_norm, u_q_norm], axis=1).astype(np.float32)

    return inputs, targets, df


def run_inference(model_path: str, inputs: np.ndarray) -> np.ndarray:
    """Load model and run inference."""
    path = Path(model_path)
    print(f"Loading model: {path.name}")

    if path.suffix == ".fbz":
        # Akida Inference
        try:
            import akida
        except ImportError:
            print("Error: 'akida' package not found. Cannot run .fbz models.")
            sys.exit(1)

        model = akida.Model(str(path))
        print("Running on Akida engine...")

        # Akida expects specific input types usually, but mapped models accept float/int depending on config
        # For a standard mapped model, we often need quantized inputs if it wasn't mapped with InputQuantizer
        # But our export usually includes InputQuantizer.
        # Let's try direct prediction.

        # Note: Akida predict returns numpy array
        preds = model.predict(inputs)

        # Akida returns quantized integers (e.g. 4-bit)
        # We need to scale them back to float roughly to match the target range [-1, 1]
        # The scaling factor depends on the specific quantization parameters.
        # A simple heuristic for visual validation is to fit the range.

        # FIXME: Proper dequantization requires reading the model parameters or configuration.
        # For visualization, we will normalize the output to match the target std dev.
        print("Note: Akida outputs raw integers. Auto-scaling for visualization...")
        preds = preds.astype(np.float32)

        # Simple auto-scaling if values are large integers
        if np.max(np.abs(preds)) > 1.5:
            # Assuming output is roughly int4/int8 range, scale to [-1, 1]
            scale = 1.0 / np.max(np.abs(preds)) if np.max(np.abs(preds)) > 0 else 1.0
            preds = preds * scale

    else:
        # Keras Inference
        try:
            model = tf.keras.models.load_model(path)
        except Exception:
            # Try loading weights if full model fails (custom object issues)
            from evaluation.snn_keras.models import AkidaController

            print("Standard load failed, trying to rebuild architecture...")
            controller = AkidaController()  # Default config
            # Try to find weights file
            weights_path = path.with_suffix(".weights.h5")
            if weights_path.exists():
                controller.model.load_weights(weights_path)
                model = controller.model
            else:
                raise ValueError("Could not load Keras model.")

        print("Running Keras inference...")
        preds = model.predict(inputs, batch_size=64, verbose=1)

    return preds


def plot_results(
    targets: np.ndarray, preds: np.ndarray, df: pd.DataFrame, filename: str
):
    """Plot ground truth vs predictions."""
    time = np.arange(len(targets)) * 1e-4  # 100us steps

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Voltage d
    axes[0].plot(
        time,
        targets[:, 0],
        label="Target $u_d$",
        color="black",
        alpha=0.6,
        linewidth=1.5,
    )
    axes[0].plot(
        time, preds[:, 0], label="Pred $u_d$", color="tab:blue", linestyle="--"
    )
    axes[0].set_ylabel("Voltage $u_d$ (Norm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Trajectory Validation: {Path(filename).name}")

    # Voltage q
    axes[1].plot(
        time,
        targets[:, 1],
        label="Target $u_q$",
        color="black",
        alpha=0.6,
        linewidth=1.5,
    )
    axes[1].plot(
        time, preds[:, 1], label="Pred $u_q$", color="tab:orange", linestyle="--"
    )
    axes[1].set_ylabel("Voltage $u_q$ (Norm)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Current (Context)
    # Plot original currents to see what the controller is reacting to
    i_d = df["i_sd"].values if "i_sd" in df else df["i_d"].values
    i_q = df["i_sq"].values if "i_sq" in df else df["i_q"].values

    axes[2].plot(time, i_d, label="$i_d$ (Actual)", color="tab:green")
    axes[2].plot(time, i_q, label="$i_q$ (Actual)", color="tab:red")
    axes[2].set_ylabel("Current [A]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Validate Akida/Keras model on trajectory"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to .keras or .fbz model"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV trajectory file"
    )
    args = parser.parse_args()

    # Config (default)
    config = DataConfig()

    # Load data
    print(f"Loading data: {args.data}")
    inputs, targets, df = load_trajectory(args.data, config)

    # Run model
    preds = run_inference(args.model, inputs)

    # Metrics
    mse = np.mean((targets - preds) ** 2)
    mae = np.mean(np.abs(targets - preds))
    print("\nResults:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    # Plot
    plot_results(targets, preds, df, args.data)


if __name__ == "__main__":
    main()
