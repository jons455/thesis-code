"""Validate training data quality and plot sample trajectory."""

import glob
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.utils.paths import DATA_RAW_DIR  # noqa: E402


def plot_trajectory(df: pd.DataFrame, filename: str):
    """Plots i_q and i_d tracking for visual inspection."""
    t = df["time"] * 1000  # Convert to ms for readability

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. q-Axis (Torque)
    ax1.step(t, df["i_q_ref"], "k--", label="Reference", alpha=0.7)
    ax1.plot(t, df["i_q"], "b-", label="Actual", linewidth=1.5)
    ax1.set_ylabel("Current $i_q$ [A]")
    ax1.set_title(f"Trajectory Validation: {filename}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. d-Axis (Flux)
    ax2.step(t, df["i_d_ref"], "k--", label="Reference", alpha=0.7)
    ax2.plot(t, df["i_d"], "r-", label="Actual", linewidth=1.5)
    ax2.set_ylabel("Current $i_d$ [A]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Voltages (Actions)
    ax3.plot(t, df["u_q"], "b-", label="$u_q$", alpha=0.6)
    ax3.plot(t, df["u_d"], "r-", label="$u_d$", alpha=0.6)
    ax3.set_ylabel("Voltage [V]")
    ax3.set_xlabel("Time [ms]")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    # Ensure docs/plots exists
    plots_dir = Path(__file__).parent.parent / "docs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / "validation_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[Plot] Saved sample trajectory to: {plot_path}")
    # plt.show() # Uncomment if you run locally and want a popup


def validate_data(data_dir: str):
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    print("=== Training Data Validation (Multi-Step) ===")
    print(f"Directory: {data_dir}")
    print(f"Total files: {len(files)}")

    if len(files) == 0:
        print("No files found!")
        return

    # Statistics accumulators
    mae_iq_list = []
    mae_id_list = []
    bad_files = []

    print("\nAnalyzing tracking error across full episodes...")

    for f in files:
        df = pd.read_csv(f)

        # Calculate Mean Absolute Error (MAE) over the WHOLE file
        # We allow small transient errors during steps, so Average is better than Max
        mae_iq = (df["i_q"] - df["i_q_ref"]).abs().mean()
        mae_id = (df["i_d"] - df["i_d_ref"]).abs().mean()

        mae_iq_list.append(mae_iq)
        mae_id_list.append(mae_id)

        # Check for catastrophic failures (e.g. mean error > 2A is huge)
        if mae_iq > 2.0:
            bad_files.append((Path(f).name, mae_iq))

    # --- Reporting ---
    print("\n--- Statistics (Average per file) ---")
    print(f"i_q Mean MAE: {np.mean(mae_iq_list):.4f} A")
    print(f"i_d Mean MAE: {np.mean(mae_id_list):.4f} A")

    print(f"i_q Worst File MAE: {np.max(mae_iq_list):.4f} A")

    # Dynamic Tracking Quality Check
    # If MAE is < 0.5A, it means the PI controller followed the steps mostly well
    # (Steps cause momentary errors, so 0.0 is impossible)
    good_files = sum([e < 0.5 for e in mae_iq_list])
    print(f"Good Tracking Files (MAE < 0.5A): {good_files} / {len(files)}")

    if bad_files:
        print(f"\nWARNING: {len(bad_files)} files seem broken (High Mean Error):")
        for name, err in bad_files[:5]:
            print(f"  {name}: {err:.2f} A average error")

    # --- Visualization ---
    if len(files) > 0:
        # Pick a random file to plot
        random_file = random.choice(files)
        df_sample = pd.read_csv(random_file)
        plot_trajectory(df_sample, random_file)

    if len(bad_files) == 0 and np.mean(mae_iq_list) < 0.5:
        print("\n=== DATA LOOKS GREAT (Dynamic & Clean) ===")
    else:
        print("\n=== DATA MIGHT HAVE ISSUES ===")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_RAW_DIR / "train")
    validate_data(data_dir)
