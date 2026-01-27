"""
Controller Comparison Script
============================

Compares PI and SNN controllers on the same scenarios and generates:
1. Step response plots
2. Metrics comparison table
3. Sparsity analysis

Usage:
    python -m embark.benchmark.compare_controllers

    # With custom checkpoint
    python -m embark.benchmark.compare_controllers --checkpoint models/checkpoints/best_model.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embark.benchmark.agents import PIControllerAgent, SNNControllerAgent  # noqa: E402
from embark.benchmark.pmsm_env import PMSMEnv  # noqa: E402
from embark.benchmark.processors import get_default_processors  # noqa: E402
from embark.utils.paths import BENCHMARK_RESULTS_DIR  # noqa: E402


def run_episode(env, agent, max_steps=500):
    """Run a single episode and collect data."""
    processors = get_default_processors(agent, env.config.i_max, env.config.u_max)

    state_raw, _ = env.reset()
    state = processors.state_preprocessor(state_raw)
    agent.reset()

    states = []
    actions = []
    rewards = []
    sparsities = []

    for _ in range(max_steps):
        action = agent(state)
        action_env = processors.action_postprocessor(action)

        # Track sparsity for SNN
        if hasattr(agent, "get_sparsity"):
            sparsity = agent.get_sparsity(state)
            sparsities.append(sparsity)

        states.append(
            state_raw.copy() if isinstance(state_raw, np.ndarray) else state_raw
        )
        actions.append(action_env.copy())

        state_raw, reward, done, truncated, info = env.step(action_env)
        state = processors.state_preprocessor(state_raw)
        rewards.append(reward)

        if done or truncated:
            break

    episode_data = env.get_episode_data()

    return {
        "episode_data": episode_data,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "sparsities": sparsities,
        "time_in_range": env.time_in_range,
    }


def compute_metrics(episode_data):
    """Compute metrics from episode data."""
    i_d = np.array([d["i_d"] for d in episode_data])
    i_q = np.array([d["i_q"] for d in episode_data])
    i_d_ref = np.array([d["i_d_ref"] for d in episode_data])
    i_q_ref = np.array([d["i_q_ref"] for d in episode_data])
    u_d = np.array([d["u_d"] for d in episode_data])
    u_q = np.array([d["u_q"] for d in episode_data])

    # Errors
    e_d = i_d_ref - i_d
    e_q = i_q_ref - i_q
    error_mag = np.sqrt(e_d**2 + e_q**2)

    # RMSE
    rmse = np.sqrt(np.mean(error_mag**2))

    # MAE
    mae = np.mean(error_mag)

    # Final error
    final_error = error_mag[-1]

    # Total Variation (control smoothness)
    u_d_diff = np.diff(u_d)
    u_q_diff = np.diff(u_q)
    tv = np.sum(np.abs(u_d_diff)) + np.sum(np.abs(u_q_diff))

    # Settling time (2% band)
    threshold = 0.02 * np.abs(i_q_ref[0]) if i_q_ref[0] != 0 else 0.02
    settled_mask = error_mag < threshold
    if np.any(settled_mask):
        # Find first time we stay within threshold
        for i in range(len(settled_mask)):
            if np.all(settled_mask[i:]):
                settling_time_ms = i * 0.1  # 10 kHz = 0.1 ms per step
                break
        else:
            settling_time_ms = np.nan
    else:
        settling_time_ms = np.nan

    return {
        "rmse": rmse * 1000,  # mA
        "mae": mae * 1000,  # mA
        "final_error": final_error * 1000,  # mA
        "total_variation": tv,
        "settling_time_ms": settling_time_ms,
    }


def plot_comparison(pi_data, snn_data, save_path=None):
    """Generate comparison plots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    pi_ep = pi_data["episode_data"]
    snn_ep = snn_data["episode_data"]

    time_pi = np.arange(len(pi_ep)) * 0.1  # ms
    time_snn = np.arange(len(snn_ep)) * 0.1

    # i_q tracking
    ax = axes[0, 0]
    ax.plot(
        time_pi, [d["i_q_ref"] for d in pi_ep], "k--", label="Reference", linewidth=2
    )
    ax.plot(time_pi, [d["i_q"] for d in pi_ep], "b-", label="PI", linewidth=1.5)
    ax.plot(
        time_snn,
        [d["i_q"] for d in snn_ep],
        "r-",
        label="SNN",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("i_q [A]")
    ax.set_title("q-axis Current Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # i_d tracking
    ax = axes[0, 1]
    ax.plot(
        time_pi, [d["i_d_ref"] for d in pi_ep], "k--", label="Reference", linewidth=2
    )
    ax.plot(time_pi, [d["i_d"] for d in pi_ep], "b-", label="PI", linewidth=1.5)
    ax.plot(
        time_snn,
        [d["i_d"] for d in snn_ep],
        "r-",
        label="SNN",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("i_d [A]")
    ax.set_title("d-axis Current Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tracking error
    ax = axes[1, 0]
    pi_error = np.sqrt(np.array([d["e_d"] ** 2 + d["e_q"] ** 2 for d in pi_ep])) * 1000
    snn_error = (
        np.sqrt(np.array([d["e_d"] ** 2 + d["e_q"] ** 2 for d in snn_ep])) * 1000
    )
    ax.plot(time_pi, pi_error, "b-", label="PI", linewidth=1.5)
    ax.plot(time_snn, snn_error, "r-", label="SNN", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Tracking Error [mA]")
    ax.set_title("Tracking Error Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Voltage output
    ax = axes[1, 1]
    ax.plot(time_pi, [d["u_q"] for d in pi_ep], "b-", label="PI u_q", linewidth=1.5)
    ax.plot(
        time_snn,
        [d["u_q"] for d in snn_ep],
        "r-",
        label="SNN u_q",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("u_q [V]")
    ax.set_title("q-axis Voltage Command")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sparsity (SNN only)
    ax = axes[2, 0]
    if snn_data["sparsities"]:
        sparsity_steps = np.arange(len(snn_data["sparsities"]))
        for key in snn_data["sparsities"][0]:
            vals = [s[key] * 100 for s in snn_data["sparsities"]]
            ax.plot(sparsity_steps, vals, label=key)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Sparsity [%]")
        ax.set_title("SNN Activation Sparsity")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No sparsity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Metrics bar chart
    ax = axes[2, 1]
    pi_metrics = compute_metrics(pi_ep)
    snn_metrics = compute_metrics(snn_ep)

    metrics_to_plot = ["rmse", "mae", "final_error"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    pi_vals = [pi_metrics[m] for m in metrics_to_plot]
    snn_vals = [snn_metrics[m] for m in metrics_to_plot]

    ax.bar(x - width / 2, pi_vals, width, label="PI", color="blue", alpha=0.7)
    ax.bar(x + width / 2, snn_vals, width, label="SNN", color="red", alpha=0.7)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Error [mA]")
    ax.set_title("Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["RMSE", "MAE", "Final Error"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def print_metrics_table(pi_metrics, snn_metrics, snn_sparsities=None):
    """Print formatted metrics comparison."""
    print("\n" + "=" * 60)
    print("CONTROLLER COMPARISON METRICS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'PI':>15} {'SNN':>15} {'Ratio':>10}")
    print("-" * 65)

    for key in ["rmse", "mae", "final_error"]:
        pi_val = pi_metrics[key]
        snn_val = snn_metrics[key]
        ratio = snn_val / pi_val if pi_val > 0 else float("inf")
        print(f"{key:<25} {pi_val:>12.2f} mA {snn_val:>12.2f} mA {ratio:>8.1f}x")

    # Total Variation
    pi_tv = pi_metrics["total_variation"]
    snn_tv = snn_metrics["total_variation"]
    tv_ratio = snn_tv / pi_tv if pi_tv > 0 else float("inf")
    print(
        f"{'total_variation':<25} {pi_tv:>12.2f} V  {snn_tv:>12.2f} V  {tv_ratio:>8.1f}x"
    )

    # Settling time
    pi_st = pi_metrics["settling_time_ms"]
    snn_st = snn_metrics["settling_time_ms"]
    if not np.isnan(pi_st) and not np.isnan(snn_st):
        st_ratio = snn_st / pi_st if pi_st > 0 else float("inf")
        print(
            f"{'settling_time':<25} {pi_st:>12.2f} ms {snn_st:>12.2f} ms {st_ratio:>8.1f}x"
        )

    # Sparsity (SNN only)
    if snn_sparsities:
        print("\n" + "-" * 65)
        print("SNN ACTIVATION SPARSITY")
        avg_sparsity = {}
        for key in snn_sparsities[0]:
            vals = [s[key] for s in snn_sparsities]
            avg_sparsity[key] = np.mean(vals)
            print(f"  {key}: {avg_sparsity[key]*100:.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare PI and SNN controllers")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pt",
        help="Path to SNN checkpoint",
    )
    parser.add_argument("--n_rpm", type=float, default=1000, help="Motor speed [rpm]")
    parser.add_argument(
        "--i_q_ref", type=float, default=2.0, help="q-axis current reference [A]"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    print("=" * 60)
    print("CONTROLLER COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Operating Point: {args.n_rpm} RPM, i_q_ref = {args.i_q_ref} A")
    print(f"SNN Checkpoint: {args.checkpoint}")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Run training first: python -m evaluation.snn.utils.train --epochs 100")
        return

    # Create environment
    env = PMSMEnv(
        n_rpm=args.n_rpm,
        i_d_ref=0.0,
        i_q_ref=args.i_q_ref,
        max_steps=args.max_steps,
    )

    # Run PI controller
    print("\nRunning PI controller...")
    pi_agent = PIControllerAgent()
    pi_data = run_episode(env, pi_agent, args.max_steps)
    pi_metrics = compute_metrics(pi_data["episode_data"])
    print(f"  RMSE: {pi_metrics['rmse']:.2f} mA")

    # Run SNN controller
    print("\nRunning SNN controller...")
    try:
        snn_agent = SNNControllerAgent(str(checkpoint_path))
        snn_data = run_episode(env, snn_agent, args.max_steps)
        snn_metrics = compute_metrics(snn_data["episode_data"])
        print(f"  RMSE: {snn_metrics['rmse']:.2f} mA")
    except Exception as e:
        print(f"  [ERROR] SNN failed: {e}")
        env.close()
        return

    env.close()

    # Print comparison table
    print_metrics_table(pi_metrics, snn_metrics, snn_data["sparsities"])

    # Generate plots
    default_output = (
        BENCHMARK_RESULTS_DIR
        / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    output_path = args.output or str(default_output)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(pi_data, snn_data, save_path=str(output_path))

    plt.show()


if __name__ == "__main__":
    main()
