#!/usr/bin/env python
"""
Run benchmark evaluation comparing SNN controller against PI baseline.

This script demonstrates how to use the benchmark framework to evaluate
a trained SNN controller against the classical PI controller baseline.

Usage:
    poetry run python evaluation/core/run_evaluation.py

    # With custom options
    poetry run python evaluation/core/run_evaluation.py --speed 1500 --iq-ref 3.0

"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from embark.benchmark import PIControllerAgent, PMSMEnv  # noqa: E402
from embark.benchmark.agents import (  # noqa: E402
    SNNControllerAgent as SNNBenchmarkController,  # noqa: E402
)
from embark.benchmark.controller_interface import run_benchmark  # noqa: E402
from embark.utils.paths import MODELS_BEST_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SNN controller against PI baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_BEST_DIR / "best_model.pt"),
        help="Path to trained SNN model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    # Operating point
    parser.add_argument(
        "--speed",
        type=float,
        default=1000.0,
        help="Motor speed in RPM",
    )
    parser.add_argument(
        "--id-ref",
        type=float,
        default=0.0,
        help="d-axis current reference in A",
    )
    parser.add_argument(
        "--iq-ref",
        type=float,
        default=5.0,
        help="q-axis current reference in A",
    )

    # Simulation
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=1,
        help="Number of SNN timesteps per control step",
    )

    # Output
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    return parser.parse_args()


def main() -> int:
    """Run the benchmark evaluation."""
    args = parse_args()

    print("=" * 70)
    print("SNN Controller Benchmark Evaluation")
    print("=" * 70)
    print()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        print(
            "Please ensure the model file exists or specify a valid path with --model"
        )
        return 1

    # Create environment
    print(f"Operating Point: {args.iq_ref:.1f}A @ {args.speed:.0f} RPM")
    print(f"Simulation: {args.max_steps} steps")
    print()

    env = PMSMEnv(
        n_rpm=args.speed,
        i_d_ref=args.id_ref,
        i_q_ref=args.iq_ref,
        max_steps=args.max_steps,
    )

    # Create controllers
    print("Loading controllers...")
    pi_controller = PIControllerAgent()
    snn_controller = SNNBenchmarkController(
        checkpoint_path=args.model,
        device=args.device,
        track_spikes=True,
        num_inference_steps=args.inference_steps,
    )

    print("  PI Controller: Ready")
    snn_info = snn_controller.get_info()
    print(
        f"  SNN Controller: {snn_info['parameters']} parameters, "
        f"{snn_info['hidden_size']} hidden neurons"
    )
    print()

    # Run PI baseline
    print("-" * 70)
    print("Running PI Controller (Baseline)")
    print("-" * 70)
    env.reset()
    pi_results = run_benchmark(pi_controller, env, verbose=args.verbose)
    print()
    print(pi_results.summary())
    print()

    # Run SNN controller
    print("-" * 70)
    print("Running SNN Controller")
    print("-" * 70)
    env.reset()
    snn_results = run_benchmark(snn_controller, env, verbose=args.verbose)
    print()
    print(snn_results.summary())
    print()

    # Comparison table
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'PI Controller':>18} {'SNN Controller':>18}")
    print("-" * 70)

    # Accuracy metrics
    print(
        f"{'RMSE i_q [mA]':<25} {pi_results.rmse_iq*1000:>18.2f} "
        f"{snn_results.rmse_iq*1000:>18.2f}"
    )
    print(
        f"{'RMSE i_d [mA]':<25} {pi_results.rmse_id*1000:>18.2f} "
        f"{snn_results.rmse_id*1000:>18.2f}"
    )
    print(
        f"{'MAE i_q [mA]':<25} {pi_results.mae_iq*1000:>18.2f} "
        f"{snn_results.mae_iq*1000:>18.2f}"
    )

    # Dynamics metrics
    print(
        f"{'Settling time i_q [ms]':<25} {pi_results.settling_time_iq*1000:>18.1f} "
        f"{snn_results.settling_time_iq*1000:>18.1f}"
    )
    print(
        f"{'Rise time i_q [ms]':<25} {pi_results.rise_time_iq*1000:>18.1f} "
        f"{snn_results.rise_time_iq*1000:>18.1f}"
    )
    print(
        f"{'Overshoot i_q [%]':<25} {pi_results.overshoot_iq:>18.1f} "
        f"{snn_results.overshoot_iq:>18.1f}"
    )

    # Stability
    pi_stable = "Yes" if pi_results.is_stable else "No"
    snn_stable = "Yes" if snn_results.is_stable else "No"
    print(f"{'Stable':<25} {pi_stable:>18} {snn_stable:>18}")

    # Neuromorphic metrics (SNN only)
    if snn_results.total_spikes is not None:
        print()
        print("-" * 70)
        print("Neuromorphic Metrics (SNN only)")
        print("-" * 70)
        print(f"{'Total spikes':<25} {'-':>18} {snn_results.total_spikes:>18,}")
        print(
            f"{'Spikes per step':<25} {'-':>18} "
            f"{snn_results.spikes_per_step:>18.1f}"
        )
        print(f"{'Sparsity [%]':<25} {'-':>18} " f"{snn_results.sparsity*100:>17.1f}%")
        print(f"{'Neurons':<25} {'-':>18} {snn_results.num_neurons:>18}")
        print(f"{'Synapses':<25} {'-':>18} {snn_results.num_synapses:>18,}")
        if snn_results.inference_latency_mean:
            print(
                f"{'Inference latency [µs]':<25} {'-':>18} "
                f"{snn_results.inference_latency_mean*1e6:>18.1f}"
            )

    print()
    print("=" * 70)

    # Calculate relative performance
    rmse_ratio = snn_results.rmse_iq / max(pi_results.rmse_iq, 1e-9)
    print(f"SNN tracking error is {rmse_ratio:.1f}x the PI baseline")

    if snn_results.is_stable:
        print("SNN controller is STABLE")
    else:
        print("WARNING: SNN controller shows instability")

    print()

    # Save results if requested
    if args.save_results:
        import json
        from datetime import datetime

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "operating_point": {
                "n_rpm": args.speed,
                "i_d_ref": args.id_ref,
                "i_q_ref": args.iq_ref,
            },
            "pi_controller": pi_results.to_dict(),
            "snn_controller": snn_results.to_dict(),
        }

        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
