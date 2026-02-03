#!/usr/bin/env python
"""
Run benchmark evaluation for Keras/Akida models (Float or Quantized).

This script allows closed-loop benchmarking of Keras (.keras) or Akida (.fbz) models
using the standard NeuroBench-aligned harness.

Usage:
    poetry run python evaluation/snn_keras/run_benchmark.py --model akida/best_model.keras

"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from embark.benchmark.agents import PIControllerAgent  # noqa: E402
from embark.benchmark.harness.closed_loop import ClosedLoopHarness  # noqa: E402
from embark.benchmark.metrics.accumulators.dynamics import (  # noqa: E402
    Overshoot,
    SettlingTime,
)
from embark.benchmark.metrics.accumulators.efficiency import ControlEffort  # noqa: E402
from embark.benchmark.metrics.accumulators.tracking import TrackingRMSE  # noqa: E402
from embark.benchmark.tasks.pmsm_current_control import (  # noqa: E402
    PMSMCurrentControlTask,
)
from evaluation.snn_keras.akida_agent import AkidaControllerAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Keras/Akida controller against PI baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Keras (.keras) or Akida (.fbz) model",
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

    # Output
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
    print("Akida/Keras Controller Benchmark Evaluation")
    print("=" * 70)
    print()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Create task
    print(f"Operating Point: {args.iq_ref:.1f}A @ {args.speed:.0f} RPM")
    print(f"Simulation: {args.max_steps} steps")
    print()

    task = PMSMCurrentControlTask.from_config(
        n_rpm=args.speed,
        i_d_ref=args.id_ref,
        i_q_ref=args.iq_ref,
        max_steps=args.max_steps,
    )

    # Metrics factory
    def create_metrics():
        return [
            TrackingRMSE(tracked_keys=["i_q", "i_d"]),
            SettlingTime(tracked_key="i_q", threshold=0.05),  # 5% settling
            Overshoot(tracked_key="i_q"),
            ControlEffort(),
        ]

    # Create controllers
    print("Loading controllers...")

    # 1. PI Controller (Baseline)
    pi_agent = PIControllerAgent.from_system_config(task.physics_engine.config)
    pi_metrics = create_metrics()
    pi_harness = ClosedLoopHarness(task=task, controller=pi_agent, metrics=pi_metrics)

    # 2. Akida/Keras Controller
    # AkidaControllerAgent is a DictController, so no adapter needed!
    try:
        akida_agent = AkidaControllerAgent(
            model_path=str(model_path),
            # Ensure these match your training configuration!
            i_max=task.physics_engine.config.i_max,
            u_max=task.physics_engine.config.u_max,
            error_gain=10.0,  # Standard training gain
        )
    except Exception as e:
        print(f"Failed to load Akida agent: {e}")
        return 1

    akida_metrics = create_metrics()
    akida_harness = ClosedLoopHarness(
        task=task, controller=akida_agent, metrics=akida_metrics
    )

    print("  PI Controller: Ready")
    info = akida_agent.get_info()
    print(f"  Akida Controller: {info['type']} ({info['name']})")
    print()

    # Run PI baseline
    print("-" * 70)
    print("Running PI Controller (Baseline)")
    print("-" * 70)
    pi_results = pi_harness.run()
    print("Done.")
    print()

    # Run Akida controller
    print("-" * 70)
    print("Running Akida/Keras Controller")
    print("-" * 70)
    try:
        akida_results = akida_harness.run()
        print("Done.")
    except Exception as e:
        print(f"Error running Akida controller: {e}")
        akida_results = {}
    print()

    # Comparison table
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'PI Controller':>18} {'Akida Controller':>18}")
    print("-" * 70)

    # Helper to safe get metric
    def get_val(results, key, scale=1.0):
        val = results.get(key, 0.0)
        return val * scale

    # Accuracy metrics
    print(
        f"{'RMSE i_q [A]':<25} {get_val(pi_results, 'rmse_i_q'):>18.4f} "
        f"{get_val(akida_results, 'rmse_i_q'):>18.4f}"
    )
    print(
        f"{'RMSE i_d [A]':<25} {get_val(pi_results, 'rmse_i_d'):>18.4f} "
        f"{get_val(akida_results, 'rmse_i_d'):>18.4f}"
    )

    # Dynamics metrics
    print(
        f"{'Settling time i_q [s]':<25} {get_val(pi_results, 'settling_time'):>18.4f} "
        f"{get_val(akida_results, 'settling_time'):>18.4f}"
    )
    print(
        f"{'Overshoot i_q [%]':<25} {get_val(pi_results, 'overshoot'):>18.1f} "
        f"{get_val(akida_results, 'overshoot'):>18.1f}"
    )

    # Efficiency
    print(
        f"{'Control Effort':<25} {get_val(pi_results, 'control_effort'):>18.1f} "
        f"{get_val(akida_results, 'control_effort'):>18.1f}"
    )

    print()
    print("=" * 70)

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
            "pi_controller": pi_results,
            "akida_controller": akida_results,
        }

        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
