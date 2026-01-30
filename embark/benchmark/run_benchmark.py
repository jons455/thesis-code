"""PMSM current control benchmark runner.

Main script to run NeuroBench closed-loop benchmarks for PMSM control.
This script validates the integration by running the PI controller
baseline through the NeuroBench BenchmarkClosedLoop framework.

Example:
    Run from project root::

        python -m embark.benchmark.run_benchmark

    Run with full metrics::

        python -m embark.benchmark.run_benchmark --full-metrics
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from embark.benchmark.agents import (  # noqa: E402
    PIControllerAgent,
    PIControllerTorchAgent,
    SNNControllerAgent,
)

# noqa: E402
from embark.benchmark.pmsm_env import PMSMEnv  # noqa: E402
from embark.benchmark.processors import get_default_processors  # noqa: E402
from embark.utils.config import DEFAULT_PMSM  # noqa: E402
from embark.utils.paths import (  # noqa: E402
    BENCHMARK_RESULTS_DIR,
    MODELS_CHECKPOINTS_DIR,
)

# Import comprehensive metrics
try:
    from embark.metrics.benchmark_metrics import (  # noqa: E402
        AccuracyMetrics,  # noqa: F401
        BenchmarkResult,
        DynamicsMetrics,  # noqa: F401
        NeuromorphicMetrics,
        PMSMParameters,  # noqa: F401
        compute_accuracy_metrics,
        compute_dynamics_metrics,
        compute_efficiency_metrics,
        compute_neuromorphic_metrics_from_spikes,  # noqa: F401
        compute_safety_metrics,
        compute_stability_metrics,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: benchmark_metrics not found. Using basic metrics only.")

# NeuroBench imports (optional - for full benchmark)
NEUROBENCH_AVAILABLE = False
try:
    from neurobench.benchmarks import BenchmarkClosedLoop
    from neurobench.metrics.static import (
        ConnectionSparsity,
        Footprint,
    )
    from neurobench.metrics.workload import (
        ActivationSparsity,
        SynapticOperations,
    )

    NEUROBENCH_AVAILABLE = True
except ImportError:
    pass  # NeuroBench not installed - run basic tests only


def run_simple_test():
    """Simple validation test without NeuroBench.

    Tests that PMSMEnv and PIControllerAgent work together correctly.

    Returns:
        True if test passed, False otherwise.
    """
    print("=" * 60)
    print("Simple Integration Test (without NeuroBench)")
    print("=" * 60)

    # Create environment
    env = PMSMEnv(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=500,
    )

    # Create PI controller
    agent = PIControllerAgent(
        kp_d=DEFAULT_PMSM.kp_d_optimum,
        ki_d=DEFAULT_PMSM.ki_stable,
        kp_q=DEFAULT_PMSM.kp_q_optimum,
        ki_q=DEFAULT_PMSM.ki_stable,
    )

    processors = get_default_processors(agent, env.config.i_max, env.config.u_max)

    # Run episode
    state_raw, info = env.reset()
    state = processors.state_preprocessor(state_raw)
    agent.reset()

    total_reward = 0
    for _ in range(500):
        action = agent(state)
        action_env = processors.action_postprocessor(action)
        state_raw, reward, done, truncated, info = env.step(action_env)
        state = processors.state_preprocessor(state_raw)
        total_reward += reward

        if done:
            break

    # Get episode data
    episode_data = env.get_episode_data()

    # Compute metrics
    final_e_d = episode_data[-1]["e_d"]
    final_e_q = episode_data[-1]["e_q"]
    final_error = np.sqrt(final_e_d**2 + final_e_q**2)

    print("\nResults:")
    print(f"  Steps completed: {len(episode_data)}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final tracking error: {final_error*1000:.2f} mA")
    print(f"  Time in target: {env.time_in_range} steps")
    print(
        f"  i_d final: {episode_data[-1]['i_d']:.4f} A (ref: {episode_data[-1]['i_d_ref']:.4f} A)"
    )
    print(
        f"  i_q final: {episode_data[-1]['i_q']:.4f} A (ref: {episode_data[-1]['i_q_ref']:.4f} A)"
    )

    env.close()

    # Check success
    if final_error < 0.1:  # Less than 100 mA error
        print("\n[OK] Simple test PASSED")
        return True
    else:
        print("\n[FAIL] Simple test FAILED - tracking error too high")
        return False


def episode_data_to_dataframe(episode_data: list, env) -> pd.DataFrame:
    """Convert episode data list to DataFrame for metrics computation."""
    df = pd.DataFrame(episode_data)

    # Add speed column (constant for now)
    if "n" not in df.columns:
        df["n"] = env.n_rpm

    return df


def run_comprehensive_benchmark(
    agent,
    agent_name: str,
    env: PMSMEnv,
    max_steps: int = 1000,
    compute_neuromorphic: bool = True,
    return_raw: bool = False,
) -> dict:
    """Run comprehensive benchmark with all metrics.

    Args:
        agent: Controller agent (PI or SNN).
        agent_name: Name for reporting.
        env: Environment instance.
        max_steps: Maximum steps to run.
        compute_neuromorphic: Whether to compute neuromorphic metrics (SNN only).

    Returns:
    dict
        Complete benchmark results
    """
    if not METRICS_AVAILABLE:
        print("Warning: Full metrics not available, using basic metrics")
        return {}

    processors = get_default_processors(agent, env.config.i_max, env.config.u_max)

    # Run episode
    state_raw, info = env.reset()
    state = processors.state_preprocessor(state_raw)
    agent.reset()

    actions_u_d = []
    actions_u_q = []

    for step in range(max_steps):
        if step % 1000 == 0:
            print(".", end="", flush=True)
        action = agent(state)
        action_env = processors.action_postprocessor(action)
        actions_u_d.append(float(action_env[0]))
        actions_u_q.append(float(action_env[1]))

        state_raw, reward, done, truncated, info = env.step(action_env)
        state = processors.state_preprocessor(state_raw)

        if np.isnan(action).any() or np.isnan(state_raw).any():
            print(f"Warning: NaN detected at step {step}")
            break

        if done:
            break

    # Get episode data
    episode_data = env.get_episode_data()
    _df = episode_data_to_dataframe(episode_data, env)  # noqa: F841 (kept for debugging)

    # Extract arrays
    time = np.array([d["time"] for d in episode_data])
    i_d = np.array([d["i_d"] for d in episode_data])
    i_q = np.array([d["i_q"] for d in episode_data])
    i_d_ref = np.array([d["i_d_ref"] for d in episode_data])
    i_q_ref = np.array([d["i_q_ref"] for d in episode_data])
    u_d = np.array(actions_u_d[: len(episode_data)])
    u_q = np.array(actions_u_q[: len(episode_data)])
    n = np.full_like(time, env.n_rpm)

    # Compute all metrics
    accuracy = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
    dynamics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
    efficiency = compute_efficiency_metrics(time, i_d, i_q, u_d, u_q, n)
    safety = compute_safety_metrics(time, i_d, i_q, u_d, u_q)
    stability = compute_stability_metrics(u_d, u_q)

    # Neuromorphic metrics (SNN only)
    neuromorphic = None
    spike_stats = None
    if compute_neuromorphic and hasattr(agent, "get_spike_statistics"):
        spike_stats = agent.get_spike_statistics()

        # Build NeuromorphicMetrics from spike stats
        if "error" not in spike_stats:
            neuromorphic = NeuromorphicMetrics(
                total_spikes=spike_stats.get("total_spikes", 0),
                spikes_per_inference=spike_stats.get("spikes_per_control_step", 0), # Corrected key
                activation_sparsity=spike_stats.get("mean_sparsity", 0),
                num_neurons=spike_stats.get("num_neurons", 0),
                num_synapses=spike_stats.get("num_synapses", 0),
                num_layers=spike_stats.get("num_layers", 0),
                inference_latency_mean=spike_stats.get("inference_latency_mean_s", 0), # Corrected key
                inference_latency_max=spike_stats.get("inference_latency_max_s", 0), # Corrected key
                inference_latency_std=spike_stats.get("inference_latency_std_s", 0), # Corrected key
                total_syops=spike_stats.get("total_syops", 0),
                syops_per_timestep=spike_stats.get("syops_per_timestep", 0),
            )

    # Build result
    result = BenchmarkResult(
        controller_name=agent_name,
        operating_point=f"id={env.i_d_ref:.1f}A, iq={env.i_q_ref:.1f}A @ {env.n_rpm:.0f}rpm",
        timestamp=datetime.now().isoformat(),
        speed_rpm=env.n_rpm,
        i_d_ref=env.i_d_ref,
        i_q_ref=env.i_q_ref,
        accuracy=accuracy,
        dynamics=dynamics,
        efficiency=efficiency,
        safety=safety,
        stability=stability,
        neuromorphic=neuromorphic,
    )

    return {
        "result": result,
        "episode_data": episode_data,
        "spike_stats": spike_stats,
    }


def run_snn_test(
    checkpoint_path: str = str(MODELS_CHECKPOINTS_DIR / "best_model.pt"),
    full_metrics: bool = False,
):
    """
    Test SNN controller in closed-loop.

    Validates that the trained SNN can control the motor without exploding.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    full_metrics : bool
        If True, compute comprehensive metrics using benchmark_metrics.py
    """
    print("\n" + "=" * 60)
    print("SNN Closed-Loop Test")
    print("=" * 60)

    from pathlib import Path

    # Resolve checkpoint path relative to this file
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        print(f"[SKIP] No checkpoint found at {checkpoint}")
        print(
            "Run training first: poetry run python -m evaluation.snn.utils.train --epochs 100"
        )
        return None

    print(f"Loading model from: {checkpoint}")

    # Create environment
    env = PMSMEnv(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=1000 if full_metrics else 500,
    )

    try:
        # Create SNN controller with spike tracking enabled
        agent = SNNControllerAgent(str(checkpoint), track_spikes=True)
        print("Model loaded successfully!")
        print(f"  Parameters: {agent.model.count_parameters():,}")
        print(f"  Network: {agent._network_stats}")

    except Exception as e:
        print(f"[FAIL] Could not load SNN model: {e}")
        import traceback

        traceback.print_exc()
        env.close()
        return False

    # Run with full metrics if available
    if full_metrics and METRICS_AVAILABLE:
        benchmark_result = run_comprehensive_benchmark(
            agent,
            "SNN",
            env,
            max_steps=1000,
            compute_neuromorphic=True,
        )

        if benchmark_result and "result" in benchmark_result:
            print("\n" + benchmark_result["result"].summary())

            # Print additional spike statistics
            if benchmark_result.get("spike_stats"):
                stats = benchmark_result["spike_stats"]
                print("\nDetailed Spike Statistics:")
                print(f"  Total spikes: {stats.get('total_spikes', 'N/A'):,}")
                print(f"  Spikes/timestep: {stats.get('spikes_per_timestep', 0):.1f}")
                print(f"  Mean sparsity: {stats.get('mean_sparsity', 0)*100:.1f}%")
                if "inference_latency_mean" in stats:
                    print(
                        f"  Inference latency: {stats['inference_latency_mean']*1e6:.1f} µs (mean)"
                    )

            env.close()
            return True

    # Fallback to basic test
    processors = get_default_processors(agent, env.config.i_max, env.config.u_max)

    state_raw, info = env.reset()
    state = processors.state_preprocessor(state_raw)
    agent.reset()

    total_reward = 0

    for step in range(500):
        action = agent(state)
        action_env = processors.action_postprocessor(action)
        state_raw, reward, done, truncated, info = env.step(action_env)
        state = processors.state_preprocessor(state_raw)
        total_reward += reward

        # Check for NaN/explosion
        if np.isnan(action).any() or np.isnan(state_raw).any():
            print(f"[FAIL] NaN detected at step {step}")
            env.close()
            return False

        if done:
            break

    # Get episode data
    episode_data = env.get_episode_data()

    # Compute basic metrics
    final_e_d = episode_data[-1]["e_d"]
    final_e_q = episode_data[-1]["e_q"]
    final_error = np.sqrt(final_e_d**2 + final_e_q**2)

    # Compute RMSE over episode
    errors = [(d["e_d"] ** 2 + d["e_q"] ** 2) ** 0.5 for d in episode_data]
    rmse = np.sqrt(np.mean([e**2 for e in errors]))

    # Get spike statistics
    spike_stats = agent.get_spike_statistics()

    print("\nSNN Results:")
    print(f"  Steps completed: {len(episode_data)}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final tracking error: {final_error*1000:.2f} mA")
    print(f"  RMSE: {rmse*1000:.2f} mA")
    print(f"  Time in target: {env.time_in_range} steps")
    print(
        f"  i_d final: {episode_data[-1]['i_d']:.4f} A (ref: {episode_data[-1]['i_d_ref']:.4f} A)"
    )
    print(
        f"  i_q final: {episode_data[-1]['i_q']:.4f} A (ref: {episode_data[-1]['i_q_ref']:.4f} A)"
    )

    # Print spike statistics
    if "error" not in spike_stats:
        print("\n  Neuromorphic Metrics:")
        print(f"    Total spikes: {spike_stats.get('total_spikes', 'N/A'):,}")
        print(f"    Spikes/timestep: {spike_stats.get('spikes_per_timestep', 0):.1f}")
        print(f"    Mean sparsity: {spike_stats.get('mean_sparsity', 0)*100:.1f}%")
        if spike_stats.get("sparsity_per_layer"):
            for i, s in enumerate(spike_stats["sparsity_per_layer"]):
                print(f"    Layer {i} sparsity: {s*100:.1f}%")
        if "inference_latency_mean" in spike_stats:
            print(
                f"    Inference latency: {spike_stats['inference_latency_mean']*1e6:.1f} µs (mean)"
            )
            print(
                f"    Inference latency: {spike_stats['inference_latency_max']*1e6:.1f} µs (max)"
            )

    env.close()

    # Evaluate result
    success_criteria = {
        "stable": not np.isnan(rmse),
        "rmse_reasonable": rmse < 5.0,  # Less than 5A RMSE
    }

    if all(success_criteria.values()):
        print("\n[OK] SNN test PASSED - closed-loop stable!")
        return True
    else:
        print("\n[WARN] SNN test completed but needs improvement:")
        for crit, passed in success_criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {crit}")
        return False


def run_neurobench_benchmark():
    """
    Run full NeuroBench closed-loop benchmark.

    Note: The NeuroBench BenchmarkClosedLoop has specific requirements
    that may not fully match our PMSM control setup. This function
    demonstrates the integration approach.
    """
    print("\n" + "=" * 60)
    print("NeuroBench Closed-Loop Benchmark")
    print("=" * 60)

    if not NEUROBENCH_AVAILABLE:
        print("\n[SKIP] NeuroBench not installed.")
        print("Install from 2025_GC branch if needed for full metrics.")
        return None

    # Create environment
    env = PMSMEnv(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=500,
    )

    # Create PyTorch-wrapped agent for NeuroBench
    agent_net = PIControllerTorchAgent()

    # Wrap in TorchAgent for NeuroBench
    try:
        from neurobench.models import TorchAgent

        agent = TorchAgent(agent_net)
    except Exception as e:
        print(f"Warning: Could not wrap agent with TorchAgent: {e}")
        print("Running with raw agent instead...")
        agent = agent_net

    # Define metrics
    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [ActivationSparsity, SynapticOperations]

    # Create benchmark
    try:
        benchmark = BenchmarkClosedLoop(
            agent=agent,
            environment=env,
            weight_update=False,
            preprocessors=[],
            postprocessors=[],
            metric_list=[static_metrics, workload_metrics],
        )

        # Run benchmark
        print("\nRunning benchmark (this may take a moment)...")
        results, avg_time = benchmark.run(
            nr_interactions=10,
            max_length=500,
            quiet=False,
        )

        print("\nNeuroBench Results:")
        print(f"  Average episode time: {avg_time:.4f} s")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"\nNeuroBench benchmark failed: {e}")
        print(
            "This is expected if the environment interface doesn't fully match NeuroBench expectations."
        )
        print("The simple test above validates that our components work correctly.")
        return False

    finally:
        env.close()


def run_full_comparison(output_dir: str = str(BENCHMARK_RESULTS_DIR)):
    """
    Run full benchmark comparison between PI and SNN controllers.

    Saves results to CSV and generates summary.
    """
    if not METRICS_AVAILABLE:
        print("Error: benchmark_metrics module required for full comparison")
        return

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Full Controller Comparison Benchmark")
    print("=" * 60)

    results = []

    # 4.1 Benchmark Scenarios (The "Holy Trinity")
    # Standardized to 1.0 s at 10 kHz (10,000 steps)
    configs = [
        {
            "name": "nominal",
            "n_rpm": 1000,
            "i_d_ref": 0.0,
            "i_q_ref": 2.0,
            "max_steps": 10000,
            "noise_std": 0.0,
        },
        {
            "name": "high_speed",
            "n_rpm": 3000,
            "i_d_ref": 0.0,
            "i_q_ref": 2.0,
            "max_steps": 10000,
            "noise_std": 0.0,
        },
        {
            "name": "robustness",
            "n_rpm": 1000,
            "i_d_ref": 0.0,
            "i_q_ref": 2.0,
            "max_steps": 10000,
            "noise_std": 0.05,
        },
    ]

    for cfg in configs:
        noise_std = cfg.get("noise_std", 0.0)
        print(
            f"\nOperating point: {cfg['name']} "
            f"(noise σ={noise_std:.2f}A)"
        )
        print("-" * 40)

        # PI Controller
        env = PMSMEnv(
            n_rpm=cfg["n_rpm"],
            i_d_ref=cfg["i_d_ref"],
            i_q_ref=cfg["i_q_ref"],
            max_steps=cfg["max_steps"],
            measurement_noise_std=noise_std,
        )
        pi_agent = PIControllerAgent(
            kp_d=DEFAULT_PMSM.kp_d_optimum,
            ki_d=DEFAULT_PMSM.ki_stable,
            kp_q=DEFAULT_PMSM.kp_q_optimum,
            ki_q=DEFAULT_PMSM.ki_stable,
        )
        pi_result = run_comprehensive_benchmark(
            pi_agent,
            "PI",
            env,
            max_steps=1000,
            compute_neuromorphic=False,
        )
        if pi_result and "result" in pi_result:
            results.append(pi_result["result"])
            print(f"  PI: RMSE_iq = {pi_result['result'].accuracy.RMSE_iq*1000:.2f} mA")
        env.close()

        # SNN Controller
        try:
            checkpoint = MODELS_CHECKPOINTS_DIR / "best_model.pt"
            if checkpoint.exists():
                env = PMSMEnv(
                    n_rpm=cfg["n_rpm"],
                    i_d_ref=cfg["i_d_ref"],
                    i_q_ref=cfg["i_q_ref"],
                    max_steps=cfg["max_steps"],
                    measurement_noise_std=noise_std,
                )
                snn_agent = SNNControllerAgent(str(checkpoint), track_spikes=True)
                snn_result = run_comprehensive_benchmark(
                    snn_agent,
                    "SNN",
                    env,
                    max_steps=1000,
                    compute_neuromorphic=True,
                )
                if snn_result and "result" in snn_result:
                    results.append(snn_result["result"])
                    print(
                        f"  SNN: RMSE_iq = {snn_result['result'].accuracy.RMSE_iq*1000:.2f} mA"
                    )
                env.close()
        except Exception as e:
            print(f"  SNN: Failed - {e}")

    # Save results to CSV
    if results:
        from embark.metrics.benchmark_metrics import compare_controllers

        df = compare_controllers(results, str(output_path))
        print(f"\nResults saved to: {output_path / 'benchmark_comparison.csv'}")

        # Print summary table
        print("\n" + "=" * 60)
        print("Summary Table")
        print("=" * 60)
        summary_cols = [
            "controller",
            "operating_point",
            "accuracy_RMSE_iq",
            "dynamics_settling_time_iq",
            "stability_TV_total",
        ]
        available_cols = [c for c in summary_cols if c in df.columns]
        if available_cols:
            print(df[available_cols].to_string(index=False))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PMSM Benchmark Runner")
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Compute comprehensive metrics (requires benchmark_metrics)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run full PI vs SNN comparison across operating points",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BENCHMARK_RESULTS_DIR),
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PMSM Current Control Benchmark - Validation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Metrics available: {METRICS_AVAILABLE}")
    print("=" * 60)

    if args.compare:
        run_full_comparison(args.output_dir)
        return

    # Run PI controller test first (baseline)
    simple_ok = run_simple_test()

    # Run SNN controller test
    snn_ok = run_snn_test(full_metrics=args.full_metrics)

    if simple_ok and not args.full_metrics:
        # Try NeuroBench benchmark (may have compatibility issues)
        run_neurobench_benchmark()

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  PI Controller:  {'[PASS]' if simple_ok else '[FAIL]'}")
    if snn_ok is None:
        print("  SNN Controller: [SKIP] (no checkpoint)")
    else:
        print(f"  SNN Controller: {'[PASS]' if snn_ok else '[NEEDS TRAINING]'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
