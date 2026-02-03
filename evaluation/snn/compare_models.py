"""
Compare trained SNN models against PI baseline.
Now includes Symmetry Tests (Positive vs Negative) to verify data balancing.

Usage:
    poetry run python -m evaluation.snn.compare_models
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from embark.benchmark.agents import PIControllerAgent, SNNControllerAgent
from embark.benchmark.metrics import (
    ControlEffort,
    Overshoot,
    SettlingTime,
    TrackingRMSE,
)
from embark.benchmark.processors import LinearActionProcessor
from embark.benchmark.tasks import PMSMCurrentControlTask
from embark.metrics.benchmark_metrics import (
    BenchmarkResult,
    NeuromorphicMetrics,
    compute_accuracy_metrics,
    compute_dynamics_metrics,
    compute_efficiency_metrics,
    compute_safety_metrics,
    compute_stability_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- UPDATED SCENARIOS ---
SCENARIOS = [
    {
        "name": "A_Nominal_Pos",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,  # Step to +2A
        "max_steps": 2000,  # Shorter is fine for step response
        "noise_std": 0.0,
        "desc": "Step Response (0 -> 2A)",
    },
    {
        "name": "B_Nominal_Neg",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": -2.0,  # Step to -2A (CRITICAL TEST for new data)
        "max_steps": 2000,
        "noise_std": 0.0,
        "desc": "Negative Step (0 -> -2A)",
    },
    {
        "name": "C_HighSpeed",
        "n_rpm": 3000,  # High speed test
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 2000,
        "noise_std": 0.0,
        "desc": "High Speed (3000 rpm)",
    },
]


def find_models():
    """Find all best_model.pt or final_model.pt files in trained_models/."""
    models = []
    base_dir = PROJECT_ROOT / "trained_models"

    if not base_dir.exists():
        print(f"Warning: {base_dir} does not exist.")
        return models

    # Scan subdirectories
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            # Check for best_model.pt first, then final_model.pt
            best = subdir / "best_model.pt"
            final = subdir / "final_model.pt"

            # Label the model clearly
            if best.exists():
                models.append((f"SNN_{subdir.name}", best))
            elif final.exists():
                models.append((f"SNN_{subdir.name}", final))

    return sorted(models)


def plot_combined_response(scenario_name, results_list):
    """Plots combined step responses for easy comparison."""
    if not results_list:
        return

    plot_dir = PROJECT_ROOT / "docs" / "plots" / "comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Reference (dashed black)
    first_data = results_list[0][1]
    time = np.array([d["time"] for d in first_data])
    ref = np.array([d["i_q_ref"] for d in first_data])
    plt.plot(time, ref, "k--", label="Reference", linewidth=2, alpha=0.7)

    # Models
    for model_name, episode_data in results_list:
        t = np.array([d["time"] for d in episode_data])
        iq = np.array([d["i_q"] for d in episode_data])

        # Style logic
        lw = 2.0 if "PI" in model_name else 1.5
        alpha = 0.9 if "PI" in model_name or "delta" in model_name else 0.7

        plt.plot(t, iq, label=model_name, linewidth=lw, alpha=alpha)

    plt.title(f"Scenario: {scenario_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Current $i_q$ [A]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = plot_dir / f"compare_{scenario_name}.png"
    plt.savefig(save_path, dpi=150)
    print(f"    Saved plot: {save_path}")
    plt.close()


def run_comprehensive_benchmark(
    controller,
    controller_name: str,
    task: PMSMCurrentControlTask,
    max_steps: int,
    compute_neuromorphic: bool,
):
    """Run comprehensive benchmark using the closed-loop task."""
    is_tensor = hasattr(controller, "forward")

    if is_tensor:
        action_proc = LinearActionProcessor(
            output_keys=["v_d", "v_q"],
            bounds={
                "v_d": (
                    -task.physics_engine.config.u_max,
                    task.physics_engine.config.u_max,
                ),
                "v_q": (
                    -task.physics_engine.config.u_max,
                    task.physics_engine.config.u_max,
                ),
            },
        )
        action_proc.configure(task.physics_engine.config)
    else:
        action_proc = None

    def _get_input_size(ctrl) -> int:
        model = getattr(ctrl, "model", None)
        if model is not None:
            config = getattr(model, "config", None)
            if config is not None and hasattr(config, "input_size"):
                return int(config.input_size)
            if hasattr(model, "input_size"):
                return int(model.input_size)
            if hasattr(model, "layers") and model.layers:
                layer0 = model.layers[0]
                if hasattr(layer0, "in_features"):
                    return int(layer0.in_features)
        return 4

    def _build_obs(state, reference, input_size) -> torch.Tensor:
        i_d = float(state["i_d"])
        i_q = float(state["i_q"])
        e_d = float(reference["i_d_ref"]) - i_d
        e_q = float(reference["i_q_ref"]) - i_q
        omega = float(state.get("omega", 0.0))

        i_max = task.physics_engine.config.i_max
        omega_max = getattr(task.physics_engine.config, "omega_max", 1.0)

        if input_size == 5:
            values = [i_d, i_q, e_d, e_q, omega]
            bounds = [(-i_max, i_max)] * 4 + [(-omega_max, omega_max)]
        else:
            values = [i_d, i_q, e_d, e_q]
            bounds = [(-i_max, i_max)] * 4

        normed = [
            2 * (val - low) / (high - low) - 1
            for val, (low, high) in zip(values, bounds)
        ]
        return torch.tensor(normed, dtype=torch.float32)

    input_size = _get_input_size(controller) if is_tensor else 0

    metrics = [
        TrackingRMSE(tracked_keys=["i_q", "i_d"]),
        SettlingTime(tracked_key="i_q", threshold=0.04),
        Overshoot(tracked_key="i_q"),
        ControlEffort(),
    ]

    state, reference = task.reset()
    controller.reset()
    for m in metrics:
        m.reset()

    episode_data = []
    step = 0
    done = False

    while not done and step < max_steps:
        if is_tensor:
            obs = _build_obs(state, reference, input_size)
            action_tensor = controller.forward(obs)
            action = action_proc(action_tensor, task.physics_engine.config)
            controller_info = getattr(controller, "last_info", None)
        else:
            action = controller(state, reference)
            controller_info = None

        next_state, next_ref, done = task.step(action)

        episode_data.append(
            {
                "time": state.get("time", step * task.physics_engine.config.tau),
                "i_d": state["i_d"],
                "i_q": state["i_q"],
                "i_d_ref": reference["i_d_ref"],
                "i_q_ref": reference["i_q_ref"],
                "v_d": action["v_d"],
                "v_q": action["v_q"],
                "omega": state.get("omega", 0.0),
            }
        )

        for m in metrics:
            m.update(state, reference, action, next_state, controller_info)

        state, reference = next_state, next_ref
        step += 1

    # Extract arrays for comprehensive metrics
    time_arr = np.array([d["time"] for d in episode_data])
    i_d = np.array([d["i_d"] for d in episode_data])
    i_q = np.array([d["i_q"] for d in episode_data])
    i_d_ref = np.array([d["i_d_ref"] for d in episode_data])
    i_q_ref = np.array([d["i_q_ref"] for d in episode_data])
    u_d = np.array([d["v_d"] for d in episode_data])
    u_q = np.array([d["v_q"] for d in episode_data])
    n = np.array([d["omega"] * 60.0 / (2 * np.pi) for d in episode_data])

    accuracy = compute_accuracy_metrics(time_arr, i_d, i_q, i_d_ref, i_q_ref)
    dynamics = compute_dynamics_metrics(time_arr, i_d, i_q, i_d_ref, i_q_ref)
    efficiency = compute_efficiency_metrics(time_arr, i_d, i_q, u_d, u_q, n)
    safety = compute_safety_metrics(time_arr, i_d, i_q, u_d, u_q)
    stability = compute_stability_metrics(u_d, u_q)

    neuromorphic = None
    spike_stats = None
    if compute_neuromorphic and hasattr(controller, "get_spike_statistics"):
        spike_stats = controller.get_spike_statistics()
        if spike_stats and "error" not in spike_stats:
            neuromorphic = NeuromorphicMetrics(
                total_spikes=spike_stats.get("total_spikes", 0),
                spikes_per_inference=spike_stats.get("spikes_per_control_step", 0),
                activation_sparsity=spike_stats.get("mean_sparsity", 0),
                num_neurons=spike_stats.get("num_neurons", 0),
                num_synapses=spike_stats.get("num_synapses", 0),
                num_layers=spike_stats.get("num_layers", 0),
                inference_latency_mean=spike_stats.get("inference_latency_mean_s", 0),
                inference_latency_max=spike_stats.get("inference_latency_max_s", 0),
                inference_latency_std=spike_stats.get("inference_latency_std_s", 0),
                total_syops=spike_stats.get("total_syops", 0),
                syops_per_timestep=spike_stats.get("syops_per_timestep", 0),
            )

    benchmark_result = BenchmarkResult(
        controller_name=controller_name,
        operating_point=f"id={i_d_ref[-1]:.1f}A, iq={i_q_ref[-1]:.1f}A @ {n.mean():.0f}rpm",
        timestamp="",
        speed_rpm=float(n.mean()) if len(n) else 0.0,
        i_d_ref=float(i_d_ref[-1]) if len(i_d_ref) else 0.0,
        i_q_ref=float(i_q_ref[-1]) if len(i_q_ref) else 0.0,
        accuracy=accuracy,
        dynamics=dynamics,
        efficiency=efficiency,
        safety=safety,
        stability=stability,
        neuromorphic=neuromorphic,
    )

    return {
        "result": benchmark_result,
        "episode_data": episode_data,
        "spike_stats": spike_stats,
    }


def run_evaluation():
    print("=" * 60)
    print("SNN FINAL SHOWDOWN")
    print("=" * 60)

    models = find_models()
    if not models:
        print("No models found!")
        return

    # Add Baseline to the list artificially so we loop over it cleanly
    # We will handle the instantiation inside the loop
    model_queue = [("PI_Baseline", None)] + models

    full_results = []
    plot_data_storage = {s["name"]: [] for s in SCENARIOS}

    for model_name, model_path in model_queue:
        print(f"\nEvaluating: {model_name}")

        for scen in SCENARIOS:
            print(f"  > {scen['name']}...", end="", flush=True)

            task = PMSMCurrentControlTask.from_config(
                n_rpm=scen["n_rpm"],
                i_d_ref=scen["i_d_ref"],
                i_q_ref=scen["i_q_ref"],
                max_steps=scen["max_steps"],
            )

            # Setup Agent
            if model_name == "PI_Baseline":
                agent = PIControllerAgent.from_system_config(task.physics_engine.config)
                is_neuromorphic = False
            else:
                # IMPORTANT: SNNControllerAgent must handle the Gain internally!
                agent = SNNControllerAgent(str(model_path), track_spikes=True)
                is_neuromorphic = True

            # Run Benchmark
            res = run_comprehensive_benchmark(
                agent,
                model_name,
                task,
                max_steps=scen["max_steps"],
                compute_neuromorphic=is_neuromorphic,
            )
            task.physics_engine.close()

            if res and "result" in res:
                r = res["result"]
                # Store for plotting
                if "episode_data" in res:
                    plot_data_storage[scen["name"]].append(
                        (model_name, res["episode_data"])
                    )

                full_results.append(
                    {
                        "Model": model_name,
                        "Scenario": scen["name"],
                        "RMSE": r.accuracy.RMSE_iq,
                        "ITAE": r.accuracy.ITAE_iq,
                        "TV": r.stability.TV_total,
                        "SyOps": r.neuromorphic.syops_per_timestep
                        if r.neuromorphic
                        else 0.0,
                        "Sparsity": r.neuromorphic.activation_sparsity
                        if r.neuromorphic
                        else 0.0,
                        "LAC": r.lac_score,
                    }
                )
                print(" Done.")
            else:
                print(" Failed.")

    # --- PLOTTING ---
    print("\nGenerating Plots...")
    for s_name, data in plot_data_storage.items():
        plot_combined_response(s_name, data)

    # --- REPORTING ---
    df = pd.DataFrame(full_results)
    if not df.empty:
        print("\n" + "=" * 60)
        print("FINAL LEADERBOARD (Sorted by RMSE)")
        print("=" * 60)

        # Average RMSE across scenarios
        leaderboard = (
            df.groupby("Model")[["RMSE", "Sparsity"]].mean().sort_values("RMSE")
        )
        print(leaderboard)

        # Save CSV
        df.to_csv("docs/final_benchmark_results.csv", index=False)
        print("\nDetailed results saved to docs/final_benchmark_results.csv")


if __name__ == "__main__":
    run_evaluation()
