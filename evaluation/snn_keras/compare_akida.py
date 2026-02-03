"""
Compare Akida Models vs PyTorch SNN vs PI Baseline.

This script runs a comprehensive benchmark comparing:
1. Classical PI Controller (Baseline)
2. PyTorch SNN (Previous Best)
3. Akida Keras Model (Float32)
4. Akida Hardware Model (.fbz) [If available]

Usage:
    poetry run python -m evaluation.snn_keras.compare_akida
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from evaluation.snn_keras.akida_agent import AkidaControllerAgent

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define Scenarios
SCENARIOS = [
    {
        "name": "Step_Response",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 2000,
        "noise_std": 0.0,
    },
    {
        "name": "Negative_Step",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": -2.0,
        "max_steps": 2000,
        "noise_std": 0.0,
    },
    {
        "name": "High_Speed",
        "n_rpm": 3000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 2000,
        "noise_std": 0.0,
    },
]


def find_best_pytorch_model():
    """Find the best existing PyTorch SNN model."""
    base_dir = PROJECT_ROOT / "trained_models"
    best_model = None

    # Heuristic: Look for a folder starting with 'pop' or 'learned' that has best_model.pt
    for p in base_dir.glob("**/best_model.pt"):
        if "snn" in str(p).lower():
            best_model = p
            break  # Just take the first one for now, or refine logic

    return best_model


def plot_scenario(scenario_name, results_map):
    """Plot i_q response for all models in one chart."""
    plt.figure(figsize=(10, 6))

    # Plot Reference (from first result)
    first_res = list(results_map.values())[0]
    time = np.array([d["time"] for d in first_res["episode_data"]])
    ref = np.array([d["i_q_ref"] for d in first_res["episode_data"]])
    plt.plot(time, ref, "k--", label="Reference", linewidth=2, alpha=0.5)

    # Plot Models
    for name, res in results_map.items():
        data = res["episode_data"]
        t = np.array([d["time"] for d in data])
        iq = np.array([d["i_q"] for d in data])

        style = "-"
        alpha = 0.8
        lw = 1.5

        if "PI" in name:
            color = "black"
            alpha = 0.4
            lw = 2
        elif "Akida" in name:
            color = "tab:red"
            lw = 2
        elif "PyTorch" in name:
            color = "tab:blue"
            style = ":"
        else:
            color = None

        plt.plot(
            t, iq, label=name, linestyle=style, color=color, alpha=alpha, linewidth=lw
        )

    plt.title(f"Comparison: {scenario_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Current $i_q$ [A]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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


def main():
    print("=" * 60)
    print("AKIDA vs SNN vs PI BENCHMARK")
    print("=" * 60)

    # Run Scenarios
    for scen in SCENARIOS:
        print(f"\n--- Scenario: {scen['name']} ---")

        results_map = {}

        task = PMSMCurrentControlTask.from_config(
            n_rpm=scen["n_rpm"],
            i_d_ref=scen["i_d_ref"],
            i_q_ref=scen["i_q_ref"],
            max_steps=scen["max_steps"],
        )

        models = []
        models.append(
            (
                "PI_Baseline",
                PIControllerAgent.from_system_config(task.physics_engine.config),
            )
        )
        pt_path = find_best_pytorch_model()
        if pt_path:
            print(f"Found PyTorch SNN: {pt_path.parent.name}")
            models.append(("PyTorch_SNN", SNNControllerAgent(str(pt_path))))
        else:
            print("No PyTorch SNN found.")

        keras_path = PROJECT_ROOT / "trained_models/akida/final_model.keras"
        if keras_path.exists():
            print(f"Found Akida Keras: {keras_path}")
            models.append(("Akida_Float", AkidaControllerAgent(str(keras_path))))
        else:
            print("No Akida Keras model found (train first!).")

        fbz_path = PROJECT_ROOT / "trained_models/akida/akida_model.fbz"
        if fbz_path.exists():
            print(f"Found Akida FBZ: {fbz_path}")
            try:
                models.append(("Akida_Chip_Sim", AkidaControllerAgent(str(fbz_path))))
            except ImportError:
                print("Skipping Akida FBZ (akida package not installed)")

        for name, agent in models:
            print(f"Running {name}...", end="", flush=True)

            # Use run_comprehensive_benchmark to get full metrics
            # Note: We disable 'compute_neuromorphic' for Akida here because
            # our Akida wrapper doesn't implement the specific spike hooks
            # expected by the neurobench-style SNNControllerAgent yet.
            res = run_comprehensive_benchmark(
                agent,
                name,
                task,
                max_steps=scen["max_steps"],
                compute_neuromorphic=(name == "PyTorch_SNN"),
            )

            if res and "result" in res:
                r = res["result"]
                results_map[name] = res
                print(f" RMSE: {r.accuracy.RMSE_iq*1000:.2f} mA")
            else:
                print(" Failed.")

            task.physics_engine.close()

        # Plot for this scenario
        if results_map:
            plot_scenario(scen["name"], results_map)


if __name__ == "__main__":
    main()
