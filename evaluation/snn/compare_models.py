"""
Compare trained SNN models against PI baseline.
Now includes Symmetry Tests (Positive vs Negative) to verify data balancing.

Usage:
    poetry run python -m evaluation.snn.compare_models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embark.benchmark.pmsm_env import PMSMEnv
from embark.benchmark.agents import PIControllerAgent, SNNControllerAgent
from embark.benchmark.run_benchmark import run_comprehensive_benchmark

# --- UPDATED SCENARIOS ---
SCENARIOS = [
    {
        "name": "A_Nominal_Pos",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,  # Step to +2A
        "max_steps": 2000, # Shorter is fine for step response
        "noise_std": 0.0,
        "desc": "Step Response (0 -> 2A)"
    },
    {
        "name": "B_Nominal_Neg",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": -2.0, # Step to -2A (CRITICAL TEST for new data)
        "max_steps": 2000,
        "noise_std": 0.0,
        "desc": "Negative Step (0 -> -2A)"
    },
    {
        "name": "C_HighSpeed",
        "n_rpm": 3000,   # High speed test
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 2000,
        "noise_std": 0.0,
        "desc": "High Speed (3000 rpm)"
    }
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
    if not results_list: return

    plot_dir = PROJECT_ROOT / "docs" / "plots" / "comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Reference (dashed black)
    first_data = results_list[0][1]
    time = np.array([d["time"] for d in first_data])
    ref = np.array([d["i_q_ref"] for d in first_data])
    plt.plot(time, ref, 'k--', label='Reference', linewidth=2, alpha=0.7)
    
    # Models
    for model_name, episode_data in results_list:
        t = np.array([d["time"] for d in episode_data])
        iq = np.array([d["i_q"] for d in episode_data])
        
        # Style logic
        lw = 2.0 if "PI" in model_name else 1.5
        alpha = 0.9 if "PI" in model_name or "delta" in model_name else 0.7
        
        plt.plot(t, iq, label=model_name, linewidth=lw, alpha=alpha)
            
    plt.title(f'Scenario: {scenario_name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Current $i_q$ [A]')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = plot_dir / f"compare_{scenario_name}.png"
    plt.savefig(save_path, dpi=150)
    print(f"    Saved plot: {save_path}")
    plt.close()

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
            
            # Setup Environment
            env = PMSMEnv(
                n_rpm=scen["n_rpm"],
                i_d_ref=scen["i_d_ref"],
                i_q_ref=scen["i_q_ref"],
                max_steps=scen["max_steps"],
                measurement_noise_std=scen["noise_std"]
            )
            
            # Setup Agent
            if model_name == "PI_Baseline":
                agent = PIControllerAgent()
                is_neuromorphic = False
            else:
                # IMPORTANT: SNNControllerAgent must handle the Gain internally!
                agent = SNNControllerAgent(str(model_path), track_spikes=True)
                is_neuromorphic = True
            
            # Run Benchmark
            res = run_comprehensive_benchmark(
                agent, 
                model_name, 
                env, 
                max_steps=scen["max_steps"], 
                compute_neuromorphic=is_neuromorphic
            )
            env.close()
            
            if res and "result" in res:
                r = res["result"]
                # Store for plotting
                if "episode_data" in res:
                    plot_data_storage[scen["name"]].append((model_name, res["episode_data"]))
                
                full_results.append({
                    "Model": model_name,
                    "Scenario": scen["name"],
                    "RMSE": r.accuracy.RMSE_iq,
                    "ITAE": r.accuracy.ITAE_iq,
                    "TV": r.stability.TV_total,
                    "SyOps": r.neuromorphic.syops_per_timestep if r.neuromorphic else 0.0,
                    "Sparsity": r.neuromorphic.activation_sparsity if r.neuromorphic else 0.0,
                    "LAC": r.lac_score
                })
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
        print("\n" + "="*60)
        print("FINAL LEADERBOARD (Sorted by RMSE)")
        print("="*60)
        
        # Average RMSE across scenarios
        leaderboard = df.groupby("Model")[["RMSE", "Sparsity"]].mean().sort_values("RMSE")
        print(leaderboard)
        
        # Save CSV
        df.to_csv("docs/final_benchmark_results.csv", index=False)
        print("\nDetailed results saved to docs/final_benchmark_results.csv")

if __name__ == "__main__":
    run_evaluation()