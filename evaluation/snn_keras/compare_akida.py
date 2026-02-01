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

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embark.benchmark.pmsm_env import PMSMEnv
from embark.benchmark.agents import PIControllerAgent, SNNControllerAgent
from embark.benchmark.run_benchmark import run_comprehensive_benchmark
from evaluation.snn_keras.akida_agent import AkidaControllerAgent

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
    }
]

def find_best_pytorch_model():
    """Find the best existing PyTorch SNN model."""
    base_dir = PROJECT_ROOT / "trained_models"
    best_model = None
    
    # Heuristic: Look for a folder starting with 'pop' or 'learned' that has best_model.pt
    for p in base_dir.glob("**/best_model.pt"):
        if "snn" in str(p).lower():
            best_model = p
            break # Just take the first one for now, or refine logic
            
    return best_model

def plot_scenario(scenario_name, results_map):
    """Plot i_q response for all models in one chart."""
    plt.figure(figsize=(10, 6))
    
    # Plot Reference (from first result)
    first_res = list(results_map.values())[0]
    time = np.array([d["time"] for d in first_res["episode_data"]])
    ref = np.array([d["i_q_ref"] for d in first_res["episode_data"]])
    plt.plot(time, ref, 'k--', label="Reference", linewidth=2, alpha=0.5)
    
    # Plot Models
    for name, res in results_map.items():
        data = res["episode_data"]
        t = np.array([d["time"] for d in data])
        iq = np.array([d["i_q"] for d in data])
        
        style = '-'
        alpha = 0.8
        lw = 1.5
        
        if "PI" in name:
            color = 'black'
            alpha = 0.4
            lw = 2
        elif "Akida" in name:
            color = 'tab:red'
            lw = 2
        elif "PyTorch" in name:
            color = 'tab:blue'
            style = ':'
        else:
            color = None
            
        plt.plot(t, iq, label=name, linestyle=style, color=color, alpha=alpha, linewidth=lw)
        
    plt.title(f"Comparison: {scenario_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Current $i_q$ [A]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("AKIDA vs SNN vs PI BENCHMARK")
    print("="*60)
    
    models = []
    
    # 1. PI Baseline
    models.append(("PI_Baseline", PIControllerAgent()))
    
    # 2. PyTorch SNN (if found)
    pt_path = find_best_pytorch_model()
    if pt_path:
        print(f"Found PyTorch SNN: {pt_path.parent.name}")
        models.append(("PyTorch_SNN", SNNControllerAgent(str(pt_path))))
    else:
        print("No PyTorch SNN found.")
        
    # 3. Akida Keras (Float)
    keras_path = PROJECT_ROOT / "trained_models/akida/final_model.keras"
    if keras_path.exists():
        print(f"Found Akida Keras: {keras_path}")
        models.append(("Akida_Float", AkidaControllerAgent(str(keras_path))))
    else:
        print("No Akida Keras model found (train first!).")

    # 4. Akida Hardware (.fbz)
    fbz_path = PROJECT_ROOT / "trained_models/akida/akida_model.fbz"
    if fbz_path.exists():
        print(f"Found Akida FBZ: {fbz_path}")
        try:
            models.append(("Akida_Chip_Sim", AkidaControllerAgent(str(fbz_path))))
        except ImportError:
            print("Skipping Akida FBZ (akida package not installed)")
    
    if len(models) <= 1:
        print("Not enough models to compare. Exiting.")
        return

    # Run Scenarios
    for scen in SCENARIOS:
        print(f"\n--- Scenario: {scen['name']} ---")
        
        results_map = {}
        
        for name, agent in models:
            print(f"Running {name}...", end="", flush=True)
            
            env = PMSMEnv(
                n_rpm=scen["n_rpm"],
                i_d_ref=scen["i_d_ref"],
                i_q_ref=scen["i_q_ref"],
                max_steps=scen["max_steps"],
                measurement_noise_std=scen["noise_std"]
            )
            
            # Use run_comprehensive_benchmark to get full metrics
            # Note: We disable 'compute_neuromorphic' for Akida here because
            # our Akida wrapper doesn't implement the specific spike hooks 
            # expected by the neurobench-style SNNControllerAgent yet.
            res = run_comprehensive_benchmark(
                agent,
                name,
                env,
                max_steps=scen["max_steps"],
                compute_neuromorphic=(name == "PyTorch_SNN")
            )
            
            if res and "result" in res:
                r = res["result"]
                results_map[name] = res
                print(f" RMSE: {r.accuracy.RMSE_iq*1000:.2f} mA")
            else:
                print(" Failed.")
            
            env.close()
            
        # Plot for this scenario
        if results_map:
            plot_scenario(scen["name"], results_map)

if __name__ == "__main__":
    main()
