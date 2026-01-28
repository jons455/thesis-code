"""
Compare all trained SNN models against the PI baseline on the Holy Trinity benchmarks.

This script:
1. Detects all available SNN checkpoints in `trained_models/`
2. Runs them through the 3 standardized scenarios (Nominal, High Speed, Robustness)
3. Compares against the PI controller baseline
4. Generates a markdown report with the winner

Usage:
    poetry run python -m evaluation.snn.compare_models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embark.benchmark.pmsm_env import PMSMEnv
from embark.benchmark.agents import PIControllerAgent, SNNControllerAgent
from embark.benchmark.run_benchmark import run_comprehensive_benchmark
from embark.utils.paths import MODELS_CHECKPOINTS_DIR

# The Holy Trinity Scenarios
SCENARIOS = [
    {
        "name": "A_Nominal",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 10000,
        "noise_std": 0.0,
        "desc": "Baseline (1000 rpm, 2A)"
    },
    {
        "name": "B_HighSpeed",
        "n_rpm": 3000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 10000,
        "noise_std": 0.0,
        "desc": "High Speed (3000 rpm)"
    },
    {
        "name": "C_Robustness",
        "n_rpm": 1000,
        "i_d_ref": 0.0,
        "i_q_ref": 2.0,
        "max_steps": 10000,
        "noise_std": 0.05,
        "desc": "Noisy (σ=0.05A)"
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
            
            if best.exists():
                models.append((f"SNN_{subdir.name}", best))
            elif final.exists():
                models.append((f"SNN_{subdir.name}", final))
                
    return sorted(models)

def run_evaluation():
    print("=" * 80)
    print("SNN MODEL COMPARISON REPORT")
    print("=" * 80)

    models = find_models()
    if not models:
        print("No SNN models found in trained_models/!")
        return

    print(f"Found {len(models)} models: {[m[0] for m in models]}")
    
    results = []

    # 1. Run Baseline (PI)
    print("\nRunning PI Baseline...")
    for scenario in SCENARIOS:
        print(f"  Scenario {scenario['name']}...", end="", flush=True)
        env = PMSMEnv(
            n_rpm=scenario["n_rpm"],
            i_d_ref=scenario["i_d_ref"],
            i_q_ref=scenario["i_q_ref"],
            max_steps=scenario["max_steps"],
            measurement_noise_std=scenario["noise_std"]
        )
        
        agent = PIControllerAgent()
        res = run_comprehensive_benchmark(agent, "PI_Baseline", env, max_steps=scenario["max_steps"], compute_neuromorphic=False)
        env.close()
        
        if res and "result" in res:
            r = res["result"]
            results.append({
                "Model": "PI_Baseline",
                "Scenario": scenario["name"],
                "RMSE": r.accuracy.RMSE_iq,
                "ITAE": r.accuracy.ITAE_iq,
                "TV": r.stability.TV_total,
                "SyOps": 0.0,
                "Sparsity": 1.0,  # 100% sparse (no spikes)
                "LAC": 0.0  # Not applicable
            })
            print(" Done.")
        else:
            print(" Failed.")

    # 2. Run SNN Models
    for model_name, model_path in models:
        print(f"\nRunning {model_name}...")
        try:
            # Load agent once to check if valid
            # (We reload per scenario to ensure clean state, but could reset)
            
            for scenario in SCENARIOS:
                print(f"  Scenario {scenario['name']}...", end="", flush=True)
                
                env = PMSMEnv(
                    n_rpm=scenario["n_rpm"],
                    i_d_ref=scenario["i_d_ref"],
                    i_q_ref=scenario["i_q_ref"],
                    max_steps=scenario["max_steps"],
                    measurement_noise_std=scenario["noise_std"]
                )
                
                # Instantiate agent
                agent = SNNControllerAgent(str(model_path), track_spikes=True)
                
                res = run_comprehensive_benchmark(agent, model_name, env, max_steps=scenario["max_steps"], compute_neuromorphic=True)
                env.close()
                
                if res and "result" in res:
                    r = res["result"]
                    results.append({
                        "Model": model_name,
                        "Scenario": scenario["name"],
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
                    
        except Exception as e:
            print(f"  Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Generate Report
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results collected.")
        return

    # Pivot table for RMSE
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    # Calculate average performance across scenarios
    avg_df = df.groupby("Model")[["RMSE", "LAC", "SyOps", "Sparsity"]].mean().sort_values("LAC")
    
    print("\nRanking (by LAC Score - Lower is Better):")
    print(avg_df.to_string())
    
    best_model = avg_df.index[0]
    # Filter out PI for best SNN
    snn_df = avg_df[avg_df.index != "PI_Baseline"]
    best_snn = snn_df.index[0] if not snn_df.empty else "None"
    
    # Save detailed report
    report_path = PROJECT_ROOT / "docs" / "MODEL_COMPARISON_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SNN Model Comparison Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Best Overall SNN:** {best_snn}\n\n")
        
        f.write("## 1. Leaderboard (Average across 3 Scenarios)\n\n")
        f.write("| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |\n")
        f.write("|---|---|---|---|---|\n")
        for model, row in avg_df.iterrows():
            f.write(f"| {model} | {row['LAC']:.4f} | {row['RMSE']:.4f} | {row['SyOps']:.1f} | {row['Sparsity']*100:.1f}% |\n")
            
        f.write("\n## 2. Detailed Results by Scenario\n\n")
        
        for scenario in SCENARIOS:
            s_name = scenario["name"]
            f.write(f"### Scenario: {s_name} ({scenario['desc']})\n\n")
            sub_df = df[df["Scenario"] == s_name].sort_values("RMSE")
            f.write(sub_df[["Model", "RMSE", "ITAE", "TV", "SyOps", "Sparsity", "LAC"]].to_string(index=False))
            f.write("\n\n")
            
    print(f"\nReport saved to: {report_path}")
    print(f"Winner: {best_snn}")

if __name__ == "__main__":
    run_evaluation()
