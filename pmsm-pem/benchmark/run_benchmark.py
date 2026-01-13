"""
PMSM Current Control Benchmark Runner
=====================================

Main script to run NeuroBench closed-loop benchmarks for PMSM control.

This script validates the integration by running the PI controller
baseline through the NeuroBench BenchmarkClosedLoop framework.

Usage:
------
    python -m benchmark.run_benchmark
    
    # Or from pmsm-pem directory:
    python benchmark/run_benchmark.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from datetime import datetime

# NeuroBench imports
from neurobench.benchmarks import BenchmarkClosedLoop
from neurobench.models import TorchAgent
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

# Local imports
from benchmark.pmsm_env import PMSMEnv, make_pmsm_env
from benchmark.agents import PIControllerAgent, PIControllerTorchAgent


def run_simple_test():
    """
    Simple validation test without NeuroBench.
    
    Tests that PMSMEnv and PIControllerAgent work together correctly.
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
    agent = PIControllerAgent()
    
    # Run episode
    state, info = env.reset()
    agent.reset()
    
    total_reward = 0
    for step in range(500):
        action = agent(state)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # Get episode data
    episode_data = env.get_episode_data()
    
    # Compute metrics
    final_e_d = episode_data[-1]['e_d']
    final_e_q = episode_data[-1]['e_q']
    final_error = np.sqrt(final_e_d**2 + final_e_q**2)
    
    print(f"\nResults:")
    print(f"  Steps completed: {len(episode_data)}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final tracking error: {final_error*1000:.2f} mA")
    print(f"  Time in target: {env.time_in_range} steps")
    print(f"  i_d final: {episode_data[-1]['i_d']:.4f} A (ref: {episode_data[-1]['i_d_ref']:.4f} A)")
    print(f"  i_q final: {episode_data[-1]['i_q']:.4f} A (ref: {episode_data[-1]['i_q_ref']:.4f} A)")
    
    env.close()
    
    # Check success
    if final_error < 0.1:  # Less than 100 mA error
        print("\n[OK] Simple test PASSED")
        return True
    else:
        print("\n[FAIL] Simple test FAILED - tracking error too high")
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
        
        print(f"\nNeuroBench Results:")
        print(f"  Average episode time: {avg_time:.4f} s")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"\nNeuroBench benchmark failed: {e}")
        print("This is expected if the environment interface doesn't fully match NeuroBench expectations.")
        print("The simple test above validates that our components work correctly.")
        return False
    
    finally:
        env.close()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("PMSM Current Control Benchmark - Validation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run simple test first
    simple_ok = run_simple_test()
    
    if simple_ok:
        # Try NeuroBench benchmark
        run_neurobench_benchmark()
    
    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

