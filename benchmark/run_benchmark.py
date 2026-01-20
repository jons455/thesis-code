"""
PMSM Current Control Benchmark Runner
=====================================

Main script to run NeuroBench closed-loop benchmarks for PMSM control.

This script validates the integration by running the PI controller
baseline through the NeuroBench BenchmarkClosedLoop framework.

Usage:
------
    cd benchmark
    python run_benchmark.py

    # Or from project root:
    python -m benchmark.run_benchmark
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import numpy as np
try:
    # When running as module: python -m benchmark.run_benchmark
    from benchmark.agents import PIControllerAgent, PIControllerTorchAgent, SNNControllerAgent
    from benchmark.pmsm_env import PMSMEnv
except ImportError:
    # When running directly from benchmark folder
    from agents import PIControllerAgent, PIControllerTorchAgent, SNNControllerAgent
    from pmsm_env import PMSMEnv

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


def run_snn_test(checkpoint_path: str = "../snn/checkpoints/best_model.pt"):
    """
    Test SNN controller in closed-loop.
    
    Validates that the trained SNN can control the motor without exploding.
    """
    print("\n" + "=" * 60)
    print("SNN Closed-Loop Test")
    print("=" * 60)
    
    from pathlib import Path
    
    # Resolve checkpoint path relative to this file
    checkpoint = Path(__file__).parent / checkpoint_path
    if not checkpoint.exists():
        # Try from project root
        checkpoint = Path(__file__).parent.parent / "snn/checkpoints/best_model.pt"
    
    if not checkpoint.exists():
        print(f"[SKIP] No checkpoint found at {checkpoint}")
        print("Run training first: poetry run python -m snn.train --epochs 100")
        return None
    
    print(f"Loading model from: {checkpoint}")
    
    # Create environment
    env = PMSMEnv(
        n_rpm=1000,
        i_d_ref=0.0,
        i_q_ref=2.0,
        max_steps=500,
    )
    
    try:
        # Create SNN controller
        agent = SNNControllerAgent(str(checkpoint))
        print(f"Model loaded successfully!")
        print(f"  Parameters: {agent.model.count_parameters():,}")
        
    except Exception as e:
        print(f"[FAIL] Could not load SNN model: {e}")
        env.close()
        return False
    
    # Run episode
    state, info = env.reset()
    agent.reset()
    
    total_reward = 0
    sparsities = []
    
    for step in range(500):
        action = agent(state)
        
        # Track sparsity every 100 steps
        if step % 100 == 0:
            sparsity = agent.get_sparsity(state)
            sparsities.append(sparsity)
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Check for NaN/explosion
        if np.isnan(action).any() or np.isnan(state).any():
            print(f"[FAIL] NaN detected at step {step}")
            env.close()
            return False
        
        if done:
            break
    
    # Get episode data
    episode_data = env.get_episode_data()
    
    # Compute metrics
    final_e_d = episode_data[-1]["e_d"]
    final_e_q = episode_data[-1]["e_q"]
    final_error = np.sqrt(final_e_d**2 + final_e_q**2)
    
    # Compute RMSE over episode
    errors = [(d["e_d"]**2 + d["e_q"]**2)**0.5 for d in episode_data]
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    # Average sparsity
    avg_sparsity = {}
    if sparsities:
        for key in sparsities[0].keys():
            avg_sparsity[key] = np.mean([s[key] for s in sparsities])
    
    print("\nSNN Results:")
    print(f"  Steps completed: {len(episode_data)}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final tracking error: {final_error*1000:.2f} mA")
    print(f"  RMSE: {rmse*1000:.2f} mA")
    print(f"  Time in target: {env.time_in_range} steps")
    print(f"  i_d final: {episode_data[-1]['i_d']:.4f} A (ref: {episode_data[-1]['i_d_ref']:.4f} A)")
    print(f"  i_q final: {episode_data[-1]['i_q']:.4f} A (ref: {episode_data[-1]['i_q_ref']:.4f} A)")
    
    if avg_sparsity:
        print("\n  Activation Sparsity (higher = more efficient):")
        for key, val in avg_sparsity.items():
            print(f"    {key}: {val*100:.1f}%")
    
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


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("PMSM Current Control Benchmark - Validation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run PI controller test first (baseline)
    simple_ok = run_simple_test()

    # Run SNN controller test
    snn_ok = run_snn_test()
    
    if simple_ok:
        # Try NeuroBench benchmark (may have compatibility issues)
        run_neurobench_benchmark()

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  PI Controller:  {'[PASS]' if simple_ok else '[FAIL]'}")
    if snn_ok is None:
        print(f"  SNN Controller: [SKIP] (no checkpoint)")
    else:
        print(f"  SNN Controller: {'[PASS]' if snn_ok else '[NEEDS TRAINING]'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
