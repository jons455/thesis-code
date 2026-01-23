"""Test the benchmark API with PI controller."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.controller_interface import run_benchmark, BenchmarkConfig
from benchmark.pmsm_env import PMSMEnv
from benchmark.agents import PIControllerAgent

print("Testing Benchmark API")
print("=" * 60)

# Create environment
env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0, max_steps=2000)
print(f"Environment: {env.n_rpm} rpm, i_q_ref = {env.i_q_ref} A")

# Create PI controller  
controller = PIControllerAgent()
print(f"Controller: {controller.__class__.__name__}")

# Run benchmark
print("\nRunning benchmark...")
results = run_benchmark(controller, env, verbose=True)

# Print summary
print("\n" + results.summary())

# Close environment
env.close()

print("\n" + "=" * 60)
print("Benchmark API test complete!")
