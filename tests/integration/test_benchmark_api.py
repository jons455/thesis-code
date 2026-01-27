"""Test the benchmark API with PI controller."""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embark.benchmark.agents import PIControllerAgent  # noqa: E402
from embark.benchmark.controller_interface import run_benchmark  # noqa: E402
from embark.benchmark.pmsm_env import PMSMEnv  # noqa: E402

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
