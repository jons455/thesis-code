"""Profile the benchmark to find bottlenecks."""
import time
import torch
from pathlib import Path
from tests.test_v10_end_to_end import load_v10_model, create_v10_controller
from embark.benchmark.harness import STANDARD_SCENARIOS

# Load model
print("Loading model...")
model = load_v10_model(Path('tests/model/best_model.pt'), 'cpu')
model.eval()

# Profile single inference
print("\n1. Single inference speed:")
x = torch.randn(1, 13)
with torch.no_grad():
    # Warmup
    for _ in range(10):
        model(x, deterministic=True)
    
    # Time
    start = time.time()
    for _ in range(100):
        model(x, deterministic=True)
    elapsed = time.time() - start
    
print(f"   100 inferences: {elapsed:.2f}s = {elapsed/100*1000:.1f}ms per inference")

# Profile scenario
print("\n2. Single scenario breakdown:")
controller = create_v10_controller(Path('tests/model/best_model.pt'), 'cpu')
scenario = STANDARD_SCENARIOS[1]  # Mid speed

print(f"   Scenario: {scenario.name}")
print(f"   Steps: {scenario.max_steps}")
print(f"   Expected time: {scenario.max_steps * elapsed/100:.1f}s at current speed")

# Estimate full benchmark
total_steps = sum(s.max_steps for s in STANDARD_SCENARIOS)
print(f"\n3. Full benchmark estimate:")
print(f"   Total steps: {total_steps:,}")
print(f"   Estimated time: {total_steps * elapsed/100:.1f}s = {total_steps * elapsed/100/60:.1f} minutes")

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"\n4. CUDA available: YES - could run {torch.cuda.get_device_name(0)}")
    print("   Re-run with --device cuda for 10-50x speedup!")
else:
    print("\n4. CUDA available: NO - running on CPU")
    print("   This is why it's slow (48 rate steps per inference on CPU)")

