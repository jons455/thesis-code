from embark.benchmark.harness import BenchmarkSuite, STANDARD_SCENARIOS, QUICK_SCENARIOS
from embark.benchmark.tasks.reference_generators import MultiStepReference

print('✓ Successfully imported BenchmarkSuite and scenarios')
print(f'✓ STANDARD_SCENARIOS contains {len(STANDARD_SCENARIOS)} scenarios')
print(f'✓ QUICK_SCENARIOS contains {len(QUICK_SCENARIOS)} scenarios')

# Test MultiStepReference
ref = MultiStepReference(steps=[(0.0, 0.0, 0.0), (0.1, 0.0, 2.0)])
result = ref(0, 0.05)
print(f'✓ MultiStepReference works: {result}')

# Test scenario creation
scenario = STANDARD_SCENARIOS[3]  # multi_step_bidirectional
print(f'✓ Scenario 4 name: {scenario.name}')
print(f'✓ Scenario 4 steps: {scenario.max_steps}')

print('\n✅ All components working correctly!')

# Print all scenario details
print('\nScenario Details:')
for i, s in enumerate(STANDARD_SCENARIOS, 1):
    print(f'{i}. {s.name} ({s.n_rpm} RPM, {s.max_steps} steps)')
