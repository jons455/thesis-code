# EMBARK Documentation Index

Welcome to the EMBARK (Efficient Motor Benchmark for Adaptive Rate-encoding and Key metrics) documentation.

## Documentation layout

| Folder | Contents |
|--------|----------|
| **[benchmark/](benchmark/)** | API reference, user guide, quick references, scenarios, validation template, plan |
| **[scenarios/](scenarios/)** | Scenario timelines and scientific rationale |
| **[analysis/](analysis/)** | PWM, normalization, and SNN controller analysis |
| **[reference/](reference/)** | Metrics, error handling, future work |

---

## Getting Started

### Quick Start
- **[README.md](../README.md)** - Main project overview and quick start guide
  - Installation instructions
  - Basic usage example
  - Feature configuration guide
  - Quick comparison with PI baseline

### First Steps
1. Read the [README](../README.md) for installation and basic usage
2. Review [BENCHMARK_SUITE_QUICK_REFERENCE.md](benchmark/BENCHMARK_SUITE_QUICK_REFERENCE.md) for scenario overview
3. Run your first benchmark using the quick start example

---

## Core Documentation

### API Documentation
- **[BENCHMARK_API.md](benchmark/BENCHMARK_API.md)** - Complete API reference
  - Task interface
  - Controller interface
  - Metric interface
  - Harness usage

### Rate-SNN Specific Guide
- **[RATE_SNN_BENCHMARK_INTERFACE.md](benchmark/RATE_SNN_BENCHMARK_INTERFACE.md)** - Rate-encoding SNN integration guide
  - State processor configuration
  - Action processor configuration
  - Feature engineering
  - Output mode selection

---

## Benchmark Scenarios (NEW)

### Quick Reference
- **[BENCHMARK_SUITE_QUICK_REFERENCE.md](benchmark/BENCHMARK_SUITE_QUICK_REFERENCE.md)** - Quick reference guide
  - Scenario summary table
  - Usage examples
  - Performance targets
  - Interpretation tips

### Comprehensive Guide
- **[BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md)** - Detailed scenario specifications
  - Design philosophy
  - Scenario specifications (all 6 scenarios)
  - Coverage matrix
  - Metric computation
  - Interpretation guide
  - Best practices

### Visual Reference
- **[SCENARIO_TIMELINES.md](scenarios/SCENARIO_TIMELINES.md)** - Visual timeline diagrams
  - ASCII timeline diagrams for each scenario
  - Operating point coverage visualization
  - Difficulty ranking
  - Design rationale

### Scientific Background
- **[SCENARIO_SCIENTIFIC_RATIONALE.md](scenarios/SCENARIO_SCIENTIFIC_RATIONALE.md)** - Research-backed rationale
  - Literature references
  - Design principles
  - Coverage analysis
  - What was excluded and why
  - Validation against alternatives

---

## Documentation Map

### For Different User Types

#### **New Users** (First Time Using EMBARK)
1. [README.md](../README.md) - Installation and quick start
2. [BENCHMARK_SUITE_QUICK_REFERENCE.md](benchmark/BENCHMARK_SUITE_QUICK_REFERENCE.md) - Scenario overview
3. [RATE_SNN_BENCHMARK_INTERFACE.md](benchmark/RATE_SNN_BENCHMARK_INTERFACE.md) - Configure your SNN

#### **Developers** (Implementing Controllers)
1. [BENCHMARK_API.md](benchmark/BENCHMARK_API.md) - API reference
2. [RATE_SNN_BENCHMARK_INTERFACE.md](benchmark/RATE_SNN_BENCHMARK_INTERFACE.md) - SNN integration
3. [BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md) - Understand what you're tested on

#### **Researchers** (Publishing Results)
1. [BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md) - Scenario specifications
2. [SCENARIO_SCIENTIFIC_RATIONALE.md](scenarios/SCENARIO_SCIENTIFIC_RATIONALE.md) - Scientific background
3. [SCENARIO_TIMELINES.md](scenarios/SCENARIO_TIMELINES.md) - Visualization for papers

#### **Contributors** (Extending EMBARK)
1. [BENCHMARK_API.md](benchmark/BENCHMARK_API.md) - Architecture overview
2. [BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md) - Design patterns
3. [SCENARIO_SCIENTIFIC_RATIONALE.md](scenarios/SCENARIO_SCIENTIFIC_RATIONALE.md) - Extension guidelines

---

## Key Concepts

### The 6 Standard Scenarios

| # | Name | Purpose | Key Insight |
|---|------|---------|-------------|
| 1 | Low Speed (500 RPM) | Low-speed sensitivity | Parameter robustness |
| 2 | **Mid Speed (1500 RPM)** ⭐ | **Primary reference** | **Nominal performance** |
| 3 | High Speed (2500 RPM) | High-speed limits | Voltage saturation |
| 4 | Multi-Step Bidirectional | Dynamic tracking | Consistency, memory effects |
| 5 | Four-Quadrant Transition | Torque reversal | Regenerative braking |
| 6 | Field-Weakening | Multivariable control | d-q decoupling |

⭐ **Scenario 2 is your primary reference** - use this for headline numbers.

### Design Philosophy

The benchmark suite provides:
- **Complete coverage** with minimum redundancy
- **Fixed-speed operation** (constant RPM per scenario)
- **Current control focus** (FOC inner loop)
- **Rate-based SNN evaluation** vs PI baseline
- **Manageable runtime** (~3 seconds total)

### Coverage Summary

✅ **Speed range**: 500, 1500, 2500 RPM  
✅ **Transients**: Single-step, multi-step, reversal  
✅ **Quadrants**: Motoring, generating, zero-crossing  
✅ **Advanced**: Field-weakening, d-q coupling  
✅ **Runtime**: ~5-10 seconds per controller

---

## Quick Links

### Usage Examples
```python
# Full benchmark
from embark.benchmark.harness import BenchmarkSuite
suite = BenchmarkSuite()
summary = suite.run(controller=my_controller, name="MySNN-v1")
suite.print_summary(summary)
```

### Key Files
- `embark/benchmark/harness/benchmark_suite.py` - Main suite implementation
- `embark/benchmark/tasks/reference_generators.py` - Reference signal generators
- `embark/benchmark/processors/rate_snn.py` - Rate-SNN processors

### Important Constants
- **Sampling time**: 100 µs (10 kHz)
- **Settling threshold**: ±5% of reference
- **Total scenarios**: 6 standard + 2 quick
- **Total runtime**: ~3.1 seconds simulated time

---

## FAQ

### How do I choose which scenarios to run?

**Development**: Use `QUICK_SCENARIOS` (2 scenarios, ~0.5s)
```python
from embark.benchmark.harness import QUICK_SCENARIOS
suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
```

**Validation**: Use `STANDARD_SCENARIOS` (6 scenarios, ~3s)
```python
from embark.benchmark.harness import STANDARD_SCENARIOS
suite = BenchmarkSuite(scenarios=STANDARD_SCENARIOS)  # or just BenchmarkSuite()
```

**Publication**: Use `STANDARD_SCENARIOS` + multiple runs
```python
results = [suite.run(controller, name=f"MySNN-seed{i}") for i in range(5)]
```

### How do I interpret the results?

See [BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md#interpretation-guide) for detailed guidance.

**Quick check**:
- ✅ MAE within 10% of PI baseline
- ✅ Settling time within 20% of PI
- ✅ Overshoot < 10%
- ✅ Zero safety violations

### How do I add custom scenarios?

See [BENCHMARK_SUITE_QUICK_REFERENCE.md](benchmark/BENCHMARK_SUITE_QUICK_REFERENCE.md#custom-scenarios) or [BENCHMARK_SCENARIOS.md](benchmark/BENCHMARK_SCENARIOS.md#extending-the-suite).

### What's the difference between the documentation files?

| File | Audience | Content | Length |
|------|----------|---------|--------|
| [README.md](../README.md) | Everyone | Quick start, basic usage | Medium |
| [QUICK_REFERENCE](benchmark/BENCHMARK_SUITE_QUICK_REFERENCE.md) | Developers | Usage patterns, quick lookup | Short |
| [BENCHMARK_SCENARIOS](benchmark/BENCHMARK_SCENARIOS.md) | Researchers | Complete specifications | Long |
| [TIMELINES](scenarios/SCENARIO_TIMELINES.md) | Visual learners | Diagrams, visualizations | Medium |
| [SCIENTIFIC_RATIONALE](scenarios/SCENARIO_SCIENTIFIC_RATIONALE.md) | Academics | Research background | Long |

---

## Version History

### v2.0 (Current)
- Updated to 6 optimal scenarios based on motor control best practices
- Added comprehensive documentation suite
- Added `MultiStepReference` generator
- Improved scenario coverage and reduced redundancy

### v1.0
- Initial release with 6 scenarios
- Basic documentation

---

## Contributing

See the main [README.md](../README.md#contributing) for contribution guidelines.

When adding new scenarios or features:
1. Follow existing patterns
2. Add tests
3. Update documentation
4. Consider scientific rationale

---

## Citations

If you use EMBARK in your research, please cite:

```bibtex
@software{embark2025,
  title={EMBARK: Efficient Motor Benchmark for Adaptive Rate-encoding and Key metrics},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/embark}
}
```

### Research References

The benchmark suite is based on:
1. [Neuromorphic Motor Control Benchmarking](https://arxiv.org/html/2512.06603v1) - ArXiv, 2024
2. [Transient Performance Evaluation](https://arxiv.org/html/2402.01782v1) - ArXiv, 2024
3. [Regenerative Braking in PMSM Systems](https://www.nature.com/articles/s41598-025-02396-y) - Nature, 2025
4. [Field-Oriented Control and d-q Decoupling](https://www.nature.com/articles/s41598-025-19384-x) - Nature, 2025

---

## Contact and Support

- **Issues**: Open an issue on GitHub
- **Questions**: See the [FAQ](#faq) or the detailed documentation
- **Contributions**: Follow the [contribution guidelines](../README.md#contributing)

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
