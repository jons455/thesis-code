"""
Benchmark Metrics Module for Neuromorphic PMSM Controller Evaluation
=====================================================================

This module provides a comprehensive metrics framework for evaluating
both conventional and neuromorphic controllers for PMSM current control.

Usage:
------
    from metrics import run_benchmark, BenchmarkResult, NeuromorphicMetrics
    
    # Load simulation data
    df = pd.read_csv('simulation.csv')
    
    # Run benchmark
    result = run_benchmark(df, controller_name="My Controller")
    print(result.summary())

Metric Categories:
------------------
1. AccuracyMetrics: Tracking errors (ITAE, MAE, RMSE)
2. DynamicsMetrics: Step response (rise time, settling time, overshoot)
3. EfficiencyMetrics: Energy efficiency (losses, power, efficiency)
4. SafetyMetrics: Constraint violations (current, voltage limits)
5. NeuromorphicMetrics: SNN-specific (SyOps, sparsity, latency, energy)
"""

from .benchmark_metrics import (
    # Data classes
    PMSMParameters,
    AccuracyMetrics,
    DynamicsMetrics,
    EfficiencyMetrics,
    SafetyMetrics,
    NeuromorphicMetrics,
    BenchmarkResult,
    
    # Computation functions
    compute_accuracy_metrics,
    compute_dynamics_metrics,
    compute_efficiency_metrics,
    compute_safety_metrics,
    compute_neuromorphic_metrics_from_spikes,
    
    # High-level functions
    run_benchmark,
    compare_controllers,
    
    # Constants
    DEFAULT_MOTOR,
    MetricCategory,
)

__all__ = [
    'PMSMParameters',
    'AccuracyMetrics',
    'DynamicsMetrics',
    'EfficiencyMetrics',
    'SafetyMetrics',
    'NeuromorphicMetrics',
    'BenchmarkResult',
    'compute_accuracy_metrics',
    'compute_dynamics_metrics',
    'compute_efficiency_metrics',
    'compute_safety_metrics',
    'compute_neuromorphic_metrics_from_spikes',
    'run_benchmark',
    'compare_controllers',
    'DEFAULT_MOTOR',
    'MetricCategory',
]
