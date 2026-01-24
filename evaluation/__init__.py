"""
Evaluation Module for SNN Controller Benchmark
===============================================

This module provides evaluation utilities for testing trained SNN controllers
against the PMSM current control benchmark. It serves as an example of how
external users can integrate their controllers with the benchmark framework.

Components:
-----------
- SNNBenchmarkController: Benchmark-compatible wrapper for trained SNN models
- run_evaluation: Script for running benchmark comparisons

Example:
--------
    from evaluation import SNNBenchmarkController
    from benchmark import PMSMEnv
    from benchmark.controller_interface import run_benchmark

    # Load trained SNN controller
    controller = SNNBenchmarkController("evaluation/models/best_model.pt")

    # Run benchmark
    env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0)
    results = run_benchmark(controller, env)
    print(results.summary())
"""

from benchmark.agents import SNNControllerAgent as SNNBenchmarkController

__all__ = [
    "SNNBenchmarkController",
]
