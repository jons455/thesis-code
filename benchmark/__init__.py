"""
NeuroBench Integration Module for PMSM Current Control Benchmark
================================================================

This module provides the interface layer between the GEM (gym-electric-motor)
PMSM simulation and the NeuroBench closed-loop benchmark framework.

Components:
-----------
- PMSMEnv: Gymnasium-compatible wrapper for GEM PMSM environment
- PIControllerAgent: Baseline PI controller as NeuroBench agent
- SNNControllerAgent: Spiking neural network controller

Usage:
------
    from benchmark import PMSMEnv, PIControllerAgent, SNNControllerAgent
    from neurobench.benchmarks import BenchmarkClosedLoop

    env = PMSMEnv()
    agent = PIControllerAgent()
    # or: agent = SNNControllerAgent("snn/checkpoints/best_model.pt")

    benchmark = BenchmarkClosedLoop(agent, env, ...)
    results = benchmark.run()
"""

from .agents import (
    PIControllerAgent,
    PIControllerTorchAgent,
    PIParameters,
    SNNControllerAgent,
    SNNControllerTorchAgent,
)
from .pmsm_env import PMSMEnv

__all__ = [
    "PMSMEnv",
    "PIControllerAgent",
    "PIControllerTorchAgent",
    "PIParameters",
    "SNNControllerAgent",
    "SNNControllerTorchAgent",
]
