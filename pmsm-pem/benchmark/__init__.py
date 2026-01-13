"""
NeuroBench Integration Module for PMSM Current Control Benchmark
================================================================

This module provides the interface layer between the GEM (gym-electric-motor)
PMSM simulation and the NeuroBench closed-loop benchmark framework.

Components:
-----------
- PMSMEnv: Gymnasium-compatible wrapper for GEM PMSM environment
- PIControllerAgent: Baseline PI controller as NeuroBench agent
- SNNControllerAgent: Spiking neural network controller (future)

Usage:
------
    from benchmark import PMSMEnv, PIControllerAgent
    from neurobench.benchmarks import BenchmarkClosedLoop
    
    env = PMSMEnv()
    agent = PIControllerAgent(env)
    
    benchmark = BenchmarkClosedLoop(agent, env, ...)
    results = benchmark.run()
"""

from .pmsm_env import PMSMEnv
from .agents import PIControllerAgent

__all__ = [
    'PMSMEnv',
    'PIControllerAgent',
]

