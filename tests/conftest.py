"""
Shared pytest fixtures for integration and regression tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "benchmark"))
sys.path.insert(0, str(project_root / "metrics"))


@pytest.fixture
def project_root():
    """Return project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_trajectory_data():
    """Generate sample PI controller trajectory data."""
    dt = 1e-4
    time = np.arange(0, 0.1, dt)
    n = len(time)
    
    # Simulate step response
    step_time = 0.01
    tau = 0.003  # 3ms time constant (fast PI)
    
    i_q_ref = np.where(time >= step_time, 2.0, 0.0)
    i_d_ref = np.zeros(n)
    
    i_q = np.where(
        time >= step_time,
        2.0 * (1 - np.exp(-(time - step_time) / tau)),
        0.0
    )
    i_d = np.random.normal(0, 0.01, n)
    
    u_q = 10 * np.ones(n)
    u_d = -2 * np.ones(n)
    speed = 1000 * np.ones(n)
    
    return pd.DataFrame({
        'time': time,
        'i_d': i_d,
        'i_q': i_q,
        'i_d_ref': i_d_ref,
        'i_q_ref': i_q_ref,
        'u_d': u_d,
        'u_q': u_q,
        'n': speed,
    })


@pytest.fixture
def pmsm_env():
    """Create and yield PMSMEnv, then close."""
    from pmsm_env import PMSMEnv
    env = PMSMEnv(max_steps=100)
    yield env
    env.close()


@pytest.fixture
def pi_agent():
    """Create PI controller agent."""
    from agents import PIControllerAgent
    return PIControllerAgent()

