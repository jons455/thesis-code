"""
Example: How to test a custom controller with the benchmark.

This script shows how to wrap ANY controller to work with the benchmark API.

Usage:
    poetry run python scripts/example_custom_controller.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gym_electric_motor as gem  # noqa: E402
import numpy as np  # noqa: E402
from gym_electric_motor.physical_systems import ConstantSpeedLoad  # noqa: E402

from embark.benchmark.agents import PIControllerAgent  # noqa: E402
from embark.benchmark.harness.closed_loop import ClosedLoopHarness  # noqa: E402
from embark.benchmark.physics.pmsm import PMSMPhysicsEngine  # noqa: E402
from embark.benchmark.tasks.pmsm_current_control import (  # noqa: E402
    PMSMCurrentControlTask,
)
from embark.benchmark.tasks.reference_generators import StepGenerator  # noqa: E402


class SimplePController:
    """
    A simple proportional controller for demonstration.

    This shows the MINIMUM you need to implement:
    - __call__(state) -> action
    - reset()

    """

    def __init__(self, kp: float = 5.0):
        self.kp = kp
        # Motor limits (from PMSMEnv defaults)
        self.i_max = 10.8
        self.u_max = 48.0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control action.

        Input:
            state: [i_d, i_q, e_d, e_q] - all normalized to [-1, 1]
                   e_d = (i_d_ref - i_d) / i_max
                   e_q = (i_q_ref - i_q) / i_max

        Output:
            action: [u_d, u_q] - normalized to [-1, 1]

        """
        # Extract normalized errors
        e_d_norm = state[2]
        e_q_norm = state[3]

        # Denormalize to Amps
        e_d = e_d_norm * self.i_max
        e_q = e_q_norm * self.i_max

        # P control: u = Kp * e
        u_d = self.kp * e_d
        u_q = self.kp * e_q

        # Normalize output to [-1, 1]
        u_d_norm = np.clip(u_d / self.u_max, -1.0, 1.0)
        u_q_norm = np.clip(u_q / self.u_max, -1.0, 1.0)

        return np.array([u_d_norm, u_q_norm], dtype=np.float32)

    def reset(self) -> None:
        """Reset controller state (P controller has no state)."""
        pass


# =============================================================================
# EXAMPLE 2: PI Controller from Scratch
# =============================================================================


class SimplePIController:
    """A simple PI controller showing how to handle internal state."""

    def __init__(self, kp: float = 5.0, ki: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.i_max = 10.8
        self.u_max = 48.0
        self.dt = 1e-4  # 10 kHz control

        # Internal state - MUST be reset!
        self.integral_d = 0.0
        self.integral_q = 0.0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        # Extract and denormalize errors
        e_d = state[2] * self.i_max
        e_q = state[3] * self.i_max

        # Update integrals
        self.integral_d += e_d * self.dt
        self.integral_q += e_q * self.dt

        # PI control
        u_d = self.kp * e_d + self.ki * self.integral_d
        u_q = self.kp * e_q + self.ki * self.integral_q

        # Normalize output
        u_d_norm = np.clip(u_d / self.u_max, -1.0, 1.0)
        u_q_norm = np.clip(u_q / self.u_max, -1.0, 1.0)

        return np.array([u_d_norm, u_q_norm], dtype=np.float32)

    def reset(self) -> None:
        """Reset integrator states for new episode."""
        self.integral_d = 0.0
        self.integral_q = 0.0


# =============================================================================
# EXAMPLE 3: Wrapper for External Model (Template)
# =============================================================================


class ExternalModelWrapper:
    """
    Template for wrapping an external model (e.g., from MATLAB, TensorFlow, JAX).

    Modify this to match your model's API.

    """

    def __init__(self, model_path: str):
        # Load your model here
        # self.model = load_my_model(model_path)
        self.model = None  # Placeholder
        self.hidden_state = None

        self.i_max = 10.8
        self.u_max = 48.0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.model is None:
            # Fallback: just return zero action
            return np.zeros(2, dtype=np.float32)

        # Preprocess state for your model
        # input_tensor = self.preprocess(state)

        # Run inference
        # output, self.hidden_state = self.model(input_tensor, self.hidden_state)

        # Postprocess output
        # action = self.postprocess(output)

        return np.zeros(2, dtype=np.float32)  # Placeholder

    def reset(self) -> None:
        self.hidden_state = None


# =============================================================================
# MAIN: Run benchmarks
# =============================================================================


def main():
    print("=" * 60)
    print("Custom Controller Benchmark Examples")
    print("=" * 60)

    # Test configurations
    env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0, max_steps=2000)

    # Test 1: P Controller
    print("\n--- Test 1: Simple P Controller ---")
    p_controller = SimplePController(kp=5.0)
    p_results = run_benchmark(p_controller, env, verbose=True)
    print(f"  RMSE i_q: {p_results.rmse_iq * 1000:.2f} mA")
    print(f"  Stable: {p_results.is_stable}")

    # Reset environment for next test
    env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0, max_steps=2000)

    # Test 2: PI Controller
    print("\n--- Test 2: Simple PI Controller ---")
    pi_controller = SimplePIController(kp=5.0, ki=1000.0)
    pi_results = run_benchmark(pi_controller, env, verbose=True)
    print(f"  RMSE i_q: {pi_results.rmse_iq * 1000:.2f} mA")
    print(f"  Final error: {pi_results.final_error_iq * 1000:.2f} mA")
    print(f"  Stable: {pi_results.is_stable}")

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Controller':<20} {'RMSE [mA]':>12} {'Final Err [mA]':>15} {'Stable':>8}")
    print("-" * 60)
    print(
        f"{'P (Kp=5.0)':<20} {p_results.rmse_iq * 1000:>12.2f} {p_results.final_error_iq * 1000:>15.2f} {'Yes' if p_results.is_stable else 'No':>8}"
    )
    print(
        f"{'PI (Kp=5, Ki=1000)':<20} {pi_results.rmse_iq * 1000:>12.2f} {pi_results.final_error_iq * 1000:>15.2f} {'Yes' if pi_results.is_stable else 'No':>8}"
    )
    print("=" * 60)

    print("\nTo add your own controller, just implement:")
    print("  - __call__(state: np.ndarray) -> np.ndarray")
    print("  - reset() -> None")
    print("\nSee the examples above for templates!")


if __name__ == "__main__":
    main()
