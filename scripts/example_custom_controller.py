"""
Example: How to test a custom controller with the benchmark.

This script shows how to wrap ANY controller to work with the benchmark API.

Usage:
    poetry run python scripts/example_custom_controller.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.benchmark.harness.closed_loop import ClosedLoopHarness  # noqa: E402
from embark.benchmark.interfaces import (  # noqa: E402
    ActionDict,
    DictController,
    ReferenceDict,
    StateDict,
)
from embark.benchmark.tasks.pmsm_current_control import (  # noqa: E402
    PMSMCurrentControlTask,
)


class SimplePController(DictController):
    """
    A simple proportional controller for demonstration.

    This shows the MINIMUM you need to implement:
    - __call__(state, reference) -> action
    - reset()

    """

    def __init__(self, kp: float = 5.0):
        self.kp = kp
        # Motor limits (from PMSMEnv defaults)
        self.i_max = 20.0  # Approx max current
        self.u_max = 300.0  # Approx max voltage (DC link)

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        """
        Compute control action.

        Input:
            state: {"i_d": ..., "i_q": ...} (Physical units)
            reference: {"i_d_ref": ..., "i_q_ref": ...} (Physical units)

        Output:
            action: {"v_d": ..., "v_q": ...} (Volts)

        """
        # Extract errors
        i_d = state.get("i_d", 0.0)
        i_q = state.get("i_q", 0.0)
        i_d_ref = reference.get("i_d_ref", 0.0)
        i_q_ref = reference.get("i_q_ref", 0.0)

        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # P control: u = Kp * e
        u_d = self.kp * e_d
        u_q = self.kp * e_q

        # Simple clamping (optional, physics will also clamp)
        u_d = np.clip(u_d, -self.u_max, self.u_max)
        u_q = np.clip(u_q, -self.u_max, self.u_max)

        return {"v_d": float(u_d), "v_q": float(u_q)}

    def reset(self) -> None:
        """Reset controller state (P controller has no state)."""
        pass

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass


# =============================================================================
# EXAMPLE 2: PI Controller from Scratch
# =============================================================================


class SimplePIController(DictController):
    """A simple PI controller showing how to handle internal state."""

    def __init__(self, kp: float = 5.0, ki: float = 100.0, dt: float = 1e-4):
        self.kp = kp
        self.ki = ki
        self.dt = dt  # 10 kHz control

        # Internal state - MUST be reset!
        self.integral_d = 0.0
        self.integral_q = 0.0

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        # Extract errors
        i_d = state.get("i_d", 0.0)
        i_q = state.get("i_q", 0.0)
        i_d_ref = reference.get("i_d_ref", 0.0)
        i_q_ref = reference.get("i_q_ref", 0.0)

        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # Update integrals
        self.integral_d += e_d * self.dt
        self.integral_q += e_q * self.dt

        # PI control
        u_d = self.kp * e_d + self.ki * self.integral_d
        u_q = self.kp * e_q + self.ki * self.integral_q

        return {"v_d": float(u_d), "v_q": float(u_q)}

    def reset(self) -> None:
        """Reset integrator states for new episode."""
        self.integral_d = 0.0
        self.integral_q = 0.0

    def get_state(self) -> dict[str, Any]:
        return {"int_d": self.integral_d, "int_q": self.integral_q}

    def set_state(self, state: dict[str, Any]) -> None:
        self.integral_d = state.get("int_d", 0.0)
        self.integral_q = state.get("int_q", 0.0)


# =============================================================================
# EXAMPLE 3: Wrapper for External Model (Template)
# =============================================================================


class ExternalModelWrapper(DictController):
    """
    Template for wrapping an external model (e.g., from MATLAB, TensorFlow, JAX).

    Modify this to match your model's API.

    """

    def __init__(self, model_path: str):
        # Load your model here
        # self.model = load_my_model(model_path)
        self.model = None  # Placeholder
        self.hidden_state = None

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        if self.model is None:
            # Fallback: just return zero action
            return {"v_d": 0.0, "v_q": 0.0}

        # 1. Preprocess: Extract features from dicts
        # input_vector = [state['i_d'], state['i_q'], ...]

        # 2. Run inference
        # output, self.hidden_state = self.model(input_vector, self.hidden_state)

        # 3. Postprocess: Convert to voltage dict
        # return {"v_d": output[0], "v_q": output[1]}

        return {"v_d": 0.0, "v_q": 0.0}  # Placeholder

    def reset(self) -> None:
        self.hidden_state = None

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass


# =============================================================================
# MAIN: Run benchmarks
# =============================================================================


def main():
    print("=" * 60)
    print("Custom Controller Benchmark Examples")
    print("=" * 60)

    # 1. Create Task
    task = PMSMCurrentControlTask.from_config(
        n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0, max_steps=2000
    )

    # Test 1: P Controller
    print("\n--- Test 1: Simple P Controller ---")
    p_controller = SimplePController(kp=5.0)
    harness = ClosedLoopHarness(task=task, controller=p_controller)
    results = harness.run()
    print(f"  Steps: {results['steps']}")
    # Note: ClosedLoopHarness returns aggregated metrics if any were passed.
    # Since we passed no metrics, we just see steps.
    # To see error, we would need to add metrics to the harness.

    # Test 2: PI Controller
    print("\n--- Test 2: Simple PI Controller ---")
    pi_controller = SimplePIController(kp=5.0, ki=1000.0)
    harness = ClosedLoopHarness(task=task, controller=pi_controller)
    results = harness.run()
    print(f"  Steps: {results['steps']}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("To add your own controller, just implement DictController:")
    print("  - __call__(state: dict, reference: dict) -> dict")
    print("  - reset() -> None")
    print("\nSee the examples above for templates!")


if __name__ == "__main__":
    main()
