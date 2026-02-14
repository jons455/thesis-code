"""
Shared pytest fixtures for the benchmark test suite.

Provides reusable test doubles (dummy physics, tasks, controllers) that conform to the
embark benchmark protocols.  These fixtures avoid importing heavy dependencies (GEM,
snntorch) and enable fast, deterministic unit tests.

"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Dummy system config
# ---------------------------------------------------------------------------


@dataclass
class DummyConfig:
    """Minimal SystemConfig for testing."""

    i_max: float = 1.0
    u_max: float = 1.0
    tau: float = 0.1


# ---------------------------------------------------------------------------
# Dummy physics engine
# ---------------------------------------------------------------------------


class DummyPhysicsEngine:
    """Trivial physics that echoes action as next state."""

    def __init__(self) -> None:
        self.config = DummyConfig()
        self._time = 0.0

    @property
    def state_keys(self) -> set[str]:
        return {"i_d", "i_q", "time"}

    @property
    def action_keys(self) -> set[str]:
        return {"v_d", "v_q"}

    def reset(self, seed: int | None = None) -> dict[str, float]:  # noqa: ARG002
        self._time = 0.0
        return {"i_d": 0.0, "i_q": 0.0, "time": 0.0}

    def step(self, action: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        self._time += self.config.tau
        return {
            "i_d": action.get("v_d", 0.0) * 0.1,
            "i_q": action.get("v_q", 0.0) * 0.1,
            "time": self._time,
        }, {}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Dummy task
# ---------------------------------------------------------------------------


class DummyTask:
    """Minimal ClosedLoopTask for testing."""

    def __init__(self, max_steps: int = 5) -> None:
        self.physics_engine = DummyPhysicsEngine()
        self._step = 0
        self.max_steps = max_steps

    @property
    def reference_keys(self) -> set[str]:
        return {"i_q_ref", "i_d_ref"}

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        self._step = 0
        state = self.physics_engine.reset(seed)
        return state, {"i_q_ref": 1.0, "i_d_ref": 0.0}

    def step(
        self, action: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        self._step += 1
        state, _ = self.physics_engine.step(action)
        done = self._step >= self.max_steps
        return state, {"i_q_ref": 1.0, "i_d_ref": 0.0}, done


# ---------------------------------------------------------------------------
# Dummy controllers
# ---------------------------------------------------------------------------


class DummyDictController:
    """P-controller test double implementing Controller protocol."""

    def reset(self) -> None:
        pass

    def __call__(
        self, state: dict[str, float], reference: dict[str, float]
    ) -> dict[str, float]:
        return {
            "v_d": reference.get("i_d_ref", 0.0) - state.get("i_d", 0.0),
            "v_q": reference.get("i_q_ref", 0.0) - state.get("i_q", 0.0),
        }

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: ARG002
        pass


class DummyTensorController:
    """Neural controller test double implementing TensorController."""

    def __init__(self, output_dim: int = 2) -> None:
        self.output_dim = output_dim
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_info = None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch = observation.shape[0] if observation.dim() > 1 else 1
        action = torch.zeros(batch, self.output_dim)
        self.last_info = {"total_spikes": 10, "syops": 42, "sparsity": 0.5}
        return action

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: ARG002
        pass


class DummySNNModule(torch.nn.Module):
    """Minimal torch.nn.Module pretending to be an SNN."""

    def __init__(self, in_features: int = 5, out_features: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_config() -> DummyConfig:
    """Provide a minimal SystemConfig."""
    return DummyConfig()


@pytest.fixture
def dummy_physics() -> DummyPhysicsEngine:
    """Provide a trivial physics engine."""
    return DummyPhysicsEngine()


@pytest.fixture
def dummy_task() -> DummyTask:
    """Provide a minimal closed-loop task (5 steps)."""
    return DummyTask(max_steps=5)


@pytest.fixture
def dummy_dict_controller() -> DummyDictController:
    """Provide a classical P-controller test double."""
    return DummyDictController()


@pytest.fixture
def dummy_tensor_controller() -> DummyTensorController:
    """Provide a neural controller test double with spike info."""
    return DummyTensorController()


@pytest.fixture
def dummy_snn_module() -> DummySNNModule:
    """Provide a minimal torch.nn.Module SNN."""
    return DummySNNModule()
