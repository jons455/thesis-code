"""Tests for the new closed-loop harness and interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.interfaces import DictController, TensorController
from embark.benchmark.metrics.accumulators import TrackingRMSE
from embark.benchmark.processors import IdentityActionProcessor, IdentityStateProcessor


@dataclass
class DummyConfig:
    i_max: float = 1.0
    u_max: float = 1.0
    tau: float = 0.1


class DummyPhysics:
    def __init__(self) -> None:
        self.config = DummyConfig()
        self._time = 0.0

    @property
    def state_keys(self) -> set[str]:
        return {"i_q", "time"}

    @property
    def action_keys(self) -> set[str]:
        return {"v_alpha"}

    def reset(self, seed: int | None = None):  # noqa: ARG002
        self._time = 0.0
        return {"i_q": 0.0, "time": self._time}

    def step(self, action: dict[str, float]):
        self._time += self.config.tau
        i_q = action.get("v_alpha", 0.0)
        return {"i_q": i_q, "time": self._time}, {}

    def close(self) -> None:
        pass


class DummyTask:
    def __init__(self) -> None:
        self.physics_engine = DummyPhysics()
        self._step = 0
        self.max_steps = 3

    @property
    def reference_keys(self) -> set[str]:
        return {"i_q_ref"}

    def reset(self, seed: int | None = None):  # noqa: ARG002
        self._step = 0
        state = self.physics_engine.reset()
        return state, {"i_q_ref": 0.0}

    def step(self, action: dict[str, float]):
        self._step += 1
        state, _info = self.physics_engine.step(action)
        done = self._step >= self.max_steps
        return state, {"i_q_ref": 0.0}, done


class DummyDictController(DictController):
    def reset(self) -> None:
        pass

    def __call__(self, state: dict[str, float], reference: dict[str, float]):
        return {"v_alpha": state["i_q"] - reference["i_q_ref"]}

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):  # noqa: ARG002
        pass

    @classmethod
    def from_system_config(
        cls, config, tuning: str = "technical_optimum"
    ):  # noqa: ARG002
        return cls()


class DummyTensorController(TensorController):
    def reset(self) -> None:
        pass

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1)

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):  # noqa: ARG002
        pass


def test_harness_runs_dict_controller():
    task = DummyTask()
    controller = DummyDictController()
    metrics = [TrackingRMSE(tracked_keys=["i_q"])]
    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()
    assert "rmse_i_q" in results


def test_harness_runs_tensor_controller():
    task = DummyTask()
    tensor_controller = DummyTensorController()
    state_proc = IdentityStateProcessor(state_keys=["i_q"], reference_keys=["i_q_ref"])
    action_proc = IdentityActionProcessor(action_keys=["v_alpha"])

    # Wrap TensorController with adapter for unified interface
    controller = TensorControllerAdapter(
        controller=tensor_controller,
        state_processor=state_proc,
        action_processor=action_proc,
    )
    controller.configure(task.physics_engine.config, task)

    metrics = [TrackingRMSE(tracked_keys=["i_q"])]
    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()
    assert "rmse_i_q" in results
