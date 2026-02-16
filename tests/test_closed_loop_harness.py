"""Tests for the new closed-loop harness and interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.interfaces import (
    ActionDict,
    ActionProcessor,
    DictController,
    SystemConfig,
    TensorController,
)
from embark.benchmark.metrics.accumulators import TrackingMAE
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

    def reset(self, _seed: int | None = None):
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

    def reset(self, _seed: int | None = None):
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

    def set_state(self, _state: dict):
        pass

    @classmethod
    def from_system_config(cls, config, tuning: str = "technical_optimum"):
        return cls()


class DummyTensorController(TensorController):
    def reset(self) -> None:
        pass

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1)

    def get_state(self) -> dict:
        return {}

    def set_state(self, _state: dict):
        pass


def test_harness_runs_dict_controller():
    task = DummyTask()
    controller = DummyDictController()
    metrics = [TrackingMAE(tracked_keys=["i_q"])]
    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()
    assert "mae_i_q" in results


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

    metrics = [TrackingMAE(tracked_keys=["i_q"])]
    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()
    assert "mae_i_q" in results


# ---------------------------------------------------------------------------
# set_currents wiring
# ---------------------------------------------------------------------------


@dataclass
class _CurrentTrackingActionProcessor(ActionProcessor):
    """Spy that records set_currents calls."""

    action_keys: list = field(default_factory=lambda: ["v_alpha"])
    received_currents: list = field(default_factory=list)

    def configure(self, _physics_config: SystemConfig) -> None:
        pass

    def set_currents(self, i_d: float, i_q: float) -> None:
        self.received_currents.append((i_d, i_q))

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:
        values = action.detach().cpu().flatten().tolist()
        return {k: values[i] for i, k in enumerate(self.action_keys)}


class DummyPhysicsWithCurrents:
    """Physics stub whose state includes i_d and i_q."""

    def __init__(self) -> None:
        self.config = DummyConfig()
        self._time = 0.0

    @property
    def state_keys(self) -> set[str]:
        return {"i_d", "i_q", "time"}

    @property
    def action_keys(self) -> set[str]:
        return {"v_alpha"}

    def reset(self, _seed: int | None = None):
        self._time = 0.0
        return {"i_d": 1.0, "i_q": 2.0, "time": self._time}

    def step(self, action: dict[str, float]):
        self._time += self.config.tau
        return {
            "i_d": 1.0,
            "i_q": action.get("v_alpha", 0.0),
            "time": self._time,
        }, {}

    def close(self) -> None:
        pass


class DummyTaskWithCurrents:
    def __init__(self) -> None:
        self.physics_engine = DummyPhysicsWithCurrents()
        self._step = 0
        self.max_steps = 3

    @property
    def reference_keys(self) -> set[str]:
        return {"i_q_ref"}

    def reset(self, _seed: int | None = None):
        self._step = 0
        state = self.physics_engine.reset()
        return state, {"i_q_ref": 0.0}

    def step(self, action: dict[str, float]):
        self._step += 1
        state, _info = self.physics_engine.step(action)
        done = self._step >= self.max_steps
        return state, {"i_q_ref": 0.0}, done


def test_adapter_calls_set_currents_on_action_processor():
    """TensorControllerAdapter should feed i_d/i_q to set_currents when available."""
    task = DummyTaskWithCurrents()
    tensor_ctrl = DummyTensorController()
    state_proc = IdentityStateProcessor(state_keys=["i_q"], reference_keys=["i_q_ref"])
    action_proc = _CurrentTrackingActionProcessor()

    adapter = TensorControllerAdapter(
        controller=tensor_ctrl,
        state_processor=state_proc,
        action_processor=action_proc,
    )
    adapter.configure(task.physics_engine.config, task)

    metrics = [TrackingMAE(tracked_keys=["i_q"])]
    harness = ClosedLoopHarness(task=task, controller=adapter, metrics=metrics)
    harness.run()

    # set_currents should have been called once per step (3 steps)
    assert len(action_proc.received_currents) == task.max_steps
    # First call should see the reset state (i_d=1.0, i_q=2.0)
    assert action_proc.received_currents[0] == (1.0, 2.0)
