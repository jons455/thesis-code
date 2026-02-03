"""Adapter to make TensorController conform to unified Controller interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from embark.benchmark.interfaces import (
    ActionDict,
    ActionProcessor,
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
    TensorController,
)

if TYPE_CHECKING:
    import torch


@dataclass
class TensorControllerAdapter:
    """
    Wraps TensorController + processors into unified Controller interface.

    This eliminates if/else logic in the harness by making neural controllers
    behave identically to classical controllers from the harness's perspective.

    The adapter exposes intermediate values for metrics:
    - `last_observation`: Input tensor (normalized)
    - `last_action_tensor`: Output tensor (normalized, before denormalization)
    - `last_info`: Spike statistics from underlying controller
    - `model`: Direct access to underlying controller for hook registration

    Usage:
        snn = SNNControllerAgent(...)
        state_proc = MinMaxProcessor(...)
        action_proc = LinearActionProcessor(...)

        # Wrap into unified interface
        controller = TensorControllerAdapter(
            controller=snn,
            state_processor=state_proc,
            action_processor=action_proc,
        )

        # Configure processors
        controller.configure(task.physics_engine.config, task)

        # Access underlying model for hooks (e.g., NeuroBench WorkloadMetric)
        workload_metric = WorkloadMetric(controller.model, ...)

        # Now usable in harness without special handling
        harness = ClosedLoopHarness(task=task, controller=controller, ...)

    """

    controller: TensorController
    state_processor: StateProcessor
    action_processor: ActionProcessor
    _physics_config: SystemConfig | None = field(default=None, repr=False)

    # Intermediate values exposed for metrics (not swallowed)
    _last_observation: "torch.Tensor | None" = field(default=None, repr=False)
    _last_action_tensor: "torch.Tensor | None" = field(default=None, repr=False)

    def configure(self, physics_config: SystemConfig, task: ClosedLoopTask) -> None:
        """Configure processors with physics bounds."""
        self._physics_config = physics_config
        self.state_processor.configure(physics_config, task)
        self.action_processor.configure(physics_config)

    def reset(self) -> None:
        """Reset underlying controller state."""
        self.controller.reset()
        self._last_observation = None
        self._last_action_tensor = None

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        """
        Process state, run controller, process action.

        Intermediate tensors are stored and accessible via properties:
        - self.last_observation: input tensor
        - self.last_action_tensor: output tensor (before denormalization)

        """
        if self._physics_config is None:
            raise RuntimeError(
                "TensorControllerAdapter.configure() must be called before use."
            )

        # State dict → observation tensor (store for metrics)
        self._last_observation = self.state_processor(state, reference)

        # Neural controller inference (store output for metrics)
        self._last_action_tensor = self.controller.forward(self._last_observation)

        # Action tensor → action dict
        action = self.action_processor(self._last_action_tensor, self._physics_config)

        return action

    def get_state(self) -> dict[str, Any]:
        """Get controller state for checkpointing."""
        return self.controller.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore controller state from checkpoint."""
        self.controller.set_state(state)

    @property
    def model(self) -> Any:
        """
        Direct access to underlying controller/model for hook registration.

        Use this to register forward hooks for NeuroBench WorkloadMetric or other
        external metric tools that need model access.

        This property attempts to unwrap the controller to find the actual
        torch.nn.Module. For example, if using SNNControllerAgent, it returns the
        underlying snn_model.

        """
        # Unwrap SNNControllerAgent to get the actual PyTorch model
        if hasattr(self.controller, "model"):
            return self.controller.model
        return self.controller

    @property
    def last_observation(self) -> "torch.Tensor | None":
        """Last input tensor (normalized) passed to the controller."""
        return self._last_observation

    @property
    def last_action_tensor(self) -> "torch.Tensor | None":
        """
        Last output tensor (normalized) from the controller.

        This is the raw neural network output before denormalization to physical voltage
        units.

        """
        return self._last_action_tensor

    @property
    def last_info(self) -> dict[str, Any] | None:
        """
        Spike statistics from underlying controller.

        Returns whatever the underlying TensorController stores in its
        `last_info` attribute after forward(). Typically includes:
        - total_spikes: int
        - total_syops: int
        - layer_spikes: dict[str, int]
        - sparsity: float

        """
        return getattr(self.controller, "last_info", None)
