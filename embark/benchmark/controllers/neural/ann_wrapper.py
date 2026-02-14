"""
Wrapper for ANN controllers to match TensorController protocol.

Provides ``ANNControllerWrapper`` which adapts any standard
``torch.nn.Module`` (non-spiking) for use inside the
``TensorControllerAdapter`` and ``ClosedLoopHarness`` pipeline.

"""

from __future__ import annotations

from typing import Any

import torch

from embark.benchmark.interfaces import TensorController


class ANNControllerWrapper(TensorController):
    """
    Wrap a standard PyTorch ANN model to satisfy the ``TensorController`` protocol.

    Unlike ``SNNControllerWrapper``, this wrapper does not expect spike
    statistics from the model.  ``last_info`` is always ``None``.

    Args:
        model: A ``torch.nn.Module`` implementing a conventional (non-spiking)
            neural network controller.

    Attributes:
        model: The underlying PyTorch module, accessible for NeuroBench
            hook registration.
        last_info: Always ``None`` for ANN controllers (no spike data).

    Example::

        import torch.nn as nn

        ann = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))
        wrapper = ANNControllerWrapper(ann)
        action = wrapper.forward(torch.randn(1, 5))

    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        """Reset internal state if the model supports it."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Run ANN inference on the observation tensor.

        Args:
            observation: Normalized input tensor.

        Returns:
            Action tensor from the neural network.

        """
        return self.model(observation)

    def get_state(self) -> dict[str, Any]:
        """
        Serialize internal state for checkpointing.

        Returns:
            Empty dictionary (ANN controllers are stateless between steps).

        """
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: ARG002
        """
        Restore internal state from checkpoint.

        Args:
            state: Dictionary previously returned by ``get_state()``.
                Ignored for stateless ANN controllers.

        """
