"""
Wrapper for SNN controllers to match TensorController protocol.

Provides ``SNNControllerWrapper`` which adapts any ``torch.nn.Module``-based
spiking neural network for use inside the ``TensorControllerAdapter`` and
``ClosedLoopHarness`` pipeline.

"""

from __future__ import annotations

from typing import Any

import torch

from embark.benchmark.interfaces import TensorController


class SNNControllerWrapper(TensorController):
    """
    Wrap a PyTorch SNN model to satisfy the ``TensorController`` protocol.

    This wrapper handles models that return either a single tensor or a
    ``(action, info)`` tuple.  When the model returns spike statistics as
    the second element of a tuple, they are captured in ``last_info`` for
    downstream metric accumulators.

    Args:
        model: A ``torch.nn.Module`` implementing a spiking neural network.
            Must accept an observation tensor and return either a single
            action tensor or a ``(action_tensor, info_dict)`` tuple.

    Attributes:
        model: The underlying PyTorch SNN module, accessible for NeuroBench
            hook registration.
        last_info: Dictionary of spike statistics from the most recent
            ``forward()`` call, or ``None`` if the model does not return
            info.

    Example::

        import torch.nn as nn

        class MySNN(nn.Module):
            def forward(self, x):
                action = ...  # compute
                info = {"total_spikes": 42, "sparsity": 0.85}
                return action, info

        wrapper = SNNControllerWrapper(MySNN())
        action = wrapper.forward(observation)
        print(wrapper.last_info)  # {"total_spikes": 42, "sparsity": 0.85}

    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._state: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        """Reset internal state and clear cached spike info."""
        self._state = None
        if hasattr(self.model, "reset"):
            self.model.reset()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Run SNN inference on the observation tensor.

        If the underlying model returns a ``(tensor, dict)`` tuple, the
        dict is captured in ``self.last_info`` for metric accumulators.

        Args:
            observation: Normalized input tensor (e.g.
                ``[i_d, i_q, e_d, e_q, n]``).

        Returns:
            Action tensor (e.g. normalized voltages in ``[-1, 1]``).

        """
        output = self.model(observation)
        if isinstance(output, tuple) and len(output) >= 2:
            action, info = output[0], output[1]
            if isinstance(info, dict):
                self.last_info = info
            return action
        return output

    def get_state(self) -> dict[str, Any]:
        """
        Serialize internal state for checkpointing.

        Returns:
            Dictionary containing the SNN hidden state.

        """
        return {"state": self._state}

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore internal state from checkpoint.

        Args:
            state: Dictionary previously returned by ``get_state()``.

        """
        self._state = state.get("state")
