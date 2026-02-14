"""
NeuroBench-compatible model wrapper for closed-loop controllers.

This module provides ``NeuroBenchClosedLoopModel``, a thin wrapper that
exposes an embark ``Controller`` in a form compatible with NeuroBench's
hook-based workload metrics (``SynapticOperations``, ``ActivationSparsity``).

.. note::

    This is **experimental** interop code. NeuroBench does not yet support
    closed-loop benchmarks natively. This wrapper is designed for the
    ``2025_GC`` branch and will need updating as upstream evolves.

Why a wrapper?
--------------
NeuroBench's workload metrics register ``torch.nn.Module`` forward hooks.
Embark's controllers operate on ``StateDict`` / ``ReferenceDict`` dicts
and may or may not contain a neural network.  This wrapper exposes the
underlying ``torch.nn.Module`` (if present) via ``.net`` while keeping
the ``Controller`` protocol intact for the harness.

Usage::

    from embark.benchmark.contrib.neurobench import NeuroBenchClosedLoopModel

    adapter = TensorControllerAdapter(snn, state_proc, action_proc)
    adapter.configure(task.physics_engine.config, task)

    nb_model = NeuroBenchClosedLoopModel(adapter)

    # Access the underlying torch.nn.Module for NeuroBench hooks
    pytorch_model = nb_model.net

    # Use in harness as before — pass-through to wrapped controller
    harness = ClosedLoopHarness(task=task, controller=nb_model)

"""

from __future__ import annotations

from typing import Any

from embark.benchmark.interfaces import (
    ActionDict,
    Controller,
    ReferenceDict,
    StateDict,
)


class NeuroBenchClosedLoopModel:
    """
    Expose an embark ``Controller`` for NeuroBench hook registration.

    This wrapper does **not** import or depend on the ``neurobench``
    package. It simply provides the ``.net`` access pattern that
    NeuroBench metrics expect.

    Args:
        controller: Any object satisfying the ``Controller`` protocol.
            For neural controllers this is typically a
            ``TensorControllerAdapter``.
        name: Human-readable name for reports.

    Attributes:
        controller: The wrapped controller instance.
        net: The underlying ``torch.nn.Module`` (if available), or
            ``None`` for classical controllers.

    """

    def __init__(self, controller: Controller, name: str = "ClosedLoopModel") -> None:
        self.controller = controller
        self._name = name

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    @property
    def net(self) -> Any:
        """
        Underlying ``torch.nn.Module`` for hook registration.

        Returns:
            The PyTorch module, or ``None`` if the controller has no
            neural network.

        """
        if hasattr(self.controller, "model"):
            return self.controller.model
        return None

    def reset(self) -> None:
        """Reset the underlying controller state."""
        self.controller.reset()

    def __call__(self, state: StateDict, reference: ReferenceDict) -> ActionDict:
        """
        Compute control action (delegates to wrapped controller).

        Args:
            state: Current system state.
            reference: Current reference signals.

        Returns:
            Action dict with control commands.

        """
        return self.controller(state, reference)

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return self.controller.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        self.controller.set_state(state)

    @property
    def last_info(self) -> dict[str, Any] | None:
        """Spike / inference statistics from the last forward pass."""
        return getattr(self.controller, "last_info", None)

    def __repr__(self) -> str:
        return (
            f"NeuroBenchClosedLoopModel(name={self._name!r}, "
            f"controller={self.controller.__class__.__name__})"
        )
