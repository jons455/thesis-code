"""Protocol definitions for state/action processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .task import ClosedLoopTask
from .types import ActionDict, ReferenceDict, StateDict, SystemConfig

if TYPE_CHECKING:
    import torch


class StateProcessor(Protocol):
    """
    Converts a physics state dict into a controller observation tensor.

    Responsible for selecting relevant state variables, combining them
    with reference signals, and normalizing to the range expected by the
    neural controller (typically ``[-1, 1]``).

    Implementors: ``MinMaxProcessor``, ``StandardScalerProcessor``,
    ``IdentityStateProcessor``.

    """

    def configure(self, physics_config: SystemConfig, task: ClosedLoopTask) -> None:
        """
        One-time configuration with system parameters.

        Called once before the first episode to learn normalization
        bounds from the physics configuration.

        Args:
            physics_config: Physical system parameters (limits, time step).
            task: The task instance, providing reference key information.

        """
        ...

    def __call__(self, state: StateDict, reference: ReferenceDict) -> "torch.Tensor":
        """
        Transform state and reference dicts into an observation tensor.

        Args:
            state: Current system state (e.g. ``{"i_d": ..., "i_q": ...}``).
            reference: Current reference signals (e.g. ``{"i_q_ref": ...}``).

        Returns:
            Normalized observation tensor suitable for neural controller
            input.

        """
        ...

    @property
    def output_dim(self) -> int:
        """
        Dimension of the output observation tensor.

        Returns:
            Integer number of features in the produced tensor.

        """
        ...


class ActionProcessor(Protocol):
    """
    Converts a controller action tensor into a physics action dict.

    Responsible for denormalizing the neural controller output from its
    normalized range (e.g. ``[-1, 1]``) to physical units (e.g. volts),
    and mapping tensor elements to named action keys.

    Implementors: ``LinearActionProcessor``, ``PWMActionProcessor``,
    ``IdentityActionProcessor``.

    """

    def configure(self, physics_config: SystemConfig) -> None:
        """
        One-time configuration with system parameters.

        Called once before the first episode to learn action bounds
        from the physics configuration.

        Args:
            physics_config: Physical system parameters (voltage limits, etc.).

        """
        ...

    def __call__(
        self, action: "torch.Tensor", physics_config: SystemConfig
    ) -> ActionDict:
        """
        Convert action tensor to a dict of physical control signals.

        Args:
            action: Normalized action tensor from the neural controller.
            physics_config: Physical system parameters for denormalization.

        Returns:
            Dictionary mapping action names to physical values
            (e.g. ``{"v_d": 12.5, "v_q": -3.2}``).

        """
        ...
