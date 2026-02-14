"""Protocol definition for closed-loop tasks."""

from __future__ import annotations

from typing import Protocol

from .physics import PhysicsEngine
from .types import ActionDict, ReferenceDict, StateDict


class ClosedLoopTask(Protocol):
    """
    Defines the control objective for a closed-loop benchmark episode.

    A task owns a ``PhysicsEngine``, manages reference signal generation,
    and determines episode termination (e.g. via safety limits or a
    maximum step count).

    Implementors must provide:

    - A ``physics_engine`` that simulates the dynamical system.
    - A ``reset()`` method that initialises the environment and returns
      the initial ``(state, reference)`` pair.
    - A ``step()`` method that advances the simulation by one timestep
      and returns the next ``(state, reference, done)`` triple.

    Example implementor: ``PMSMCurrentControlTask``.

    """

    @property
    def physics_engine(self) -> PhysicsEngine:
        """
        The underlying dynamical system.

        Returns:
            The physics engine managing simulation dynamics.

        """
        ...

    @property
    def reference_keys(self) -> set[str]:
        """
        Keys provided in the reference dictionary at each step.

        Returns:
            Set of string keys (e.g. ``{"i_q_ref", "i_d_ref"}``).

        """
        ...

    @property
    def max_steps(self) -> int | None:
        """
        Maximum episode length.

        Returns:
            Integer step limit, or ``None`` for unlimited episodes.

        """
        ...

    def reset(self, seed: int | None = None) -> tuple[StateDict, ReferenceDict]:
        """
        Reset the task and physics engine for a new episode.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Tuple of ``(initial_state, initial_reference)`` dicts.

        """
        ...

    def step(self, action: ActionDict) -> tuple[StateDict, ReferenceDict, bool]:
        """
        Advance the simulation by one timestep.

        Applies the control action to the physics engine, generates the
        next reference signal, and checks termination conditions.

        Args:
            action: Control action dict (e.g. ``{"v_d": ..., "v_q": ...}``).

        Returns:
            Tuple of ``(next_state, next_reference, done)`` where
            ``done`` is ``True`` if the episode should terminate (safety
            violation or step limit reached).

        """
        ...
