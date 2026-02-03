"""Reference generator implementations for closed-loop tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from embark.benchmark.interfaces import ReferenceDict


class ReferenceGenerator(Protocol):
    """Callable reference generator."""

    def reset(self) -> None:
        """Reset internal generator state."""
        ...

    def __call__(self, step: int, time_s: float) -> ReferenceDict:
        """Return reference at the given step/time."""
        ...


@dataclass
class StepReference:
    """Step reference generator for i_d and i_q."""

    i_d_ref: float
    i_q_ref: float
    step_time_s: float = 0.0

    def reset(self) -> None:
        pass

    def __call__(self, step: int, time_s: float) -> ReferenceDict:
        if time_s < self.step_time_s:
            return {"i_d_ref": 0.0, "i_q_ref": 0.0}
        return {"i_d_ref": self.i_d_ref, "i_q_ref": self.i_q_ref}


@dataclass
class ConstantReference:
    """Constant reference generator."""

    i_d_ref: float
    i_q_ref: float

    def reset(self) -> None:
        pass

    def __call__(self, step: int, time_s: float) -> ReferenceDict:
        return {"i_d_ref": self.i_d_ref, "i_q_ref": self.i_q_ref}


@dataclass
class SinusoidalReference:
    """Sinusoidal i_q reference with constant i_d."""

    i_d_ref: float
    i_q_amp: float
    i_q_offset: float = 0.0
    frequency_hz: float = 1.0

    def reset(self) -> None:
        pass

    def __call__(self, step: int, time_s: float) -> ReferenceDict:
        i_q_ref = self.i_q_offset + self.i_q_amp * math.sin(
            2 * math.pi * self.frequency_hz * time_s
        )
        return {"i_d_ref": self.i_d_ref, "i_q_ref": i_q_ref}
