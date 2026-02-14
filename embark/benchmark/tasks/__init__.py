"""Closed-loop task implementations."""

from .pmsm_current_control import PMSMCurrentControlTask, SafetyLimits
from .reference_generators import (
    ConstantReference,
    ReferenceGenerator,
    SinusoidalReference,
    StepReference,
)

__all__ = [
    "ConstantReference",
    "PMSMCurrentControlTask",
    "ReferenceGenerator",
    "SafetyLimits",
    "SinusoidalReference",
    "StepReference",
]
