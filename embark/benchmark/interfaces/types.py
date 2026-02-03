"""Shared type aliases for benchmark interfaces."""

from __future__ import annotations

from typing import Any, Protocol

StateDict = dict[str, float]
ActionDict = dict[str, float]
ReferenceDict = dict[str, float]
ControllerInfo = dict[str, Any]


class SystemConfig(Protocol):
    """Minimum physical configuration required by processors/controllers."""

    i_max: float
    u_max: float
    tau: float
