"""
Controller agents for PMSM current control benchmark.
"""

from .pi_controller import PIControllerAgent
from .snn_controller import SNNControllerAgent, SNNControllerTorchAgent

__all__ = [
    "PIControllerAgent",
    "SNNControllerAgent",
    "SNNControllerTorchAgent",
]
