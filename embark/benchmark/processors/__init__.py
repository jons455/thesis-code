"""Processor implementations for rate-encoded SNNs."""

from .identity import IdentityActionProcessor, IdentityStateProcessor
from .pwm import PWMActionProcessor, PWMConverter
from .rate_snn import RateSNNActionProcessor, RateSNNStateProcessor

__all__ = [
    # Rate-encoding SNN processors (main)
    "RateSNNActionProcessor",
    "RateSNNStateProcessor",
    # Hardware deployment (PWM conversion)
    "PWMActionProcessor",
    "PWMConverter",
    # Identity processors (debugging/passthrough)
    "IdentityActionProcessor",
    "IdentityStateProcessor",
]
