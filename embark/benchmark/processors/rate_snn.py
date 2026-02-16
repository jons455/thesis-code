"""
Unified processors for all rate-encoding SNN controllers.

This module provides configurable state and action processors that handle
any rate-encoding SNN architecture through feature flags rather than
specialized subclasses.

Classes
-------
RateSNNStateProcessor
    Configurable state processor supporting currents, errors, references,
    speed, derivatives, EMA filters, previous actions, and integrals.
RateSNNActionProcessor
    Action processor supporting both absolute and incremental output modes.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from embark.benchmark.interfaces import (
    ActionDict,
    ActionProcessor,
    ClosedLoopTask,
    ReferenceDict,
    StateDict,
    StateProcessor,
    SystemConfig,
)


@dataclass
class RateSNNStateProcessor:
    """
    Configurable state processor for all rate-encoding SNN controllers.

    Produces a flat observation tensor by concatenating selected feature
    groups in a fixed order.  Each group can be toggled on or off via
    boolean flags.

    Feature order (when enabled):
        [i_d, i_q]                    — raw currents
        [i_d_ref, i_q_ref]            — raw references
        [e_d, e_q]                    — tracking errors
        [n]                           — motor speed
        [u_d_prev, u_q_prev]          — previous actions
        [de_d, de_q, dn]              — derivatives
        [ema_slow_e_d, ema_slow_e_q]  — slow EMA of errors
        [ema_fast_e_d, ema_fast_e_q]  — fast EMA of errors
        [int_e_d, int_e_q]            — integral of errors

    Parameters
    ----------
    error_gain : float
        Amplification factor for error signals before clipping to [-1, 1].
    n_max : float
        Maximum speed in RPM for normalization.
    ema_alpha_slow : float
        Smoothing factor for slow EMA filter (close to 1 = more smoothing).
    ema_alpha_fast : float
        Smoothing factor for fast EMA filter.
    integral_limit : float
        Anti-windup clamp for error integrals (in normalized units).
    delta_max : float
        Maximum allowed derivative magnitude for derivative clipping.
    include_currents : bool
        Include normalized i_d, i_q.
    include_references : bool
        Include normalized i_d_ref, i_q_ref.
    include_errors : bool
        Include normalized, gain-amplified tracking errors e_d, e_q.
    include_speed : bool
        Include normalized motor speed n.
    include_prev_action : bool
        Include previous action voltages u_d_prev, u_q_prev.
    include_derivatives : bool
        Include finite-difference derivatives de_d, de_q, dn.
    include_ema_slow : bool
        Include slow EMA-filtered errors.
    include_ema_fast : bool
        Include fast EMA-filtered errors.
    include_integral : bool
        Include accumulated error integrals.

    """

    # Normalization parameters
    error_gain: float = 10.0
    n_max: float = 4000.0
    ema_alpha_slow: float = 0.98
    ema_alpha_fast: float = 0.70
    integral_limit: float = 1.0
    delta_max: float = 10.0

    # Feature flags
    include_currents: bool = True
    include_references: bool = False
    include_errors: bool = True
    include_speed: bool = True
    include_prev_action: bool = False
    include_derivatives: bool = False
    include_ema_slow: bool = False
    include_ema_fast: bool = False
    include_integral: bool = False

    # Internal state (not constructor args)
    _i_max: float = field(default=1.0, init=False, repr=False)
    _u_max: float = field(default=1.0, init=False, repr=False)
    _tau: float = field(default=1e-4, init=False, repr=False)

    # Stateful features
    _prev_e_d: float = field(default=0.0, init=False, repr=False)
    _prev_e_q: float = field(default=0.0, init=False, repr=False)
    _prev_n: float = field(default=0.0, init=False, repr=False)
    _ema_slow_e_d: float = field(default=0.0, init=False, repr=False)
    _ema_slow_e_q: float = field(default=0.0, init=False, repr=False)
    _ema_fast_e_d: float = field(default=0.0, init=False, repr=False)
    _ema_fast_e_q: float = field(default=0.0, init=False, repr=False)
    _int_e_d: float = field(default=0.0, init=False, repr=False)
    _int_e_q: float = field(default=0.0, init=False, repr=False)
    _prev_u_d: float = field(default=0.0, init=False, repr=False)
    _prev_u_q: float = field(default=0.0, init=False, repr=False)
    _first_step: bool = field(default=True, init=False, repr=False)

    def configure(
        self, physics_config: SystemConfig, task: ClosedLoopTask  # noqa: ARG002
    ) -> None:
        """One-time setup with system parameters."""
        self._i_max = physics_config.i_max
        self._u_max = getattr(physics_config, "u_max", 1.0)
        self._tau = getattr(physics_config, "tau", 1e-4)

    def reset(self) -> None:
        """Reset all stateful features to initial values."""
        self._prev_e_d = 0.0
        self._prev_e_q = 0.0
        self._prev_n = 0.0
        self._ema_slow_e_d = 0.0
        self._ema_slow_e_q = 0.0
        self._ema_fast_e_d = 0.0
        self._ema_fast_e_q = 0.0
        self._int_e_d = 0.0
        self._int_e_q = 0.0
        self._prev_u_d = 0.0
        self._prev_u_q = 0.0
        self._first_step = True

    def set_prev_action(self, u_d: float, u_q: float) -> None:
        """Feed back the previous control action for incremental models."""
        self._prev_u_d = u_d
        self._prev_u_q = u_q

    def __call__(self, state: StateDict, reference: ReferenceDict) -> torch.Tensor:
        """Transform state and reference dicts into a normalized observation tensor."""
        features: list[float] = []

        # --- Compute shared intermediate values ---
        i_d = state["i_d"]
        i_q = state["i_q"]
        i_d_ref = reference["i_d_ref"]
        i_q_ref = reference["i_q_ref"]

        # Normalized currents
        i_d_norm = i_d / self._i_max
        i_q_norm = i_q / self._i_max

        # Normalized errors (with gain and clipping)
        e_d_raw = (i_d_ref - i_d) / self._i_max
        e_q_raw = (i_q_ref - i_q) / self._i_max
        e_d = max(-1.0, min(1.0, e_d_raw * self.error_gain))
        e_q = max(-1.0, min(1.0, e_q_raw * self.error_gain))

        # Normalized speed (omega rad/s -> RPM -> normalized)
        omega = state.get("omega", 0.0)
        n_rpm = omega * 60.0 / (2.0 * math.pi)
        n = n_rpm / self.n_max

        # --- Assemble features in fixed order ---

        if self.include_currents:
            features.extend([i_d_norm, i_q_norm])

        if self.include_references:
            features.extend([i_d_ref / self._i_max, i_q_ref / self._i_max])

        if self.include_errors:
            features.extend([e_d, e_q])

        if self.include_speed:
            features.append(n)

        if self.include_prev_action:
            features.extend(
                [
                    self._prev_u_d / self._u_max,
                    self._prev_u_q / self._u_max,
                ]
            )

        if self.include_derivatives:
            if self._first_step:
                de_d = 0.0
                de_q = 0.0
                dn = 0.0
            else:
                de_d = (e_d - self._prev_e_d) / self._tau
                de_q = (e_q - self._prev_e_q) / self._tau
                dn = (n - self._prev_n) / self._tau
                # Clip derivatives
                de_d = max(-self.delta_max, min(self.delta_max, de_d))
                de_q = max(-self.delta_max, min(self.delta_max, de_q))
                dn = max(-self.delta_max, min(self.delta_max, dn))
            features.extend([de_d, de_q, dn])

        if self.include_ema_slow:
            alpha = self.ema_alpha_slow
            if self._first_step:
                self._ema_slow_e_d = e_d
                self._ema_slow_e_q = e_q
            else:
                self._ema_slow_e_d = alpha * self._ema_slow_e_d + (1 - alpha) * e_d
                self._ema_slow_e_q = alpha * self._ema_slow_e_q + (1 - alpha) * e_q
            features.extend([self._ema_slow_e_d, self._ema_slow_e_q])

        if self.include_ema_fast:
            alpha = self.ema_alpha_fast
            if self._first_step:
                self._ema_fast_e_d = e_d
                self._ema_fast_e_q = e_q
            else:
                self._ema_fast_e_d = alpha * self._ema_fast_e_d + (1 - alpha) * e_d
                self._ema_fast_e_q = alpha * self._ema_fast_e_q + (1 - alpha) * e_q
            features.extend([self._ema_fast_e_d, self._ema_fast_e_q])

        if self.include_integral:
            self._int_e_d += e_d_raw * self._tau
            self._int_e_q += e_q_raw * self._tau
            # Anti-windup clamp
            self._int_e_d = max(
                -self.integral_limit, min(self.integral_limit, self._int_e_d)
            )
            self._int_e_q = max(
                -self.integral_limit, min(self.integral_limit, self._int_e_q)
            )
            features.extend([self._int_e_d, self._int_e_q])

        # --- Update state for next step ---
        self._prev_e_d = e_d
        self._prev_e_q = e_q
        self._prev_n = n
        self._first_step = False

        return torch.tensor(features, dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        """Dimension of the output observation tensor."""
        dim = 0
        if self.include_currents:
            dim += 2
        if self.include_references:
            dim += 2
        if self.include_errors:
            dim += 2
        if self.include_speed:
            dim += 1
        if self.include_prev_action:
            dim += 2
        if self.include_derivatives:
            dim += 3
        if self.include_ema_slow:
            dim += 2
        if self.include_ema_fast:
            dim += 2
        if self.include_integral:
            dim += 2
        return dim


@dataclass
class RateSNNActionProcessor:
    """
    Action processor for rate-encoding SNN controllers.

    Supports two output modes:

    - **Absolute** (``incremental=False``): The network output is scaled
      from [-1, 1] to [-u_max, u_max] as the final voltage command.
    - **Incremental** (``incremental=True``): The network output is
      interpreted as a delta ``Δu`` which is accumulated over time.
      The delta is scaled by ``delta_max`` and clamped to ``[-u_max, u_max]``.

    Parameters
    ----------
    output_keys : tuple[str, ...]
        Keys for the output action dict (default: ``("v_d", "v_q")``).
    incremental : bool
        If True, output is accumulated as deltas.
    delta_max : float
        Maximum per-step voltage change in incremental mode (volts).

    """

    output_keys: tuple[str, ...] = ("v_d", "v_q")
    incremental: bool = False
    delta_max: float = 0.2

    # Internal state
    _u_max: float = field(default=1.0, init=False, repr=False)
    _accum: list[float] = field(default_factory=list, init=False, repr=False)

    def configure(self, physics_config: SystemConfig) -> None:
        """One-time setup with system parameters."""
        self._u_max = getattr(physics_config, "u_max", 1.0)
        self._accum = [0.0] * len(self.output_keys)

    def reset(self) -> None:
        """Reset accumulated state for incremental mode."""
        self._accum = [0.0] * len(self.output_keys)

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig  # noqa: ARG002
    ) -> ActionDict:
        """Convert action tensor to physical voltage dict."""
        action_list = action.detach().cpu().flatten().tolist()
        if len(action_list) < len(self.output_keys):
            raise ValueError("Action tensor smaller than number of output keys.")

        output: ActionDict = {}

        if self.incremental:
            for idx, key in enumerate(self.output_keys):
                delta = action_list[idx] * self.delta_max
                self._accum[idx] += delta
                # Clamp to voltage limits
                self._accum[idx] = max(-self._u_max, min(self._u_max, self._accum[idx]))
                output[key] = self._accum[idx]
        else:
            for idx, key in enumerate(self.output_keys):
                # Scale from [-1, 1] to [-u_max, u_max]
                output[key] = float(action_list[idx] * self._u_max)

        return output

    @property
    def last_accumulated(self) -> list[float]:
        """Current accumulated voltages (incremental mode)."""
        return list(self._accum)
