"""Closed-loop PMSM current control task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from embark.benchmark.interfaces import ReferenceDict, StateDict
from embark.benchmark.physics import PMSMConfig, PMSMPhysicsEngine
from embark.benchmark.tasks.reference_generators import (
    ReferenceGenerator,
    StepReference,
)
from embark.benchmark.utils.validation import validate_numeric_dict
from embark.utils.config import DEFAULT_MAX_STEPS


@dataclass
class SafetyLimits:
    """
    Safety limits for early termination.

    If any limit is exceeded, the episode terminates with done=True.
    Set a limit to None to disable that check.

    Safety checks are split into two phases:
    - Action limits: Checked BEFORE physics step (prevents commanding crazy voltages)
    - State limits: Checked AFTER physics step (detects system instability)

    """

    max_current_a: float | None = 20.0  # Max |i_d| or |i_q| in Amperes
    max_voltage_v: float | None = None  # Max |v_d| or |v_q| in Volts (None = no limit)
    max_speed_rpm: float | None = None  # Max omega in RPM (None = no limit)

    def check_action(self, action: dict[str, float]) -> str | None:
        """
        Check action limits BEFORE applying to physics.

        Args:
            action: Action dict with v_d, v_q (or v_alpha, v_beta).

        Returns:
            Violation reason string if violated, None if safe.

        """
        if self.max_voltage_v is not None:
            for key in ["v_d", "v_q", "v_alpha", "v_beta"]:
                if key in action and abs(action[key]) > self.max_voltage_v:
                    return f"voltage_limit_exceeded:{key}={action[key]:.1f}V"

        # NaN check on action
        for key, val in action.items():
            if np.isnan(val) or np.isinf(val):
                return f"action_nan:{key}"

        return None

    def check_state(self, state: StateDict) -> str | None:
        """
        Check state limits AFTER physics step.

        Args:
            state: State dict with i_d, i_q, omega.

        Returns:
            Violation reason string if violated, None if safe.

        """
        # Current limits
        if self.max_current_a is not None:
            if abs(state.get("i_d", 0.0)) > self.max_current_a:
                return f"current_limit_exceeded:i_d={state['i_d']:.2f}A"
            if abs(state.get("i_q", 0.0)) > self.max_current_a:
                return f"current_limit_exceeded:i_q={state['i_q']:.2f}A"

        # Speed limits (omega is in rad/s, convert to RPM)
        if self.max_speed_rpm is not None:
            omega_rpm = abs(state.get("omega", 0.0)) * 60 / (2 * np.pi)
            if omega_rpm > self.max_speed_rpm:
                return f"speed_limit_exceeded:omega={omega_rpm:.0f}rpm"

        # NaN check on state
        for key in ["i_d", "i_q", "omega"]:
            if key in state:
                val = state[key]
                if np.isnan(val) or np.isinf(val):
                    return f"state_nan:{key}"

        return None

    def check(self, state: StateDict, action: dict[str, float] | None = None) -> bool:
        """
        Legacy combined check (for backward compatibility).

        Prefer using check_action() and check_state() separately.

        """
        if action is not None:
            if self.check_action(action) is not None:
                return True
        return self.check_state(state) is not None


@dataclass
class PMSMCurrentControlTask:
    """
    Closed-loop current control task for PMSM.

    This task composes:

    - A physics engine (PMSM simulation via GEM)
    - A reference generator (step, sinusoidal, etc.)
    - Safety limits (optional early termination)

    The reference generator is injected via dependency injection,
    allowing the same task logic for different benchmarks.

    Example::

        # Step response benchmark
        task = PMSMCurrentControlTask(
            physics_engine=engine,
            reference_generator=StepReference(i_q_ref=2.0),
        )

        # Tracking benchmark (different generator, same task)
        task = PMSMCurrentControlTask(
            physics_engine=engine,
            reference_generator=SinusoidalReference(amplitude=2.0, freq_hz=10),
        )

    """

    physics_engine: PMSMPhysicsEngine = field(default_factory=PMSMPhysicsEngine)
    reference_generator: ReferenceGenerator = field(
        default_factory=lambda: StepReference(i_d_ref=0.0, i_q_ref=0.0)
    )
    max_steps: int | None = DEFAULT_MAX_STEPS
    safety_limits: SafetyLimits | None = field(default_factory=SafetyLimits)
    on_safety_violation: Callable[[StateDict, str], None] | None = None

    _step: int = field(default=0, init=False, repr=False)
    _terminated_by_safety: bool = field(default=False, init=False, repr=False)
    _last_violation_reason: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps is not None and (
            not isinstance(self.max_steps, int) or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer or None.")
        if not hasattr(self.physics_engine, "reset") or not hasattr(
            self.physics_engine, "step"
        ):
            raise TypeError("physics_engine must provide reset() and step() methods.")
        if not callable(self.reference_generator):
            raise TypeError("reference_generator must be callable.")
        if not hasattr(self.reference_generator, "reset"):
            raise TypeError("reference_generator must provide a reset() method.")
        if self.on_safety_violation is not None and not callable(
            self.on_safety_violation
        ):
            raise TypeError("on_safety_violation must be callable or None.")

    @property
    def reference_keys(self) -> set[str]:
        return {"i_d_ref", "i_q_ref"}

    @property
    def terminated_by_safety(self) -> bool:
        """True if the last episode was terminated due to safety violation."""
        return self._terminated_by_safety

    @property
    def last_violation_reason(self) -> str | None:
        """Reason for the last safety violation, or None if no violation."""
        return self._last_violation_reason

    @classmethod
    def from_config(
        cls,
        n_rpm: float = 1000.0,
        i_d_ref: float = 0.0,
        i_q_ref: float = 2.0,
        step_time_s: float = 0.0,
        max_steps: int | None = DEFAULT_MAX_STEPS,
        config: PMSMConfig | None = None,
        safety_limits: SafetyLimits | None = None,
    ) -> "PMSMCurrentControlTask":
        """
        Factory method for creating a task with default step reference.

        Args:
            n_rpm: Motor speed in RPM.
            i_d_ref: Target d-axis current.
            i_q_ref: Target q-axis current.
            step_time_s: Time when step occurs (0 = immediate).
            max_steps: Maximum episode length.
            config: PMSM configuration (uses default if None).
            safety_limits: Safety limits (uses default if None).

        """
        physics = PMSMPhysicsEngine(n_rpm=n_rpm, config=config or PMSMConfig())
        reference = StepReference(
            i_d_ref=i_d_ref, i_q_ref=i_q_ref, step_time_s=step_time_s
        )
        return cls(
            physics_engine=physics,
            reference_generator=reference,
            max_steps=max_steps,
            safety_limits=(
                safety_limits if safety_limits is not None else SafetyLimits()
            ),
        )

    def reset(self, seed: int | None = None) -> tuple[StateDict, ReferenceDict]:
        """Reset task state and return initial observation."""
        self._step = 0
        self._terminated_by_safety = False
        self._last_violation_reason = None
        self.reference_generator.reset()
        state = self.physics_engine.reset(seed=seed)
        validate_numeric_dict(state, "state", required_keys=("time",))
        reference = self.reference_generator(self._step, state["time"])
        validate_numeric_dict(
            reference, "reference", required_keys=("i_d_ref", "i_q_ref")
        )
        return state, reference

    def step(self, action: dict[str, float]) -> tuple[StateDict, ReferenceDict, bool]:
        """
        Execute one control step.

        Safety checks are performed in two phases:
        1. Action limits BEFORE physics (prevents commanding crazy voltages)
        2. State limits AFTER physics (detects system instability)

        Args:
            action: Control action dict (v_d, v_q in Volts).

        Returns:
            Tuple of (next_state, next_reference, done).
            done=True if max_steps reached OR safety limit violated.

        """
        validate_numeric_dict(action, "action")
        has_dq = "v_d" in action and "v_q" in action
        has_ab = "v_alpha" in action and "v_beta" in action
        if not (has_dq or has_ab):
            raise KeyError(
                "action must contain either ('v_d', 'v_q') or ('v_alpha', 'v_beta')."
            )

        violation_reason: str | None = None

        # Phase 1: Check action limits BEFORE physics
        if self.safety_limits is not None:
            violation_reason = self.safety_limits.check_action(action)

        # Run physics (even if action violated - physics may clamp internally)
        next_state, _debug = self.physics_engine.step(action)
        validate_numeric_dict(next_state, "next_state", required_keys=("time",))
        self._step += 1
        reference = self.reference_generator(self._step, next_state["time"])
        validate_numeric_dict(
            reference, "reference", required_keys=("i_d_ref", "i_q_ref")
        )

        # Phase 2: Check state limits AFTER physics
        if violation_reason is None and self.safety_limits is not None:
            violation_reason = self.safety_limits.check_state(next_state)

        # Handle safety violation
        if violation_reason is not None:
            self._terminated_by_safety = True
            self._last_violation_reason = violation_reason
            if self.on_safety_violation is not None:
                self.on_safety_violation(next_state, violation_reason)

        # Episode terminates on max_steps OR safety violation
        done = violation_reason is not None or (
            self.max_steps is not None and self._step >= self.max_steps
        )

        return next_state, reference, done
