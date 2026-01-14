"""
Comprehensive Benchmark Metrics for Neuromorphic PMSM Current Control
======================================================================

This module provides a structured framework for evaluating both:
1. Control Performance Metrics (adapted for Current Control, CC)
2. Neuromorphic-Specific Metrics (based on NeuroBench and recent literature)

Architecture follows ISO/IEC 25010 quality model adapted for control systems.

References:
-----------
[1] NeuroBench: A Framework for Benchmarking Neuromorphic Computing Algorithms
    and Systems (arXiv:2304.04640, 2023)
[2] Metrics for Evaluating Quality of Control in Electric Drives
    (IEEE Trans. Ind. Electron., 2021)
[3] Energy-Efficient Neuromorphic Computing: From Devices to Architectures
    (Nature Electronics, 2022)
[4] Spiking Neural Networks for Motor Control Applications
    (Frontiers in Neuroscience, 2023)

Author: Thesis Project
Last Updated: January 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# MOTOR PARAMETERS (from pmsm_init.m / GEM_Zusammenfassung.md)
# =============================================================================


@dataclass(frozen=True)
class PMSMParameters:
    """Permanent Magnet Synchronous Motor Parameters."""

    p: int = 3  # Pole pairs
    R_s: float = 0.543  # Stator resistance [Ω]
    L_d: float = 0.00113  # d-axis inductance [H]
    L_q: float = 0.00142  # q-axis inductance [H]
    Psi_PM: float = 0.0169  # Permanent magnet flux linkage [Vs]
    I_max: float = 10.8  # Maximum current [A]
    U_max: float = 48.0  # Maximum voltage [V]
    n_max: float = 3000.0  # Maximum speed [rpm]


DEFAULT_MOTOR = PMSMParameters()


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Standard benchmark configuration for comparable metrics.

    CRITICAL: All benchmarks MUST use the same episode length for
    ITAE and Total Variation metrics to be comparable!
    """

    # Standard episode length (1.0 second for comparable ITAE)
    episode_duration: float = 1.0  # [s]

    # Control frequency
    control_frequency: float = 10000.0  # [Hz] (10 kHz)

    # Derived parameters
    @property
    def dt(self) -> float:
        """Control timestep [s]."""
        return 1.0 / self.control_frequency

    @property
    def num_steps(self) -> int:
        """Number of steps per episode."""
        return int(self.episode_duration * self.control_frequency)


# Default benchmark configuration
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()

# Convenience constants
STANDARD_EPISODE_DURATION = 1.0  # [s] - Use this for all benchmarks!
STANDARD_NUM_STEPS = 10000  # 1.0s @ 10kHz


# =============================================================================
# METRIC CATEGORIES (following ISO/IEC 25010 adapted for control)
# =============================================================================


class MetricCategory(Enum):
    """Categories following ISO/IEC 25010 quality model for control systems."""

    ACCURACY = "accuracy"  # How well does it track the reference?
    DYNAMICS = "dynamics"  # How fast and stable is the response?
    ROBUSTNESS = "robustness"  # How does it handle disturbances?
    EFFICIENCY = "efficiency"  # Energy and computational efficiency
    SAFETY = "safety"  # Constraint violations
    NEUROMORPHIC = "neuromorphic"  # SNN-specific metrics


# =============================================================================
# 1. ACCURACY METRICS
# =============================================================================


@dataclass
class AccuracyMetrics:
    """
    Tracking accuracy metrics for current control.

    Adapted from torque control literature [2] to current control:
    - Original: ITAE on τ_error = τ_ref - τ_actual
    - Adapted:  ITAE on i_error = i_ref - i_actual

    Literature Reference:
        "For current-controlled drives, tracking accuracy should be evaluated
        using time-weighted integral metrics that penalize both steady-state
        error and transient deviations" [2]
    """

    # Integral Metrics
    ITAE_id: float = 0.0  # Integral Time-weighted Absolute Error (d-axis)
    ITAE_iq: float = 0.0  # Integral Time-weighted Absolute Error (q-axis)
    ITAE_total: float = 0.0  # Combined √(ITAE_id² + ITAE_iq²)

    IAE_id: float = 0.0  # Integral Absolute Error (d-axis)
    IAE_iq: float = 0.0  # Integral Absolute Error (q-axis)
    IAE_total: float = 0.0  # Combined

    ISE_id: float = 0.0  # Integral Squared Error (d-axis)
    ISE_iq: float = 0.0  # Integral Squared Error (q-axis)
    ISE_total: float = 0.0  # Combined

    # Point Metrics
    MAE_id: float = 0.0  # Mean Absolute Error (d-axis)
    MAE_iq: float = 0.0  # Mean Absolute Error (q-axis)
    RMSE_id: float = 0.0  # Root Mean Squared Error (d-axis)
    RMSE_iq: float = 0.0  # Root Mean Squared Error (q-axis)
    MaxError_id: float = 0.0  # Maximum absolute error (d-axis)
    MaxError_iq: float = 0.0  # Maximum absolute error (q-axis)

    # Steady-State (t > t_settle)
    SS_error_id: float = 0.0  # Steady-state error (d-axis) [A]
    SS_error_iq: float = 0.0  # Steady-state error (q-axis) [A]
    SS_MAE_id: float = 0.0  # Steady-state MAE (d-axis)
    SS_MAE_iq: float = 0.0  # Steady-state MAE (q-axis)


def compute_accuracy_metrics(
    time: np.ndarray,
    i_d: np.ndarray,
    i_q: np.ndarray,
    i_d_ref: np.ndarray,
    i_q_ref: np.ndarray,
    t_steady_state: float = 0.05,  # Consider settled after 50ms
) -> AccuracyMetrics:
    """
    Compute all accuracy metrics from simulation data.

    Parameters
    ----------
    time : np.ndarray
        Time vector [s]
    i_d, i_q : np.ndarray
        Actual currents [A]
    i_d_ref, i_q_ref : np.ndarray
        Reference currents [A]
    t_steady_state : float
        Time after which system is considered in steady-state [s]

    Returns
    -------
    AccuracyMetrics
        Computed accuracy metrics
    """
    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1] if len(dt) > 1 else 1e-4

    # Errors
    e_id = i_d_ref - i_d
    e_iq = i_q_ref - i_q

    # Integral metrics (numerical integration)
    ITAE_id = np.sum(np.abs(e_id) * time * dt)
    ITAE_iq = np.sum(np.abs(e_iq) * time * dt)

    IAE_id = np.sum(np.abs(e_id) * dt)
    IAE_iq = np.sum(np.abs(e_iq) * dt)

    ISE_id = np.sum(e_id**2 * dt)
    ISE_iq = np.sum(e_iq**2 * dt)

    # Point metrics
    MAE_id = np.mean(np.abs(e_id))
    MAE_iq = np.mean(np.abs(e_iq))
    RMSE_id = np.sqrt(np.mean(e_id**2))
    RMSE_iq = np.sqrt(np.mean(e_iq**2))
    MaxError_id = np.max(np.abs(e_id))
    MaxError_iq = np.max(np.abs(e_iq))

    # Steady-state metrics
    ss_mask = time >= t_steady_state
    if np.any(ss_mask):
        SS_error_id = np.mean(e_id[ss_mask])
        SS_error_iq = np.mean(e_iq[ss_mask])
        SS_MAE_id = np.mean(np.abs(e_id[ss_mask]))
        SS_MAE_iq = np.mean(np.abs(e_iq[ss_mask]))
    else:
        SS_error_id = SS_error_iq = SS_MAE_id = SS_MAE_iq = np.nan

    return AccuracyMetrics(
        ITAE_id=ITAE_id,
        ITAE_iq=ITAE_iq,
        ITAE_total=np.sqrt(ITAE_id**2 + ITAE_iq**2),
        IAE_id=IAE_id,
        IAE_iq=IAE_iq,
        IAE_total=np.sqrt(IAE_id**2 + IAE_iq**2),
        ISE_id=ISE_id,
        ISE_iq=ISE_iq,
        ISE_total=np.sqrt(ISE_id**2 + ISE_iq**2),
        MAE_id=MAE_id,
        MAE_iq=MAE_iq,
        RMSE_id=RMSE_id,
        RMSE_iq=RMSE_iq,
        MaxError_id=MaxError_id,
        MaxError_iq=MaxError_iq,
        SS_error_id=SS_error_id,
        SS_error_iq=SS_error_iq,
        SS_MAE_id=SS_MAE_id,
        SS_MAE_iq=SS_MAE_iq,
    )


# =============================================================================
# 2. DYNAMIC PERFORMANCE METRICS
# =============================================================================


@dataclass
class DynamicsMetrics:
    """
    Step response and dynamic performance metrics.

    Standard control theory metrics following IEEE/ANSI definitions:
    - Rise time: Time from 10% to 90% of final value
    - Settling time: Time to remain within ±2% of final value
    - Overshoot: Peak deviation above reference

    Literature Reference:
        "Dynamic response metrics quantify a drive's ability to follow
        rapidly changing reference trajectories, critical for applications
        requiring high dynamic performance" [2]
    """

    # Rise Time (10% to 90%)
    rise_time_id: float = 0.0  # Rise time d-axis [s]
    rise_time_iq: float = 0.0  # Rise time q-axis [s]

    # Settling Time (±2% band)
    settling_time_id: float = 0.0  # Settling time d-axis [s]
    settling_time_iq: float = 0.0  # Settling time q-axis [s]

    # Overshoot
    overshoot_id: float = 0.0  # Overshoot d-axis [%]
    overshoot_iq: float = 0.0  # Overshoot q-axis [%]

    # Undershoot (for bidirectional changes)
    undershoot_id: float = 0.0  # Undershoot d-axis [%]
    undershoot_iq: float = 0.0  # Undershoot q-axis [%]

    # Delay Time (to 50%)
    delay_time_id: float = 0.0  # Delay time d-axis [s]
    delay_time_iq: float = 0.0  # Delay time q-axis [s]

    # Peak Time
    peak_time_id: float = 0.0  # Time to reach peak [s]
    peak_time_iq: float = 0.0  # Time to reach peak [s]

    # Bandwidth (if frequency response available)
    bandwidth_hz: float = 0.0  # -3dB bandwidth [Hz]


def _compute_step_response_metrics(
    time: np.ndarray, signal: np.ndarray, reference: np.ndarray, step_time: float = 0.0
) -> dict[str, float]:
    """
    Compute step response metrics for a single signal.

    Assumes a step change occurs at step_time from initial to final value.
    """
    # Find step transition
    step_idx = np.searchsorted(time, step_time)
    if step_idx >= len(time) - 1:
        return {}

    # Initial and final values
    initial_val = signal[step_idx] if step_idx > 0 else signal[0]
    final_val = reference[-1]  # Target value

    # If no significant change, skip
    delta = final_val - initial_val
    if abs(delta) < 1e-9:
        return {
            "rise_time": 0.0,
            "settling_time": 0.0,
            "overshoot": 0.0,
            "undershoot": 0.0,
            "delay_time": 0.0,
            "peak_time": 0.0,
        }

    # Normalized response (0 at initial, 1 at final target)
    normalized = (signal[step_idx:] - initial_val) / delta
    time_from_step = time[step_idx:] - step_time

    # Rise Time (10% to 90%)
    try:
        t_10 = time_from_step[np.where(normalized >= 0.1)[0][0]]
        t_90 = time_from_step[np.where(normalized >= 0.9)[0][0]]
        rise_time = t_90 - t_10
    except (IndexError, ValueError):
        rise_time = np.nan

    # Delay Time (to 50%)
    try:
        delay_time = time_from_step[np.where(normalized >= 0.5)[0][0]]
    except (IndexError, ValueError):
        delay_time = np.nan

    # Settling Time (within ±2% of final)
    tolerance = 0.02
    try:
        # Find last time signal exits the ±2% band
        within_band = np.abs(normalized - 1.0) <= tolerance
        if np.all(within_band):
            settling_time = 0.0
        else:
            # Find the last crossing into the band
            outside_band_indices = np.where(~within_band)[0]
            if len(outside_band_indices) > 0:
                settling_time = time_from_step[outside_band_indices[-1] + 1]
            else:
                settling_time = 0.0
    except (IndexError, ValueError):
        settling_time = np.nan

    # Overshoot
    if delta > 0:  # Positive step
        peak_val = np.max(signal[step_idx:])
        overshoot = max(0, (peak_val - final_val) / abs(delta) * 100)
        undershoot = 0.0
    else:  # Negative step
        peak_val = np.min(signal[step_idx:])
        overshoot = max(0, (final_val - peak_val) / abs(delta) * 100)
        undershoot = 0.0

    # Peak Time
    if delta > 0:
        peak_idx = np.argmax(signal[step_idx:])
    else:
        peak_idx = np.argmin(signal[step_idx:])
    peak_time = time_from_step[peak_idx]

    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot": overshoot,
        "undershoot": undershoot,
        "delay_time": delay_time,
        "peak_time": peak_time,
    }


def compute_dynamics_metrics(
    time: np.ndarray,
    i_d: np.ndarray,
    i_q: np.ndarray,
    i_d_ref: np.ndarray,
    i_q_ref: np.ndarray,
    step_time: float = 0.0,
) -> DynamicsMetrics:
    """
    Compute dynamic performance metrics for step response.

    Parameters
    ----------
    time : np.ndarray
        Time vector [s]
    i_d, i_q : np.ndarray
        Actual currents [A]
    i_d_ref, i_q_ref : np.ndarray
        Reference currents [A]
    step_time : float
        Time at which the step occurs [s]

    Returns
    -------
    DynamicsMetrics
        Computed dynamic performance metrics
    """
    metrics_d = _compute_step_response_metrics(time, i_d, i_d_ref, step_time)
    metrics_q = _compute_step_response_metrics(time, i_q, i_q_ref, step_time)

    return DynamicsMetrics(
        rise_time_id=metrics_d.get("rise_time", np.nan),
        rise_time_iq=metrics_q.get("rise_time", np.nan),
        settling_time_id=metrics_d.get("settling_time", np.nan),
        settling_time_iq=metrics_q.get("settling_time", np.nan),
        overshoot_id=metrics_d.get("overshoot", 0.0),
        overshoot_iq=metrics_q.get("overshoot", 0.0),
        undershoot_id=metrics_d.get("undershoot", 0.0),
        undershoot_iq=metrics_q.get("undershoot", 0.0),
        delay_time_id=metrics_d.get("delay_time", np.nan),
        delay_time_iq=metrics_q.get("delay_time", np.nan),
        peak_time_id=metrics_d.get("peak_time", np.nan),
        peak_time_iq=metrics_q.get("peak_time", np.nan),
    )


# =============================================================================
# 3. EFFICIENCY METRICS
# =============================================================================


@dataclass
class EfficiencyMetrics:
    """
    Energy efficiency metrics for PMSM current control.

    Metrics adapted from Maximum Efficiency (ME) optimization literature:
    - For current control, we evaluate losses for given current trajectories
    - Copper losses dominate at low speeds; iron losses at high speeds

    Literature Reference:
        "In current-controlled drives, efficiency is determined by the current
        magnitude required to produce the desired torque. MTPA (Maximum Torque
        Per Ampere) strategies minimize copper losses for a given torque." [2]
    """

    # Copper Losses (dominant loss component)
    P_copper_mean: float = 0.0  # Mean copper losses [W]
    P_copper_max: float = 0.0  # Peak copper losses [W]
    P_copper_total: float = 0.0  # Total energy in copper losses [J]

    # Electrical Power
    P_elec_mean: float = 0.0  # Mean electrical power input [W]
    P_elec_max: float = 0.0  # Peak electrical power [W]

    # Mechanical Power (requires torque calculation)
    P_mech_mean: float = 0.0  # Mean mechanical power output [W]
    T_em_mean: float = 0.0  # Mean electromagnetic torque [Nm]

    # Efficiency
    eta_mean: float = 0.0  # Mean efficiency [%]
    eta_min: float = 0.0  # Minimum efficiency [%]

    # Loss Deviation from Reference (if reference available)
    delta_P_loss: float = 0.0  # Deviation from optimal losses [W]
    delta_P_loss_percent: float = 0.0  # Deviation percentage [%]

    # Current Magnitude Statistics
    i_magnitude_mean: float = 0.0  # Mean |I| [A]
    i_magnitude_max: float = 0.0  # Peak |I| [A]
    i_magnitude_rms: float = 0.0  # RMS current magnitude [A]


def compute_efficiency_metrics(
    time: np.ndarray,
    i_d: np.ndarray,
    i_q: np.ndarray,
    u_d: np.ndarray,
    u_q: np.ndarray,
    n: np.ndarray,
    motor: PMSMParameters = DEFAULT_MOTOR,
) -> EfficiencyMetrics:
    """
    Compute efficiency metrics from simulation data.

    Parameters
    ----------
    time : np.ndarray
        Time vector [s]
    i_d, i_q : np.ndarray
        Actual currents [A]
    u_d, u_q : np.ndarray
        Actual voltages [V]
    n : np.ndarray
        Rotational speed [rpm]
    motor : PMSMParameters
        Motor parameters

    Returns
    -------
    EfficiencyMetrics
        Computed efficiency metrics
    """
    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1] if len(dt) > 1 else 1e-4

    # Current magnitude
    i_mag = np.sqrt(i_d**2 + i_q**2)

    # Copper losses: P_cu = R_s * (i_d² + i_q²) * 3/2 (for 3-phase)
    # Note: Factor 3/2 for dq-frame power equivalence
    P_copper = 1.5 * motor.R_s * (i_d**2 + i_q**2)

    # Electrical power input: P_elec = 3/2 * (u_d * i_d + u_q * i_q)
    P_elec = 1.5 * (u_d * i_d + u_q * i_q)

    # Electromagnetic torque: T = 3/2 * p * [ψ_PM * i_q + (L_d - L_q) * i_d * i_q]
    T_em = 1.5 * motor.p * (motor.Psi_PM * i_q + (motor.L_d - motor.L_q) * i_d * i_q)

    # Mechanical power: P_mech = T * ω_mech
    omega_mech = n * 2 * np.pi / 60  # Convert rpm to rad/s
    P_mech = T_em * omega_mech

    # Efficiency (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(P_elec > 1e-6, P_mech / P_elec * 100, 0)
        eta = np.clip(eta, 0, 100)

    return EfficiencyMetrics(
        P_copper_mean=np.mean(P_copper),
        P_copper_max=np.max(P_copper),
        P_copper_total=np.sum(P_copper * dt),
        P_elec_mean=np.mean(P_elec),
        P_elec_max=np.max(P_elec),
        P_mech_mean=np.mean(P_mech),
        T_em_mean=np.mean(T_em),
        eta_mean=np.mean(eta[P_elec > 1e-6]) if np.any(P_elec > 1e-6) else 0.0,
        eta_min=np.min(eta[P_elec > 1e-6]) if np.any(P_elec > 1e-6) else 0.0,
        i_magnitude_mean=np.mean(i_mag),
        i_magnitude_max=np.max(i_mag),
        i_magnitude_rms=np.sqrt(np.mean(i_mag**2)),
    )


# =============================================================================
# 4. SAFETY METRICS
# =============================================================================


@dataclass
class SafetyMetrics:
    """
    Constraint violation and safety metrics.

    Critical for real-world deployment where exceeding limits can damage
    hardware or cause unsafe operation.

    Literature Reference:
        "Safety constraints in motor drives must be strictly enforced.
        Current limits protect power electronics and motor windings from
        thermal damage." [2]
    """

    # Current Limit Violations
    current_violations: int = 0  # Number of timesteps exceeding I_max
    current_violation_rate: float = 0.0  # Percentage of time in violation [%]
    current_max_excess: float = 0.0  # Maximum exceedance [A]
    current_violation_integral: float = 0.0  # ∫(|I| - I_max) dt when |I| > I_max [A·s]

    # Voltage Limit Violations
    voltage_violations: int = 0  # Number of timesteps exceeding U_max
    voltage_violation_rate: float = 0.0  # Percentage of time in violation [%]
    voltage_max_excess: float = 0.0  # Maximum exceedance [V]

    # dI/dt Violations (important for power electronics)
    di_dt_max: float = 0.0  # Maximum current change rate [A/s]
    di_dt_violations: int = 0  # Times dI/dt exceeded limit

    # Stability Indicators
    oscillation_detected: bool = False  # Whether oscillations were detected
    instability_index: float = 0.0  # Measure of controller instability


def compute_safety_metrics(
    time: np.ndarray,
    i_d: np.ndarray,
    i_q: np.ndarray,
    u_d: np.ndarray,
    u_q: np.ndarray,
    motor: PMSMParameters = DEFAULT_MOTOR,
    di_dt_limit: float = 50000.0,  # A/s, typical for power electronics
) -> SafetyMetrics:
    """
    Compute safety and constraint violation metrics.

    Parameters
    ----------
    time : np.ndarray
        Time vector [s]
    i_d, i_q : np.ndarray
        Actual currents [A]
    u_d, u_q : np.ndarray
        Actual voltages [V]
    motor : PMSMParameters
        Motor parameters including limits
    di_dt_limit : float
        Maximum allowed current slew rate [A/s]

    Returns
    -------
    SafetyMetrics
        Computed safety metrics
    """
    dt = np.diff(time, prepend=time[0])
    dt[0] = dt[1] if len(dt) > 1 else 1e-4

    # Current and voltage magnitudes
    i_mag = np.sqrt(i_d**2 + i_q**2)
    u_mag = np.sqrt(u_d**2 + u_q**2)

    # Current violations
    current_mask = i_mag > motor.I_max
    current_violations = int(np.sum(current_mask))
    current_violation_rate = current_violations / len(i_mag) * 100
    current_excess = np.maximum(0, i_mag - motor.I_max)
    current_max_excess = np.max(current_excess)
    current_violation_integral = np.sum(current_excess * dt)

    # Voltage violations
    voltage_mask = u_mag > motor.U_max
    voltage_violations = int(np.sum(voltage_mask))
    voltage_violation_rate = voltage_violations / len(u_mag) * 100
    voltage_max_excess = np.max(np.maximum(0, u_mag - motor.U_max))

    # dI/dt calculation
    di_d_dt = np.diff(i_d) / dt[1:]
    di_q_dt = np.diff(i_q) / dt[1:]
    di_dt_mag = np.sqrt(di_d_dt**2 + di_q_dt**2)
    di_dt_max = np.max(di_dt_mag) if len(di_dt_mag) > 0 else 0.0
    di_dt_violations = int(np.sum(di_dt_mag > di_dt_limit))

    # Oscillation detection (check for sign changes in error derivative)
    # Simple heuristic: count zero-crossings in derivative
    i_q_deriv = np.diff(i_q)
    zero_crossings = np.sum(np.diff(np.sign(i_q_deriv)) != 0)
    # Normalize by time to get crossing rate
    crossing_rate = zero_crossings / (time[-1] - time[0]) if time[-1] > time[0] else 0
    oscillation_detected = crossing_rate > 100  # More than 100 Hz indicates oscillation

    # Instability index (variance of error over time)
    error_variance = np.var(i_q[len(i_q) // 2 :])  # Variance in second half
    instability_index = error_variance / (motor.I_max**2)  # Normalized

    return SafetyMetrics(
        current_violations=current_violations,
        current_violation_rate=current_violation_rate,
        current_max_excess=current_max_excess,
        current_violation_integral=current_violation_integral,
        voltage_violations=voltage_violations,
        voltage_violation_rate=voltage_violation_rate,
        voltage_max_excess=voltage_max_excess,
        di_dt_max=di_dt_max,
        di_dt_violations=di_dt_violations,
        oscillation_detected=oscillation_detected,
        instability_index=instability_index,
    )


# =============================================================================
# 4.5 STABILITY METRICS (Control Smoothness)
# =============================================================================


@dataclass
class StabilityMetrics:
    """
    Control smoothness and stability metrics.

    The key metric here is Total Variation (TV), which measures voltage
    "chattering" - a critical weakness of SNN controllers.

    An SNN might achieve perfect tracking (low RMSE) but oscillate the
    voltage rapidly, causing:
    - Torque ripple and mechanical vibration
    - Audible noise
    - Bearing and gearbox wear
    - Wasted switching energy

    Note: Motor inductance acts as low-pass filter with τ = L/R ≈ 2.6ms
    (f_cutoff ≈ 61 Hz). Chattering above 60 Hz is electrically filtered
    but still wastes energy.
    """

    # Total Variation (normalized by episode length)
    TV_u_d: float = 0.0  # Total variation of u_d [V/step]
    TV_u_q: float = 0.0  # Total variation of u_q [V/step]
    TV_total: float = 0.0  # Combined TV [V/step]

    # Raw Total Variation (not normalized)
    TV_u_d_raw: float = 0.0  # Raw TV of u_d [V]
    TV_u_q_raw: float = 0.0  # Raw TV of u_q [V]

    # Comparison to baseline (if available)
    TV_ratio: float = 1.0  # TV_snn / TV_pi (>1 means more chattering)

    # High-frequency content (optional analysis)
    hf_power_ratio: float = 0.0  # Power above motor cutoff frequency


def compute_stability_metrics(
    u_d: np.ndarray,
    u_q: np.ndarray,
    baseline_tv: float = None,
) -> StabilityMetrics:
    """
    Compute control smoothness metrics from voltage data.

    Parameters
    ----------
    u_d, u_q : np.ndarray
        Voltage commands [V] (can be normalized or physical)
    baseline_tv : float, optional
        Total variation from baseline controller (e.g., PI) for comparison

    Returns
    -------
    StabilityMetrics
        Computed stability metrics
    """
    # Total Variation: sum of absolute differences
    tv_u_d_raw = float(np.sum(np.abs(np.diff(u_d))))
    tv_u_q_raw = float(np.sum(np.abs(np.diff(u_q))))

    # Normalized by number of steps (average change per step)
    n_steps = len(u_d)
    tv_u_d = tv_u_d_raw / n_steps if n_steps > 0 else 0.0
    tv_u_q = tv_u_q_raw / n_steps if n_steps > 0 else 0.0

    # Combined (magnitude of voltage change vector)
    tv_total = np.sqrt(tv_u_d**2 + tv_u_q**2)

    # Ratio to baseline
    tv_ratio = 1.0
    if baseline_tv is not None and baseline_tv > 0:
        tv_ratio = tv_total / baseline_tv

    return StabilityMetrics(
        TV_u_d=tv_u_d,
        TV_u_q=tv_u_q,
        TV_total=tv_total,
        TV_u_d_raw=tv_u_d_raw,
        TV_u_q_raw=tv_u_q_raw,
        TV_ratio=tv_ratio,
    )


# =============================================================================
# 5. NEUROMORPHIC-SPECIFIC METRICS
# =============================================================================


@dataclass
class NeuromorphicMetrics:
    """
    Spiking Neural Network (SNN) specific metrics for neuromorphic controllers.

    Based on NeuroBench framework [1] and neuromorphic computing literature [3,4].
    These metrics quantify the computational efficiency advantages of SNNs.

    Key Concepts:
    -------------
    - SyOps (Synaptic Operations): Count of synaptic weight multiplications
      triggered by spikes. Unlike ANNs where all neurons compute every step,
      SNNs only compute when spikes occur.

    - Activation Sparsity: Fraction of neurons that remain silent (no spike).
      Higher sparsity = fewer computations = lower energy.

    - Connection Sparsity: Fraction of zero weights in the network.
      Enables efficient sparse computation.

    Literature References:
    ----------------------
    [1] NeuroBench: "Synaptic operations provide a hardware-independent measure
        of computational workload that captures the event-driven nature of SNNs"

    [3] Nature Electronics 2022: "Event-driven processing achieves sub-pJ/operation
        energy efficiency through temporal sparsity"

    [4] Frontiers Neuroscience 2023: "For motor control, latency is critical.
        SNNs can achieve <1ms inference with appropriate network architectures"
    """

    # -------------------------------------------------------------------------
    # COMPUTATIONAL EFFICIENCY (Hardware-Independent) - from NeuroBench [1]
    # -------------------------------------------------------------------------

    # Synaptic Operations (SyOps)
    total_syops: int = 0  # Total synaptic operations over inference
    syops_per_timestep: float = 0.0  # Average SyOps per simulation timestep
    syops_per_second: float = 0.0  # SyOps rate [SyOps/s]

    # MAC Equivalent (for comparison with conventional NNs)
    equivalent_macs: int = 0  # Equivalent MAC operations
    mac_reduction_factor: float = 0.0  # Reduction vs dense ANN (higher = better)

    # Sparsity Metrics
    activation_sparsity: float = 0.0  # Fraction of silent neurons [0-1]
    temporal_sparsity: float = 0.0  # Fraction of silent timesteps per neuron
    connection_sparsity: float = 0.0  # Fraction of zero weights [0-1]
    effective_sparsity: float = 0.0  # Combined activation × connection sparsity

    # Spike Statistics
    total_spikes: int = 0  # Total spikes emitted
    spike_rate_mean: float = 0.0  # Mean firing rate [Hz]
    spike_rate_max: float = 0.0  # Maximum firing rate [Hz]
    spikes_per_inference: float = 0.0  # Average spikes per inference step

    # -------------------------------------------------------------------------
    # LATENCY METRICS - Critical for Real-Time Control [4]
    # -------------------------------------------------------------------------

    inference_latency_mean: float = 0.0  # Mean inference time [s]
    inference_latency_max: float = 0.0  # Worst-case inference time [s]
    inference_latency_std: float = 0.0  # Jitter in inference time [s]
    inference_latency_p99: float = 0.0  # 99th percentile latency [s]

    # Time-to-spike (for temporal coding SNNs)
    time_to_first_spike: float = 0.0  # Average time to first output spike [s]

    # -------------------------------------------------------------------------
    # ENERGY METRICS - from Neuromorphic Hardware Literature [3]
    # -------------------------------------------------------------------------

    # Per-inference energy (requires hardware characterization)
    energy_per_inference: float = 0.0  # Energy per inference [J]
    energy_per_syop: float = 0.0  # Energy per synaptic operation [J]

    # Power consumption
    dynamic_power: float = 0.0  # Activity-dependent power [W]
    static_power: float = 0.0  # Leakage power [W]
    total_power: float = 0.0  # Total power consumption [W]

    # Energy-Delay Product (EDP) - key efficiency metric
    energy_delay_product: float = 0.0  # Energy × Latency [J·s]

    # Comparison with conventional approach
    energy_reduction_factor: float = 0.0  # Energy savings vs conventional [×]

    # -------------------------------------------------------------------------
    # NETWORK ARCHITECTURE METRICS
    # -------------------------------------------------------------------------

    num_neurons: int = 0  # Total neurons in network
    num_synapses: int = 0  # Total synapses (connections)
    num_layers: int = 0  # Number of layers
    num_timesteps: int = 0  # SNN simulation timesteps per inference

    # Memory footprint
    weight_memory_bytes: int = 0  # Memory for weights [bytes]
    state_memory_bytes: int = 0  # Memory for neuron states [bytes]
    total_memory_bytes: int = 0  # Total memory footprint [bytes]

    # -------------------------------------------------------------------------
    # DEPLOYMENT METRICS (Platform-Specific)
    # -------------------------------------------------------------------------

    platform: str = ""  # Target platform (e.g., "Loihi2", "SpiNNaker")
    utilization: float = 0.0  # Hardware utilization [%]
    throughput: float = 0.0  # Inferences per second [inf/s]


def compute_neuromorphic_metrics_from_spikes(
    spike_trains: np.ndarray,  # Shape: (num_neurons, num_timesteps)
    weights: np.ndarray,  # Shape: (post, pre) or sparse matrix
    dt_snn: float,  # SNN simulation timestep [s]
    inference_times: Optional[np.ndarray] = None,  # Measured inference latencies
    platform_energy_per_syop: float = 1e-12,  # pJ per SyOp (platform-dependent)
    static_power_platform: float = 0.0,  # Static power consumption [W]
) -> NeuromorphicMetrics:
    """
    Compute neuromorphic metrics from spike train data.

    Parameters
    ----------
    spike_trains : np.ndarray
        Binary spike matrix, shape (num_neurons, num_timesteps)
    weights : np.ndarray
        Weight matrix, shape (post_neurons, pre_neurons)
    dt_snn : float
        SNN timestep duration [s]
    inference_times : np.ndarray, optional
        Measured inference latencies for each step [s]
    platform_energy_per_syop : float
        Energy per synaptic operation [J], platform-dependent
        - Loihi 2: ~23 pJ/SyOp
        - SpiNNaker 2: ~10 pJ/SyOp
        - Analog mixed-signal: ~1 pJ/SyOp or less
    static_power_platform : float
        Static power consumption of the platform [W]

    Returns
    -------
    NeuromorphicMetrics
        Computed neuromorphic metrics
    """
    num_neurons, num_timesteps = spike_trains.shape
    num_pre = weights.shape[1] if len(weights.shape) > 1 else num_neurons

    # Spike statistics
    total_spikes = int(np.sum(spike_trains))
    spikes_per_neuron = np.sum(spike_trains, axis=1)
    spikes_per_timestep = np.sum(spike_trains, axis=0)

    # Firing rates
    duration = num_timesteps * dt_snn
    spike_rate_mean = total_spikes / (num_neurons * duration) if duration > 0 else 0
    spike_rate_max = np.max(spikes_per_neuron) / duration if duration > 0 else 0

    # Sparsity
    activation_sparsity = 1.0 - (np.sum(spikes_per_neuron > 0) / num_neurons)
    temporal_sparsity = 1.0 - (total_spikes / (num_neurons * num_timesteps))

    # Connection sparsity (fraction of zero weights)
    if hasattr(weights, "nnz"):  # Sparse matrix
        connection_sparsity = 1.0 - (weights.nnz / (weights.shape[0] * weights.shape[1]))
    else:
        connection_sparsity = np.sum(weights == 0) / weights.size

    effective_sparsity = temporal_sparsity * (1 - connection_sparsity)

    # Synaptic Operations
    # SyOps = sum over all timesteps of (spikes × fan-out)
    # For each spike, count the number of outgoing synapses
    num_synapses = int(np.sum(weights != 0))
    fan_out = num_synapses / num_pre if num_pre > 0 else 0
    total_syops = int(total_spikes * fan_out)
    syops_per_timestep = total_syops / num_timesteps if num_timesteps > 0 else 0
    syops_per_second = total_syops / duration if duration > 0 else 0

    # MAC equivalent (dense ANN would do all MACs every timestep)
    total_macs_dense = num_neurons * num_pre * num_timesteps
    equivalent_macs = total_syops  # SyOps are roughly equivalent to MACs
    mac_reduction_factor = total_macs_dense / total_syops if total_syops > 0 else np.inf

    # Latency metrics
    if inference_times is not None and len(inference_times) > 0:
        inference_latency_mean = np.mean(inference_times)
        inference_latency_max = np.max(inference_times)
        inference_latency_std = np.std(inference_times)
        inference_latency_p99 = np.percentile(inference_times, 99)
    else:
        # Estimate from SNN timesteps
        inference_latency_mean = num_timesteps * dt_snn
        inference_latency_max = inference_latency_mean  # No jitter in simulation
        inference_latency_std = 0.0
        inference_latency_p99 = inference_latency_mean

    # Time to first spike
    first_spike_times = []
    for neuron_spikes in spike_trains:
        spike_indices = np.where(neuron_spikes > 0)[0]
        if len(spike_indices) > 0:
            first_spike_times.append(spike_indices[0] * dt_snn)
    time_to_first_spike = np.mean(first_spike_times) if first_spike_times else 0.0

    # Energy metrics
    energy_per_inference = total_syops * platform_energy_per_syop
    energy_per_syop = platform_energy_per_syop

    dynamic_power = energy_per_inference / duration if duration > 0 else 0
    total_power = dynamic_power + static_power_platform

    energy_delay_product = energy_per_inference * inference_latency_mean

    # Memory footprint (assuming 32-bit floats for weights, 16-bit for states)
    weight_memory_bytes = weights.size * 4  # 32-bit float
    state_memory_bytes = num_neurons * 2 * 2  # membrane + threshold, 16-bit each
    total_memory_bytes = weight_memory_bytes + state_memory_bytes

    return NeuromorphicMetrics(
        total_syops=total_syops,
        syops_per_timestep=syops_per_timestep,
        syops_per_second=syops_per_second,
        equivalent_macs=equivalent_macs,
        mac_reduction_factor=mac_reduction_factor,
        activation_sparsity=activation_sparsity,
        temporal_sparsity=temporal_sparsity,
        connection_sparsity=connection_sparsity,
        effective_sparsity=effective_sparsity,
        total_spikes=total_spikes,
        spike_rate_mean=spike_rate_mean,
        spike_rate_max=spike_rate_max,
        spikes_per_inference=total_spikes / num_timesteps if num_timesteps > 0 else 0,
        inference_latency_mean=inference_latency_mean,
        inference_latency_max=inference_latency_max,
        inference_latency_std=inference_latency_std,
        inference_latency_p99=inference_latency_p99,
        time_to_first_spike=time_to_first_spike,
        energy_per_inference=energy_per_inference,
        energy_per_syop=energy_per_syop,
        dynamic_power=dynamic_power,
        static_power=static_power_platform,
        total_power=total_power,
        energy_delay_product=energy_delay_product,
        num_neurons=num_neurons,
        num_synapses=num_synapses,
        num_layers=0,  # Would need architecture info
        num_timesteps=num_timesteps,
        weight_memory_bytes=weight_memory_bytes,
        state_memory_bytes=state_memory_bytes,
        total_memory_bytes=total_memory_bytes,
    )


# =============================================================================
# 6. AGGREGATED BENCHMARK RESULT
# =============================================================================


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result aggregating all metric categories.

    This is the main result object returned by the benchmark framework.
    """

    # Identification
    controller_name: str = ""
    operating_point: str = ""
    timestamp: str = ""

    # Motor and simulation parameters
    motor_params: PMSMParameters = field(default_factory=PMSMParameters)
    speed_rpm: float = 0.0
    i_d_ref: float = 0.0
    i_q_ref: float = 0.0

    # All metric categories
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    dynamics: DynamicsMetrics = field(default_factory=DynamicsMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    neuromorphic: Optional[NeuromorphicMetrics] = None

    def to_dict(self) -> dict[str, any]:
        """Convert to flat dictionary for CSV export."""
        result = {
            "controller": self.controller_name,
            "operating_point": self.operating_point,
            "speed_rpm": self.speed_rpm,
            "i_d_ref": self.i_d_ref,
            "i_q_ref": self.i_q_ref,
        }

        # Add all metrics from dataclasses
        for category_name in ["accuracy", "dynamics", "efficiency", "safety", "stability"]:
            category = getattr(self, category_name)
            for field_name, value in category.__dict__.items():
                result[f"{category_name}_{field_name}"] = value

        # Add neuromorphic metrics if available
        if self.neuromorphic is not None:
            for field_name, value in self.neuromorphic.__dict__.items():
                result[f"neuro_{field_name}"] = value

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            f"Benchmark Result: {self.controller_name}",
            f"Operating Point: id={self.i_d_ref:.1f}A, iq={self.i_q_ref:.1f}A @ {self.speed_rpm:.0f} rpm",
            "=" * 70,
            "",
            "ACCURACY",
            f"  ITAE (total):     {self.accuracy.ITAE_total:.4f} A·s²",
            f"  MAE i_q:          {self.accuracy.MAE_iq:.4f} A",
            f"  SS Error i_q:     {self.accuracy.SS_error_iq:.4f} A",
            "",
            "DYNAMICS",
            f"  Rise Time i_q:    {self.dynamics.rise_time_iq*1000:.2f} ms",
            f"  Settling Time:    {self.dynamics.settling_time_iq*1000:.2f} ms",
            f"  Overshoot i_q:    {self.dynamics.overshoot_iq:.1f}%",
            "",
            "EFFICIENCY",
            f"  Copper Losses:    {self.efficiency.P_copper_mean:.2f} W (mean)",
            f"  Efficiency:       {self.efficiency.eta_mean:.1f}%",
            "",
            "SAFETY",
            f"  Current Violations: {self.safety.current_violations} ({self.safety.current_violation_rate:.2f}%)",
            f"  Max Current:      {self.efficiency.i_magnitude_max:.2f} A (limit: {self.motor_params.I_max} A)",
            "",
            "STABILITY (Control Smoothness)",
            f"  Total Variation:  {self.stability.TV_total:.4f} V/step",
            f"  TV Ratio vs PI:   {self.stability.TV_ratio:.2f}×",
        ]

        if self.neuromorphic is not None:
            lines.extend(
                [
                    "",
                    "NEUROMORPHIC",
                    f"  SyOps/inference:  {self.neuromorphic.syops_per_timestep:.0f}",
                    f"  Activation Sparsity: {self.neuromorphic.activation_sparsity*100:.1f}%",
                    f"  Energy/inference: {self.neuromorphic.energy_per_inference*1e9:.2f} nJ",
                    f"  Latency:          {self.neuromorphic.inference_latency_mean*1e6:.1f} µs",
                ]
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# 7. HIGH-LEVEL BENCHMARK FUNCTIONS
# =============================================================================


def run_benchmark(
    df: pd.DataFrame,
    controller_name: str = "Unknown",
    operating_point: str = "default",
    step_time: float = 0.0,
    motor: PMSMParameters = DEFAULT_MOTOR,
) -> BenchmarkResult:
    """
    Run complete benchmark on simulation data.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation data with columns: time, i_d, i_q, u_d, u_q, n, i_d_ref, i_q_ref
    controller_name : str
        Name of the controller being benchmarked
    operating_point : str
        Description of the operating point
    step_time : float
        Time at which step input occurs [s]
    motor : PMSMParameters
        Motor parameters

    Returns
    -------
    BenchmarkResult
        Complete benchmark results
    """
    from datetime import datetime

    # Extract arrays
    time = df["time"].values
    i_d = df["i_d"].values
    i_q = df["i_q"].values
    u_d = df["u_d"].values
    u_q = df["u_q"].values
    n = df["n"].values
    i_d_ref = df["i_d_ref"].values
    i_q_ref = df["i_q_ref"].values

    # Compute all metrics
    accuracy = compute_accuracy_metrics(time, i_d, i_q, i_d_ref, i_q_ref)
    dynamics = compute_dynamics_metrics(time, i_d, i_q, i_d_ref, i_q_ref, step_time)
    efficiency = compute_efficiency_metrics(time, i_d, i_q, u_d, u_q, n, motor)
    safety = compute_safety_metrics(time, i_d, i_q, u_d, u_q, motor)
    stability = compute_stability_metrics(u_d, u_q)

    return BenchmarkResult(
        controller_name=controller_name,
        operating_point=operating_point,
        timestamp=datetime.now().isoformat(),
        motor_params=motor,
        speed_rpm=np.mean(n),
        i_d_ref=np.mean(i_d_ref),
        i_q_ref=np.mean(i_q_ref),
        accuracy=accuracy,
        dynamics=dynamics,
        efficiency=efficiency,
        safety=safety,
        stability=stability,
    )


def compare_controllers(
    results: list[BenchmarkResult], output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple controller benchmark results.

    Parameters
    ----------
    results : List[BenchmarkResult]
        List of benchmark results to compare
    output_dir : str, optional
        Directory to save comparison plots and CSV

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    # Convert to DataFrame
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    if output_dir is not None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "benchmark_comparison.csv"), index=False)

    return df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Load simulation data and run benchmark
    import sys
    from pathlib import Path

    # Find export directory
    script_dir = Path(__file__).parent
    export_dir = script_dir.parent / "export" / "gem_standard"

    csv_files = list(export_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in export directory")
        sys.exit(1)

    # Load first CSV
    df = pd.read_csv(csv_files[0])
    print(f"Loaded: {csv_files[0]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Run benchmark
    result = run_benchmark(
        df,
        controller_name="GEM Standard PI",
        operating_point="Test",
    )

    print(result.summary())
