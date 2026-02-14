"""Tests for PMSM reference generators."""

from __future__ import annotations

from math import isclose

from embark.benchmark.tasks.reference_generators import (
    ConstantReference,
    SinusoidalReference,
    StepReference,
)


def test_step_reference_before_and_after_step_time():
    gen = StepReference(i_d_ref=1.0, i_q_ref=2.0, step_time_s=0.5)

    gen.reset()
    # Before step time -> zero references
    ref_early = gen(step=0, time_s=0.1)
    assert ref_early == {"i_d_ref": 0.0, "i_q_ref": 0.0}

    # After step time -> configured references
    ref_late = gen(step=1, time_s=0.5)
    assert ref_late == {"i_d_ref": 1.0, "i_q_ref": 2.0}


def test_constant_reference_returns_fixed_values():
    gen = ConstantReference(i_d_ref=0.1, i_q_ref=0.2)
    gen.reset()

    for t in [0.0, 1.0, 10.0]:
        ref = gen(step=0, time_s=t)
        assert ref == {"i_d_ref": 0.1, "i_q_ref": 0.2}


def test_sinusoidal_reference_generates_expected_wave():
    gen = SinusoidalReference(
        i_d_ref=0.0,
        i_q_amp=1.0,
        i_q_offset=0.0,
        frequency_hz=1.0,
    )
    gen.reset()

    ref0 = gen(step=0, time_s=0.0)
    ref_quarter = gen(step=1, time_s=0.25)

    assert isclose(ref0["i_d_ref"], 0.0)
    assert isclose(ref0["i_q_ref"], 0.0, abs_tol=1e-6)
    # At t=0.25s with f=1Hz -> sin(2π * 0.25) = sin(π/2) = 1
    assert isclose(ref_quarter["i_q_ref"], 1.0, rel_tol=1e-6)
