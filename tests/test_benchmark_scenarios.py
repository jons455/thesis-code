"""Tests for new benchmark scenarios and MultiStepReference generator."""

from __future__ import annotations

import pytest

from embark.benchmark.harness import (
    QUICK_SCENARIOS,
    STANDARD_SCENARIOS,
    BenchmarkSuite,
)
from embark.benchmark.tasks.reference_generators import MultiStepReference


class TestMultiStepReference:
    """Test the new MultiStepReference generator."""

    def test_initialization(self):
        """Test MultiStepReference can be created."""
        steps = [(0.0, 0.0, 0.0), (0.1, 0.0, 2.0), (0.3, -1.0, 2.0)]
        gen = MultiStepReference(steps=steps)
        assert gen.steps == steps

    def test_initial_step_returns_first_values(self):
        """Test that t=0 returns first step values."""
        gen = MultiStepReference(steps=[(0.0, 0.0, 0.0), (0.1, 0.0, 2.0)])
        gen.reset()
        
        ref = gen(step=0, time_s=0.0)
        assert ref["i_d_ref"] == 0.0
        assert ref["i_q_ref"] == 0.0

    def test_transitions_to_next_step(self):
        """Test that reference transitions at specified time."""
        gen = MultiStepReference(steps=[
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 2.0),
            (0.3, -1.0, 3.0),
        ])
        gen.reset()

        # Before first transition
        ref1 = gen(step=0, time_s=0.05)
        assert ref1["i_d_ref"] == 0.0
        assert ref1["i_q_ref"] == 0.0

        # After first transition
        ref2 = gen(step=1, time_s=0.15)
        assert ref2["i_d_ref"] == 0.0
        assert ref2["i_q_ref"] == 2.0

        # After second transition
        ref3 = gen(step=2, time_s=0.35)
        assert ref3["i_d_ref"] == -1.0
        assert ref3["i_q_ref"] == 3.0

    def test_holds_last_value(self):
        """Test that reference holds at last step value."""
        gen = MultiStepReference(steps=[(0.0, 0.0, 0.0), (0.1, 1.0, 2.0)])
        gen.reset()

        # Well beyond last transition
        ref = gen(step=1000, time_s=10.0)
        assert ref["i_d_ref"] == 1.0
        assert ref["i_q_ref"] == 2.0

    def test_negative_currents(self):
        """Test that negative currents work (for generating quadrant)."""
        gen = MultiStepReference(steps=[
            (0.0, 0.0, 2.0),
            (0.1, 0.0, -2.0),  # Negative i_q
        ])
        gen.reset()

        ref = gen(step=10, time_s=0.15)
        assert ref["i_d_ref"] == 0.0
        assert ref["i_q_ref"] == -2.0


class TestStandardScenarios:
    """Test the 6 standard benchmark scenarios."""

    def test_standard_scenarios_count(self):
        """Test that we have exactly 6 standard scenarios."""
        assert len(STANDARD_SCENARIOS) == 6

    def test_quick_scenarios_count(self):
        """Test that quick scenarios is a subset."""
        assert len(QUICK_SCENARIOS) == 2

    def test_scenario_names(self):
        """Test that scenario names are correct."""
        expected_names = [
            "step_low_speed_500rpm_2A",
            "step_mid_speed_1500rpm_2A",
            "step_high_speed_2500rpm_2A",
            "multi_step_bidirectional_1500rpm",
            "four_quadrant_transition_1500rpm",
            "field_weakening_2500rpm",
        ]
        actual_names = [s.name for s in STANDARD_SCENARIOS]
        assert actual_names == expected_names

    def test_scenario_speeds(self):
        """Test that scenarios have correct speeds."""
        expected_speeds = [500.0, 1500.0, 2500.0, 1500.0, 1500.0, 2500.0]
        actual_speeds = [s.n_rpm for s in STANDARD_SCENARIOS]
        assert actual_speeds == expected_speeds

    def test_scenario_durations(self):
        """Test that scenarios have appropriate durations."""
        expected_steps = [3000, 3000, 3000, 10000, 9000, 6000]
        actual_steps = [s.max_steps for s in STANDARD_SCENARIOS]
        assert actual_steps == expected_steps

    def test_primary_reference_scenario_is_mid_speed(self):
        """Test that scenario 2 (index 1) is the primary reference."""
        primary = STANDARD_SCENARIOS[1]
        assert primary.name == "step_mid_speed_1500rpm_2A"
        assert primary.n_rpm == 1500.0

    def test_multi_step_scenario_has_correct_generator(self):
        """Test that multi-step scenario uses MultiStepReference."""
        multi_step = STANDARD_SCENARIOS[3]
        assert isinstance(multi_step.reference_generator, MultiStepReference)

    def test_field_weakening_scenario_has_negative_id(self):
        """Test that field-weakening scenario activates i_d."""
        fw_scenario = STANDARD_SCENARIOS[5]
        assert isinstance(fw_scenario.reference_generator, MultiStepReference)
        
        # Check that it includes negative i_d step
        fw_gen = fw_scenario.reference_generator
        fw_gen.reset()
        
        # At t=0.2s, should have i_d=-2A
        ref = fw_gen(step=0, time_s=0.2)
        assert ref["i_d_ref"] == -2.0

    def test_scenarios_can_create_tasks(self):
        """Test that all scenarios can create valid tasks."""
        for scenario in STANDARD_SCENARIOS:
            task = scenario.create_task()
            assert task is not None
            assert task.physics_engine is not None
            assert task.reference_generator is not None
            task.physics_engine.close()


class TestBenchmarkSuite:
    """Test the BenchmarkSuite with new scenarios."""

    def test_suite_default_scenarios(self):
        """Test that suite uses STANDARD_SCENARIOS by default."""
        suite = BenchmarkSuite()
        assert len(suite.scenarios) == 6

    def test_suite_custom_scenarios(self):
        """Test that suite can use custom scenarios."""
        suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS)
        assert len(suite.scenarios) == 2

    def test_suite_single_scenario(self):
        """Test running suite with just one scenario."""
        from embark.benchmark.agents import PIControllerAgent
        
        suite = BenchmarkSuite(scenarios=[STANDARD_SCENARIOS[1]], verbose=False)
        
        # Create a simple PI controller
        task = STANDARD_SCENARIOS[1].create_task()
        controller = PIControllerAgent.from_system_config(task.physics_engine.config)
        task.physics_engine.close()
        
        # Run suite
        summary = suite.run(controller=controller, name="TestPI")
        
        assert summary.controller_name == "TestPI"
        assert len(summary.scenario_results) == 1
        assert summary.scenario_results[0].scenario_name == "step_mid_speed_1500rpm_2A"

    @pytest.mark.slow
    def test_suite_quick_scenarios_pi_controller(self):
        """Test running quick scenarios with PI controller (integration test)."""
        from embark.benchmark.agents import PIControllerAgent
        
        suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=False)
        
        # Create PI controller
        task = QUICK_SCENARIOS[0].create_task()
        controller = PIControllerAgent.from_system_config(task.physics_engine.config)
        task.physics_engine.close()
        
        # Run suite
        summary = suite.run(controller=controller, name="PI-Quick")
        
        # Verify results
        assert len(summary.scenario_results) == 2
        assert summary.mean_mae_iq > 0
        assert summary.num_safety_violations == 0
        
        # Check that metrics are computed
        for result in summary.scenario_results:
            assert "mae_i_q" in result.metrics
            assert result.metrics["mae_i_q"] >= 0


class TestScenarioReferences:
    """Test that scenario references work correctly over time."""

    def test_low_speed_scenario_reference(self):
        """Test low speed scenario reference profile."""
        scenario = STANDARD_SCENARIOS[0]
        gen = scenario.reference_generator
        gen.reset()
        
        # Should step from 0 to 2A at t=0
        ref_before = gen(step=0, time_s=0.0)
        assert ref_before["i_q_ref"] == 2.0
        assert ref_before["i_d_ref"] == 0.0

    def test_multi_step_scenario_bidirectional(self):
        """Test multi-step scenario has both positive and negative steps."""
        scenario = STANDARD_SCENARIOS[3]
        gen = scenario.reference_generator
        gen.reset()
        
        # Check key time points
        ref_t0 = gen(step=0, time_s=0.0)
        assert ref_t0["i_q_ref"] == 0.0
        
        ref_t1 = gen(step=0, time_s=0.15)  # After first step
        assert ref_t1["i_q_ref"] == 2.0
        
        ref_t2 = gen(step=0, time_s=0.4)  # After second step
        assert ref_t2["i_q_ref"] == -2.0  # Should be negative
        
        ref_t3 = gen(step=0, time_s=0.7)  # After third step
        assert ref_t3["i_q_ref"] == 2.0  # Back to positive

    def test_four_quadrant_scenario_includes_zero_crossing(self):
        """Test four-quadrant scenario returns to zero."""
        scenario = STANDARD_SCENARIOS[4]
        gen = scenario.reference_generator
        gen.reset()
        
        # Check that it ends at zero
        ref_end = gen(step=0, time_s=0.8)
        assert ref_end["i_q_ref"] == 0.0

    def test_field_weakening_scenario_multivariable(self):
        """Test field-weakening has both i_d and i_q active."""
        scenario = STANDARD_SCENARIOS[5]
        gen = scenario.reference_generator
        gen.reset()
        
        # At t=0.4s, should have both i_d and i_q active
        ref = gen(step=0, time_s=0.4)
        assert ref["i_d_ref"] == -2.0  # Field weakening
        assert ref["i_q_ref"] == 2.0   # Torque production


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
