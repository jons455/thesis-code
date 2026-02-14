"""
Tests for NeuroBench contrib integration layer.

Validates that the experimental NeuroBench model wrapper and result
exporter in ``embark.benchmark.contrib.neurobench`` work correctly
with the core ``ClosedLoopHarness``.

"""

from __future__ import annotations

import torch

from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.controllers.neural.snn_wrapper import SNNControllerWrapper
from embark.benchmark.contrib.neurobench import (
    ClosedLoopMetricExporter,
    NeuroBenchClosedLoopModel,
)
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.metrics.accumulators import TrackingMAE
from embark.benchmark.processors import IdentityActionProcessor, IdentityStateProcessor


# ---------------------------------------------------------------------------
# NeuroBenchClosedLoopModel tests
# ---------------------------------------------------------------------------


class TestNeuroBenchClosedLoopModel:
    """Tests for the contrib NeuroBench model wrapper."""

    def test_wraps_dict_controller(self, dummy_dict_controller, dummy_task):
        """Classical controller can be wrapped in NeuroBench model."""
        nb_model = NeuroBenchClosedLoopModel(dummy_dict_controller)

        assert nb_model.name == "ClosedLoopModel"
        assert nb_model.net is None  # No neural network

        state, ref = dummy_task.reset()
        action = nb_model(state, ref)
        assert "v_d" in action or "v_q" in action

    def test_wraps_tensor_adapter(self, dummy_tensor_controller, dummy_task):
        """TensorControllerAdapter can be wrapped in NeuroBench model."""
        state_proc = IdentityStateProcessor(
            state_keys=["i_q"], reference_keys=["i_q_ref"]
        )
        action_proc = IdentityActionProcessor(action_keys=["v_q"])

        adapter = TensorControllerAdapter(
            controller=dummy_tensor_controller,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        adapter.configure(dummy_task.physics_engine.config, dummy_task)

        nb_model = NeuroBenchClosedLoopModel(adapter, name="TestSNN")
        assert nb_model.name == "TestSNN"
        # .net should expose the dummy tensor controller (it has no .model attr)
        assert nb_model.net is dummy_tensor_controller

    def test_wraps_snn_wrapper_exposes_module(self, dummy_snn_module, dummy_task):
        """SNNControllerWrapper exposes the actual nn.Module via .net."""
        snn_wrapper = SNNControllerWrapper(dummy_snn_module)

        state_proc = IdentityStateProcessor(
            state_keys=["i_q"], reference_keys=["i_q_ref"]
        )
        action_proc = IdentityActionProcessor(action_keys=["v_q"])

        adapter = TensorControllerAdapter(
            controller=snn_wrapper,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        adapter.configure(dummy_task.physics_engine.config, dummy_task)

        nb_model = NeuroBenchClosedLoopModel(adapter)

        assert nb_model.net is dummy_snn_module
        assert isinstance(nb_model.net, torch.nn.Module)

    def test_runs_in_harness(self, dummy_dict_controller, dummy_task):
        """NeuroBench-wrapped controller works inside ClosedLoopHarness."""
        nb_model = NeuroBenchClosedLoopModel(dummy_dict_controller)
        metrics = [TrackingMAE(tracked_keys=["i_q"])]

        harness = ClosedLoopHarness(
            task=dummy_task, controller=nb_model, metrics=metrics
        )
        results = harness.run()

        assert "steps" in results
        assert results["steps"] == dummy_task.max_steps
        assert "mae_i_q" in results

    def test_reset_and_state(self, dummy_dict_controller):
        """Reset and state serialization pass through correctly."""
        nb_model = NeuroBenchClosedLoopModel(dummy_dict_controller)
        nb_model.reset()
        state = nb_model.get_state()
        assert isinstance(state, dict)
        nb_model.set_state(state)

    def test_repr(self, dummy_dict_controller):
        """__repr__ produces a readable string."""
        nb_model = NeuroBenchClosedLoopModel(dummy_dict_controller, name="MyModel")
        r = repr(nb_model)
        assert "MyModel" in r
        assert "DummyDictController" in r


# ---------------------------------------------------------------------------
# ClosedLoopMetricExporter tests
# ---------------------------------------------------------------------------


class TestClosedLoopMetricExporter:
    """Tests for exporting harness results to NeuroBench format."""

    def test_export_format(self):
        """Exporter produces NeuroBench-compatible dict."""
        exporter = ClosedLoopMetricExporter(
            benchmark_name="pmsm_iq_step", model_name="SNN-PI"
        )

        harness_results = {
            "steps": 100,
            "mae_i_q": 0.015,
            "settling_time_i_q": 0.02,
            "total_syops": 5000.0,
            "syops_per_step": 50.0,
            "nb_custom_metric_total": 42.0,
        }

        report = exporter.to_neurobench_format(harness_results)

        assert report["benchmark"] == "pmsm_iq_step"
        assert report["model"] == "SNN-PI"
        assert report["steps"] == 100
        assert "mae_i_q" in report["control_metrics"]
        assert "total_syops" in report["workload_metrics"]
        assert "nb_custom_metric_total" in report["workload_metrics"]

    def test_empty_results(self):
        """Exporter handles empty results gracefully."""
        exporter = ClosedLoopMetricExporter()
        report = exporter.to_neurobench_format({})
        assert report["steps"] == 0
        assert report["control_metrics"] == {}
        assert report["workload_metrics"] == {}
