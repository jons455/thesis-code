"""Unit tests for NeuroBench metric adapters."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from embark.benchmark.contrib.neurobench import metric_adapters as ma


def test_helper_functions_basic():
    assert ma._snake_case("ConnectionSparsity") == "connection_sparsity"
    assert ma._to_float("1.5") == pytest.approx(1.5)
    assert ma._to_float("bad", default=7.0) == pytest.approx(7.0)
    assert ma._normalise_result_key("SynapticOperations", "effective_macs").startswith(
        "synaptic_operations_"
    )
    assert ma._is_task_specific_metric("ClassificationAccuracy") is True
    assert ma._is_task_specific_metric("ConnectionSparsity") is False
    # Already-prefixed key should be returned as-is.
    assert (
        ma._normalise_result_key("ConnectionSparsity", "connection_sparsity_dense")
        == "connection_sparsity_dense"
    )


def test_iter_metric_classes_skips_base_and_abstract():
    class _Base:
        pass

    class _Concrete:
        pass

    class _Abstract(abc.ABC):
        @abc.abstractmethod
        def f(self):
            raise NotImplementedError

    mod = SimpleNamespace(_Base=_Base, _Concrete=_Concrete, _Abstract=_Abstract)
    classes = ma._iter_metric_classes(mod, base_name="_Base")
    names = {c.__name__ for c in classes}
    assert "_Base" not in names
    assert "_Abstract" not in names
    assert "_Concrete" in names


def test_discover_metric_classes_filters_and_sorts(monkeypatch):
    class ZMetric:
        pass

    class AccuracyMetric:
        pass

    class AMetric:
        pass

    monkeypatch.setattr(
        ma, "_nb_static", SimpleNamespace(ZMetric=ZMetric, AMetric=AMetric)
    )
    monkeypatch.setattr(
        ma,
        "_nb_workload",
        SimpleNamespace(AccuracyMetric=AccuracyMetric, AMetric=AMetric),
    )

    static, workload = ma.discover_neurobench_metric_classes()
    static_names = [c.__name__ for c in static]
    workload_names = [c.__name__ for c in workload]
    assert "AMetric" in static_names and "ZMetric" in static_names
    # Accuracy* should be filtered as task-specific.
    assert "AccuracyMetric" not in workload_names
    assert "AMetric" in workload_names


@dataclass
class _Controller:
    model: torch.nn.Module | None = None
    last_observation: torch.Tensor | None = None
    last_action_tensor: torch.Tensor | None = None
    state_processor: object | None = None


def _dummy_state_ref():
    return {"i_d": 0.0}, {"i_d_ref": 0.0}


def test_base_adapter_metric_name_and_reset_paths():
    class _MetricWithReset:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    adapter = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=_MetricWithReset,
    )
    assert adapter.name == "nb___metric_with_reset"
    adapter.reset()
    assert adapter._metric.reset_calls == 1

    class _MetricNoReset:
        pass

    adapter2 = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=_MetricNoReset,
        metric_name_override="nb_custom",
    )
    old = adapter2._metric
    assert adapter2.name == "nb_custom"
    adapter2.reset()
    assert adapter2._metric is not old


def test_base_adapter_instantiate_falls_back_to_model_ctor():
    class _NeedsModel:
        def __init__(self, model):
            self.model = model

    ctrl = _Controller(model=torch.nn.Linear(1, 1))
    adapter = ma.NeuroBenchStaticMetricAdapter(controller=ctrl, metric_cls=_NeedsModel)
    assert adapter._metric.model is ctrl.model


def test_base_adapter_instantiate_raises_when_model_ctor_required_and_missing():
    class _NeedsModel:
        def __init__(self, model):
            self.model = model

    adapter = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=lambda: None,  # placeholder, replaced below
    )
    adapter.controller = _Controller(model=None)
    adapter.metric_cls = _NeedsModel
    with pytest.raises(TypeError):
        adapter._instantiate_metric()


def test_map_result_dict_and_scalar():
    class Metric:
        pass

    adapter = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=Metric,
    )
    mapped = adapter._map_result({"effective_macs": 10, "dense": "2"})
    assert mapped["metric_effective_macs"] == pytest.approx(10.0)
    assert mapped["nb_metric_effective_macs"] == pytest.approx(10.0)
    assert mapped["metric_dense"] == pytest.approx(2.0)

    mapped2 = adapter._map_result(3)
    assert mapped2["nb_metric"] == pytest.approx(3.0)


def test_static_adapter_compute_branches_and_aliases():
    class Footprint:
        def __call__(self, model):
            return 12.0

    ctrl = _Controller(model=torch.nn.Linear(1, 1))
    adapter = ma.NeuroBenchStaticMetricAdapter(controller=ctrl, metric_cls=Footprint)
    res = adapter.compute()
    assert res["nb_footprint"] == pytest.approx(12.0)
    assert res["footprint"] == pytest.approx(12.0)

    class ConnectionSparsity:
        def __call__(self, model):
            raise TypeError("force compute fallback")

        def compute(self, model=None):
            return 0.5 if model is not None else 0.0

    adapter2 = ma.NeuroBenchStaticMetricAdapter(
        controller=ctrl, metric_cls=ConnectionSparsity
    )
    res2 = adapter2.compute()
    assert res2["nb_connection_sparsity"] == pytest.approx(0.5)
    assert res2["connection_sparsity"] == pytest.approx(0.5)

    # No model -> empty result
    adapter3 = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=None), metric_cls=ConnectionSparsity
    )
    assert adapter3.compute() == {}

    class _ComputeNoArgOnly:
        def __call__(self, model):
            raise TypeError("force compute fallback")

        def compute(self):
            return 1.0

    adapter4 = ma.NeuroBenchStaticMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=_ComputeNoArgOnly,
    )
    out4 = adapter4.compute()
    assert any(k.startswith("nb_") and v == pytest.approx(1.0) for k, v in out4.items())

    # Explicitly hit update() branch for static adapter.
    adapter4.update({}, {}, {}, {}, None)
    assert adapter4._steps == 1


def test_workload_build_reference_tensor_order_paths():
    class _Metric:
        def compute(self):
            return 0.0

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        state_processor=SimpleNamespace(reference_keys=["i_q_ref", "i_d_ref"]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(controller=ctrl, metric_cls=_Metric)
    obs = torch.zeros(1, 2)
    ref = {"i_d_ref": 1.0, "i_q_ref": 2.0}
    t = adapter._build_reference_tensor(ref, obs)
    assert t.shape == (1, 2)
    assert t[0, 0].item() == pytest.approx(2.0)
    assert t[0, 1].item() == pytest.approx(1.0)

    # Fallback to sorted keys when no state_processor reference keys.
    ctrl2 = _Controller(model=torch.nn.Linear(2, 2), state_processor=SimpleNamespace())
    adapter2 = ma.NeuroBenchWorkloadMetricAdapter(controller=ctrl2, metric_cls=_Metric)
    t2 = adapter2._build_reference_tensor(ref, obs)
    assert t2[0, 0].item() == pytest.approx(1.0)  # i_d_ref then i_q_ref
    assert t2[0, 1].item() == pytest.approx(2.0)


def test_workload_update_callable_and_update_fallback_paths():
    class _CallableMetric:
        def __init__(self):
            self.called = False

        def __call__(self, *args):
            # Accept only second call signature branch.
            if len(args) == 3 and isinstance(args[2], tuple):
                self.called = True
                return None
            raise TypeError("try next signature")

        def compute(self):
            return {"x": 1.0}

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.tensor([0.1, 0.2]),
        last_action_tensor=torch.tensor([0.3, 0.4]),
        state_processor=SimpleNamespace(reference_keys=["i_d_ref", "i_q_ref"]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=_CallableMetric
    )
    _, ref = _dummy_state_ref()
    ref = {"i_d_ref": 0.0, "i_q_ref": 1.0}
    adapter.update({}, ref, {}, {}, None)
    assert adapter._metric.called is True

    class _UpdateOnlyMetric:
        def __init__(self):
            self.updated = False

        def __call__(self, *args):
            raise TypeError("force update path")

        def update(self, *args):
            # First signature may fail, second should be accepted here.
            if len(args) == 2:
                self.updated = True
                return None
            raise TypeError("try next update signature")

        def compute(self):
            return {"y": 2.0}

    adapter2 = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=_UpdateOnlyMetric
    )
    adapter2.update({}, ref, {}, {}, None)
    assert adapter2._metric.updated is True

    # Early return when model/obs/preds unavailable.
    ctrl_missing = _Controller(
        model=None, last_observation=None, last_action_tensor=None
    )
    adapter3 = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl_missing,
        metric_cls=_CallableMetric,
    )
    adapter3.update({}, ref, {}, {}, None)

    # Hit obs/preds already-batched branches (no unsqueeze).
    ctrl_batched = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.randn(1, 2),
        last_action_tensor=torch.randn(1, 2),
        state_processor=SimpleNamespace(reference_keys=["i_d_ref", "i_q_ref"]),
    )
    adapter4 = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl_batched, metric_cls=_CallableMetric
    )
    adapter4.update({}, ref, {}, {}, None)


def test_workload_compute_aliases_and_synops_total_fallback():
    class ActivationSparsity:
        def compute(self):
            return 0.25

    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=ActivationSparsity,
    )
    out = adapter.compute()
    assert out["nb_activation_sparsity"] == pytest.approx(0.25)
    assert out["activation_sparsity"] == pytest.approx(0.25)

    class SynapticOperations:
        def compute(self):
            # No effective macs/acs -> fallback to dense
            return {"dense": 5.0, "effective_macs": 0.0, "effective_acs": 0.0}

    adapter2 = ma.NeuroBenchWorkloadMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=SynapticOperations,
    )
    adapter2._steps = 2
    out2 = adapter2.compute()
    assert out2["dense"] == pytest.approx(5.0)
    assert out2["total_syops"] == pytest.approx(5.0)
    assert out2["syops_per_step"] == pytest.approx(2.5)
    assert out2["nb_synaptic_operations_total"] == pytest.approx(5.0)

    class _CallableOnlyMetric:
        def __call__(self):
            return 7.0

    adapter3 = ma.NeuroBenchWorkloadMetricAdapter(
        controller=_Controller(model=torch.nn.Linear(1, 1)),
        metric_cls=_CallableOnlyMetric,
    )
    out3 = adapter3.compute()
    assert any(k.startswith("nb_") and v == pytest.approx(7.0) for k, v in out3.items())


def test_workload_adapter_extracts_sparsity_from_spike_info():
    """Test that ActivationSparsity adapter extracts sparsity from controller_info."""

    class ActivationSparsity:
        def compute(self):
            return 0.0  # Should be overridden by spike_info

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.tensor([0.1, 0.2]),
        last_action_tensor=torch.tensor([0.3, 0.4]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=ActivationSparsity
    )

    # Simulate spike_info via controller_info
    controller_info = {"sparsity": 0.85}

    # Update with spike_info
    adapter.update({}, {"i_d_ref": 0.0, "i_q_ref": 0.0}, {}, {}, controller_info)

    # Compute should use spike_info, not NeuroBench hooks
    result = adapter.compute()
    assert result["activation_sparsity"] == pytest.approx(0.85)
    assert result["nb_activation_sparsity"] == pytest.approx(0.85)
    assert adapter._use_spike_info is True


def test_workload_adapter_extracts_syops_from_spike_info():
    """Test that SynapticOperations adapter extracts SyOps from controller_info."""

    class SynapticOperations:
        def compute(self):
            return {"dense": 0.0, "effective_macs": 0.0, "effective_acs": 0.0}

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.tensor([0.1, 0.2]),
        last_action_tensor=torch.tensor([0.3, 0.4]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=SynapticOperations
    )

    # Simulate spike_info via controller_info
    controller_info = {"syops": 100.0}

    # Update multiple times to accumulate
    for _ in range(5):
        adapter.update({}, {"i_d_ref": 0.0, "i_q_ref": 0.0}, {}, {}, controller_info)

    # Compute should use spike_info
    result = adapter.compute()
    assert result["total_syops"] == pytest.approx(500.0)  # 100 * 5 steps
    assert result["syops_per_step"] == pytest.approx(100.0)
    assert result["effective_acs"] == pytest.approx(500.0)
    assert result["effective_macs"] == pytest.approx(0.0)
    assert adapter._use_spike_info_syops is True


def test_workload_adapter_falls_back_to_hooks_when_spike_info_unavailable():
    """Test that adapter falls back to NeuroBench hooks when spike_info is missing."""

    class ActivationSparsity:
        def compute(self):
            return 0.75

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.tensor([0.1, 0.2]),
        last_action_tensor=torch.tensor([0.3, 0.4]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=ActivationSparsity
    )

    # Update without controller_info (no spike_info)
    adapter.update({}, {"i_d_ref": 0.0, "i_q_ref": 0.0}, {}, {}, None)

    # Compute should use NeuroBench hooks
    result = adapter.compute()
    assert result["activation_sparsity"] == pytest.approx(0.75)
    assert adapter._use_spike_info is False


def test_workload_adapter_reset_clears_spike_info_state():
    """Test that reset() clears accumulated spike_info values."""

    class ActivationSparsity:
        def compute(self):
            return 0.0

    ctrl = _Controller(
        model=torch.nn.Linear(2, 2),
        last_observation=torch.tensor([0.1, 0.2]),
        last_action_tensor=torch.tensor([0.3, 0.4]),
    )
    adapter = ma.NeuroBenchWorkloadMetricAdapter(
        controller=ctrl, metric_cls=ActivationSparsity
    )

    # Update with spike_info
    controller_info = {"sparsity": 0.9}
    adapter.update({}, {"i_d_ref": 0.0, "i_q_ref": 0.0}, {}, {}, controller_info)
    assert len(adapter._sparsity_values) == 1
    assert adapter._use_spike_info is True

    # Reset should clear state
    adapter.reset()
    assert len(adapter._sparsity_values) == 0
    assert adapter._use_spike_info is False
    assert len(adapter._syops_values) == 0
    assert adapter._use_spike_info_syops is False
