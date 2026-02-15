"""Tests for controller-aware metric factory and suite integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from embark.benchmark.harness.benchmark_suite import (
    BenchmarkSuite,
    BenchmarkSummary,
    STANDARD_SCENARIOS,
    ScenarioDefinition,
    ScenarioResult,
)
from embark.benchmark.metrics.accumulators import TrackingMAE
from embark.benchmark.metrics import neurobench_factory as nf
from embark.benchmark.metrics.neurobench_factory import create_metrics
from embark.benchmark.metrics.registry import MetricRegistry
from embark.benchmark.tasks.reference_generators import ConstantReference


def test_standard_scenarios_drop_low_speed():
    """Standard scenarios should no longer include the 200 RPM case."""
    names = [scenario.name for scenario in STANDARD_SCENARIOS]
    assert "step_low_speed" not in names
    assert len(STANDARD_SCENARIOS) == 6


def test_create_metrics_without_controller_returns_control_only():
    """Without a controller model, only control metrics are returned."""
    metrics = create_metrics()
    names = {metric.name for metric in metrics}
    assert "tracking_mae" in names
    assert "tracking_itae" in names
    assert all("nb_" not in name for name in names)


def test_create_metrics_with_non_module_model_returns_control_only():
    """Controller model must be torch.nn.Module to enable NeuroBench metrics."""

    class _ControllerWithModel:
        model = object()

    metrics = create_metrics(controller=_ControllerWithModel())
    names = {metric.name for metric in metrics}
    assert "tracking_mae" in names
    assert all("nb_" not in name for name in names)


def test_create_metrics_with_torch_model_adds_neurobench_metrics():
    """A torch-backed controller should receive NeuroBench adapter metrics."""

    class _ControllerWithTorchModel:
        model = torch.nn.Linear(4, 2)

    metrics = create_metrics(controller=_ControllerWithTorchModel())
    names = {metric.name for metric in metrics}
    assert "tracking_mae" in names
    assert any(name.startswith("nb_") for name in names)


@dataclass
class _ScenarioStub:
    """Simple scenario object compatible with BenchmarkSuite."""

    name: str
    description: str
    task: object

    def create_task(self):
        return self.task


def test_suite_passes_controller_to_metric_factory(dummy_dict_controller, dummy_task):
    """BenchmarkSuite should call metric_factory(controller)."""
    seen: dict[str, object] = {}

    def factory(controller):
        seen["controller"] = controller
        return [TrackingMAE(tracked_keys=["i_q"])]

    suite = BenchmarkSuite(
        scenarios=[
            _ScenarioStub(name="stub", description="factory call", task=dummy_task)
        ],
        metric_factory=factory,
        verbose=False,
    )
    suite.run(dummy_dict_controller, name="Dummy")
    assert seen["controller"] is dummy_dict_controller


def test_suite_supports_legacy_noarg_metric_factory(dummy_dict_controller, dummy_task):
    """BenchmarkSuite keeps compatibility with no-argument factories."""

    def legacy_factory():
        return [TrackingMAE(tracked_keys=["i_q"])]

    suite = BenchmarkSuite(
        scenarios=[
            _ScenarioStub(name="stub", description="legacy call", task=dummy_task)
        ],
        metric_factory=legacy_factory,
        verbose=False,
    )
    summary = suite.run(dummy_dict_controller, name="Dummy")
    assert len(summary.scenario_results) == 1


def test_supports_noarg_constructor_true_for_defaults_only():
    class _NoArg:
        def __init__(self, value: int = 1):
            self.value = value

    assert nf._supports_noarg_constructor(_NoArg) is True


def test_supports_noarg_constructor_false_for_required_args():
    class _RequiresArg:
        def __init__(self, required):  # noqa: ANN001
            self.required = required

    assert nf._supports_noarg_constructor(_RequiresArg) is False


def test_supports_noarg_constructor_signature_error_returns_true(monkeypatch):
    def _raise_signature_error(_metric_cls):  # noqa: ANN001
        raise TypeError("signature unavailable")

    monkeypatch.setattr(nf, "signature", _raise_signature_error)
    assert nf._supports_noarg_constructor(object) is True


def test_controller_has_model_cases():
    class _NoModelAttr:
        pass

    class _ModelIsNone:
        model = None

    @dataclass
    class _HasTorchModel:
        model: torch.nn.Module

    assert nf._controller_has_model(None) is False
    assert nf._controller_has_model(_NoModelAttr()) is False
    assert nf._controller_has_model(_ModelIsNone()) is False
    assert nf._controller_has_model(_HasTorchModel(model=torch.nn.Linear(1, 1))) is True


def test_create_neurobench_adapters_filters_required_constructors(monkeypatch):
    class _StaticNoArg:
        pass

    class _StaticNeedsArg:
        def __init__(self, x):  # noqa: ANN001
            self.x = x

    class _WorkloadNoArg:
        pass

    class _WorkloadNeedsArg:
        def __init__(self, x):  # noqa: ANN001
            self.x = x

    class _DummyStaticAdapter:
        def __init__(self, controller, metric_cls):
            self.name = f"static:{metric_cls.__name__}"

    class _DummyWorkloadAdapter:
        def __init__(self, controller, metric_cls):
            self.name = f"workload:{metric_cls.__name__}"

    # Patch where the function is imported from (inside _create_neurobench_adapters)
    monkeypatch.setattr(
        "embark.benchmark.contrib.neurobench.metric_adapters.discover_neurobench_metric_classes",
        lambda: (
            [_StaticNoArg, _StaticNeedsArg],
            [_WorkloadNoArg, _WorkloadNeedsArg],
        ),
    )
    monkeypatch.setattr(
        "embark.benchmark.contrib.neurobench.metric_adapters.NeuroBenchStaticMetricAdapter",
        _DummyStaticAdapter,
    )
    monkeypatch.setattr(
        "embark.benchmark.contrib.neurobench.metric_adapters.NeuroBenchWorkloadMetricAdapter",
        _DummyWorkloadAdapter,
    )

    adapters = nf._create_neurobench_adapters(controller=object())
    names = [adapter.name for adapter in adapters]

    assert "static:_StaticNoArg" in names
    assert "workload:_WorkloadNoArg" in names
    assert all("NeedsArg" not in name for name in names)


def test_metric_registry_register_and_resolve():
    registry = MetricRegistry()
    registry.register("tracking_mae", "mae_i_q")

    assert registry.resolve("tracking_mae") == "mae_i_q"
    assert registry.resolve("missing_metric") is None


def test_benchmark_summary_aggregate_properties_and_to_dict():
    summary = BenchmarkSummary(
        controller_name="Ctrl",
        scenario_results=[
            ScenarioResult(
                scenario_name="s1",
                description="d1",
                metrics={"mae_i_q": 1.0, "max_error_i_q": 3.0},
                safety_terminated=False,
            ),
            ScenarioResult(
                scenario_name="s2",
                description="d2",
                metrics={"mae_i_q": 3.0, "max_error_i_q": 5.0},
                safety_terminated=True,
                violation_reason="x",
            ),
        ],
    )

    assert summary.mean_mae_iq == 2.0
    assert summary.worst_max_error_iq == 5.0
    assert summary.num_safety_violations == 1

    payload = summary.to_dict()
    assert payload["aggregate"]["num_scenarios"] == 2
    assert payload["aggregate"]["mean_mae_iq"] == 2.0


def test_default_metric_factory_has_control_metrics(dummy_dict_controller):
    suite = BenchmarkSuite(verbose=False)
    metrics = suite.metric_factory(dummy_dict_controller)
    names = {m.name for m in metrics}
    assert "tracking_mae" in names


def test_suite_run_calls_configure_when_present(dummy_task):
    class _ConfigurableController:
        def __init__(self):
            self.configure_calls = 0

        def configure(self, config, task):  # noqa: ANN001
            self.configure_calls += 1

        def reset(self):
            pass

        def __call__(self, state, reference):  # noqa: ANN001
            return {"v_d": 0.0, "v_q": 0.0}

        def get_state(self):
            return {}

        def set_state(self, state):  # noqa: ANN001
            pass

    class _ScenarioLocal:
        name = "stub"
        description = "desc"

        def create_task(self):
            return dummy_task

    controller = _ConfigurableController()

    def _metrics(_controller):
        return [TrackingMAE(tracked_keys=["i_q"])]

    suite = BenchmarkSuite(
        scenarios=[_ScenarioLocal()],
        metric_factory=_metrics,
        verbose=False,
    )
    summary = suite.run(controller, name="C")

    assert controller.configure_calls == 1
    assert len(summary.scenario_results) == 1


def test_print_summary_and_save_results(tmp_path: Path, capsys):
    summary = BenchmarkSummary(
        controller_name="Ctrl",
        scenario_results=[
            ScenarioResult(
                scenario_name="s1",
                description="d1",
                metrics={"mae_i_q": 0.1, "max_error_i_q": 0.2, "settling_time": 0.5},
                safety_terminated=False,
            ),
            ScenarioResult(
                scenario_name="s2",
                description="d2",
                metrics={
                    "mae_i_q": 0.2,
                    "max_error_i_q": 0.3,
                    "settling_time": float("inf"),
                },
                safety_terminated=True,
            ),
        ],
    )

    BenchmarkSuite.print_summary(summary)
    output = capsys.readouterr().out
    assert "Benchmark Summary: Ctrl" in output
    assert "AGGREGATE" in output
    assert "N/A" in output

    out_path = tmp_path / "results" / "summary.json"
    BenchmarkSuite.save_results(summary, out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["controller_name"] == "Ctrl"


def test_scenario_definition_create_task_uses_physics_engine(monkeypatch):
    class _FakePhysics:
        def __init__(self, n_rpm):
            self.config = object()
            self.n_rpm = n_rpm

        state_keys = {"i_d", "i_q", "time"}
        action_keys = {"v_d", "v_q"}

        def reset(self, seed=None):  # noqa: ANN001
            return {"i_d": 0.0, "i_q": 0.0, "time": 0.0}

        def step(self, action):  # noqa: ANN001
            return {"i_d": 0.0, "i_q": 0.0, "time": 0.0}, {}

    monkeypatch.setattr("embark.benchmark.physics.PMSMPhysicsEngine", _FakePhysics)

    scenario = ScenarioDefinition(
        name="custom",
        description="d",
        n_rpm=1234.0,
        reference_generator=ConstantReference(i_d_ref=0.0, i_q_ref=0.0),
        max_steps=5,
    )
    task = scenario.create_task()
    assert task.physics_engine.n_rpm == 1234.0
    assert task.max_steps == 5


def test_suite_run_verbose_prints_progress_and_status(dummy_task, capsys):
    class _Controller:
        def reset(self):
            pass

        def __call__(self, state, reference):  # noqa: ANN001
            return {"v_d": 0.0, "v_q": 0.0}

        def get_state(self):
            return {}

        def set_state(self, state):  # noqa: ANN001
            pass

    class _Scenario:
        name = "verbose_case"
        description = "demo"

        def create_task(self):
            dummy_task.terminated_by_safety = False
            dummy_task.last_violation_reason = None
            return dummy_task

    def _metrics(_controller):
        return [TrackingMAE(tracked_keys=["i_q"])]

    suite = BenchmarkSuite(
        scenarios=[_Scenario()], metric_factory=_metrics, verbose=True
    )
    summary = suite.run(_Controller(), name="VerboseCtrl")
    assert len(summary.scenario_results) == 1

    out = capsys.readouterr().out
    assert "[1/1] verbose_case: demo" in out
    assert "MAE=" in out and "MaxErr=" in out
