"""Adapters bridging NeuroBench metrics into the MetricAccumulator protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
from typing import Any

from neurobench.metrics import static as _nb_static
from neurobench.metrics import workload as _nb_workload

from embark.benchmark.interfaces import (
    ActionDict,
    MetricAccumulator,
    ReferenceDict,
    StateDict,
)


def _snake_case(name: str) -> str:
    """Convert CamelCase metric names to snake_case keys."""
    interim = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", interim).lower()


def _to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float for metric outputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_result_key(metric_name: str, key: str) -> str:
    """Normalize nested NeuroBench keys to stable snake_case names."""
    metric_prefix = _snake_case(metric_name)
    normalized = _snake_case(key).strip("_")
    if normalized.startswith(metric_prefix):
        return normalized
    return f"{metric_prefix}_{normalized}"


def _is_task_specific_metric(name: str) -> bool:
    """Best-effort filter for dataset/task metrics not suitable here."""
    blocked = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "wer",
        "cer",
        "bleu",
        "perplexity",
        "reward",
        "coco",
        "mse",
        "r2",
        "smape",
        "classification",
        "detection",
        "segmentation",
    )
    lowered = name.lower()
    return any(token in lowered for token in blocked)


def _iter_metric_classes(module: Any, base_name: str) -> list[type]:
    """Return concrete metric classes exported by a NeuroBench module."""
    classes: list[type] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == base_name:
            continue
        if inspect.isabstract(obj):
            continue
        classes.append(obj)
    return classes


def discover_neurobench_metric_classes() -> tuple[list[type], list[type]]:
    """
    Discover generic static/workload NeuroBench metrics available at runtime.

    Returns:
        Tuple of `(static_metric_classes, workload_metric_classes)`.

    """
    static_classes = _iter_metric_classes(_nb_static, base_name="StaticMetric")
    workload_classes = _iter_metric_classes(_nb_workload, base_name="WorkloadMetric")

    static_filtered = [
        cls for cls in static_classes if not _is_task_specific_metric(cls.__name__)
    ]
    workload_filtered = [
        cls for cls in workload_classes if not _is_task_specific_metric(cls.__name__)
    ]

    # Keep deterministic ordering across Python versions/platforms.
    static_filtered.sort(key=lambda cls: cls.__name__)
    workload_filtered.sort(key=lambda cls: cls.__name__)
    return static_filtered, workload_filtered


@dataclass
class _BaseNeuroBenchAdapter(MetricAccumulator):
    """Shared helper behavior for NeuroBench adapters."""

    controller: Any
    metric_cls: type
    metric_name_override: str | None = None
    _metric: Any = field(default=None, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._metric = self._instantiate_metric()

    @property
    def metric_name(self) -> str:
        return (
            self.metric_name_override or f"nb_{_snake_case(self.metric_cls.__name__)}"
        )

    @property
    def name(self) -> str:
        return self.metric_name

    @property
    def model(self) -> Any | None:
        return getattr(self.controller, "model", None)

    def _instantiate_metric(self) -> Any:
        """Instantiate NeuroBench metric, trying common constructors."""
        try:
            return self.metric_cls()
        except TypeError:
            model = self.model
            if model is not None:
                return self.metric_cls(model)
            raise

    def reset(self) -> None:
        self._steps = 0
        self._sparsity_values = []
        self._use_spike_info = False
        self._syops_values = []
        self._use_spike_info_syops = False
        if hasattr(self._metric, "reset"):
            self._metric.reset()
        else:
            # Some NeuroBench metrics are stateful without a reset hook.
            self._metric = self._instantiate_metric()

    def _map_result(self, raw_result: Any) -> dict[str, float]:
        """
        Map arbitrary NeuroBench metric outputs to stable flat keys.

        Each metric always emits at least one `nb_*` key to preserve provenance.

        """
        metric_base = _snake_case(self.metric_cls.__name__)
        nb_base = f"nb_{metric_base}"

        if isinstance(raw_result, dict):
            mapped: dict[str, float] = {}
            for key, value in raw_result.items():
                norm = _normalise_result_key(self.metric_cls.__name__, str(key))
                mapped[norm] = _to_float(value)
                mapped[f"nb_{norm}"] = _to_float(value)
            return mapped

        value = _to_float(raw_result)
        return {nb_base: value}


@dataclass
class NeuroBenchStaticMetricAdapter(_BaseNeuroBenchAdapter):
    """Adapter for NeuroBench static metrics (model-only)."""

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,  # noqa: ARG002
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        self._steps += 1

    def compute(self) -> dict[str, float]:
        model = self.model
        if model is None:
            return {}

        raw_result: Any | None = None
        if callable(self._metric):
            try:
                raw_result = self._metric(model)
            except (TypeError, AttributeError):
                raw_result = None
        if raw_result is None and hasattr(self._metric, "compute"):
            compute_fn = self._metric.compute
            try:
                raw_result = compute_fn(model)
            except (TypeError, AttributeError):
                try:
                    raw_result = compute_fn()
                except (TypeError, AttributeError):
                    raw_result = None
        if raw_result is None:
            return {}
        mapped = self._map_result(raw_result)
        metric_name = self.metric_cls.__name__

        # Stable compatibility aliases used in docs/exports.
        if metric_name == "Footprint":
            value = mapped.get("nb_footprint", 0.0)
            mapped["footprint"] = value
        elif metric_name == "ConnectionSparsity":
            value = mapped.get("nb_connection_sparsity", 0.0)
            mapped["connection_sparsity"] = value
        return mapped


@dataclass
class NeuroBenchWorkloadMetricAdapter(_BaseNeuroBenchAdapter):
    """
    Adapter for NeuroBench workload metrics (step-wise).

    This adapter bridges NeuroBench's hook-based metrics with our SNN models that return
    spike information directly. For ActivationSparsity, we extract sparsity from
    controller_info["sparsity"] (computed from spike_info) rather than relying on
    PyTorch forward hooks.

    See DESIGN_DECISIONS.md D8 for rationale on spike_info vs hooks approach.

    """

    _reference_order: list[str] | None = field(default=None, init=False, repr=False)
    _sparsity_values: list[float] = field(default_factory=list, init=False, repr=False)
    _use_spike_info: bool = field(default=False, init=False, repr=False)
    _syops_values: list[float] = field(default_factory=list, init=False, repr=False)
    _use_spike_info_syops: bool = field(default=False, init=False, repr=False)

    def _build_reference_tensor(self, reference: ReferenceDict, obs: Any) -> Any | None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a project dependency
            return None

        if self._reference_order is None:
            state_processor = getattr(self.controller, "state_processor", None)
            keys = getattr(state_processor, "reference_keys", None)
            if keys:
                self._reference_order = list(keys)
            else:
                self._reference_order = sorted(reference.keys())
        values = [float(reference.get(k, 0.0)) for k in self._reference_order]
        ref = torch.tensor(values, dtype=obs.dtype, device=obs.device)
        return ref.unsqueeze(0)

    def update(
        self,
        state: StateDict,  # noqa: ARG002
        reference: ReferenceDict,
        action: ActionDict,  # noqa: ARG002
        next_state: StateDict,  # noqa: ARG002
        controller_info: dict[str, Any] | None = None,
    ) -> None:
        metric_name = self.metric_cls.__name__
        self._steps += 1

        # For ActivationSparsity: try to extract from spike_info first
        if metric_name == "ActivationSparsity" and controller_info is not None:
            sparsity = controller_info.get("sparsity")
            if sparsity is not None:
                self._sparsity_values.append(float(sparsity))
                self._use_spike_info = True
                return  # Skip NeuroBench hook-based update

        # For SynapticOperations: try to extract from spike_info first
        if metric_name == "SynapticOperations" and controller_info is not None:
            syops = controller_info.get("syops")
            if syops is not None:
                self._syops_values.append(float(syops))
                self._use_spike_info_syops = True
                return  # Skip NeuroBench hook-based update

        obs = getattr(self.controller, "last_observation", None)
        preds = getattr(self.controller, "last_action_tensor", None)
        model = self.model
        if model is None or obs is None or preds is None:
            return

        if obs.dim() == 1:
            obs_batch = obs.unsqueeze(0)
        else:
            obs_batch = obs
        if preds.dim() == 1:
            preds_batch = preds.unsqueeze(0)
        else:
            preds_batch = preds

        ref_batch = self._build_reference_tensor(reference, obs_batch)
        data = (obs_batch, ref_batch) if ref_batch is not None else (obs_batch, None)

        if callable(self._metric):
            for args in (
                (model, preds_batch, data),
                (model, preds_batch, (obs_batch, ref_batch)),
                (preds_batch, data),
                (preds_batch, (obs_batch, ref_batch)),
            ):
                try:
                    self._metric(*args)
                    return
                except (TypeError, AttributeError):
                    continue
        if hasattr(self._metric, "update"):
            for args in (
                (model, preds_batch, data),
                (preds_batch, data),
            ):
                try:
                    self._metric.update(*args)
                    return
                except (TypeError, AttributeError):
                    continue

    def compute(self) -> dict[str, float]:
        metric_name = self.metric_cls.__name__

        # For ActivationSparsity: use spike_info if available, otherwise NeuroBench hooks
        if metric_name == "ActivationSparsity" and self._use_spike_info:
            if self._sparsity_values:
                mean_sparsity = sum(self._sparsity_values) / len(self._sparsity_values)
                return {
                    "nb_activation_sparsity": mean_sparsity,
                    "activation_sparsity": mean_sparsity,
                }
            return {}

        # For SynapticOperations: use spike_info if available, otherwise NeuroBench hooks
        if metric_name == "SynapticOperations" and self._use_spike_info_syops:
            if self._syops_values:
                total_syops = sum(self._syops_values)
                return {
                    "total_syops": total_syops,
                    "syops_per_step": total_syops / max(self._steps, 1),
                    "nb_synaptic_operations_total": total_syops,
                    "effective_macs": 0.0,
                    "effective_acs": total_syops,  # Treat all as ACs (spike-based)
                    "dense": 0.0,
                }
            return {}

        raw_result: Any | None = None
        try:
            if hasattr(self._metric, "compute"):
                raw_result = self._metric.compute()
            elif callable(self._metric):
                raw_result = self._metric()
        except (TypeError, AttributeError):
            # Metric requires features not available in this model
            return {}
        mapped = self._map_result(raw_result)

        if metric_name == "ActivationSparsity":
            mapped["activation_sparsity"] = mapped.get("nb_activation_sparsity", 0.0)
        elif metric_name == "SynapticOperations":
            # Fall back to NeuroBench hook-based computation
            macs = mapped.get("synaptic_operations_effective_macs", 0.0)
            acs = mapped.get("synaptic_operations_effective_a_cs", 0.0)
            dense = mapped.get("synaptic_operations_dense", 0.0)
            total = macs + acs
            if total <= 0.0:
                total = dense
            mapped["effective_macs"] = macs
            mapped["effective_acs"] = acs
            mapped["dense"] = dense
            mapped["total_syops"] = total
            mapped["syops_per_step"] = total / max(self._steps, 1)
            mapped["nb_synaptic_operations_total"] = total
        return mapped
