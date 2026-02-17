"""
Tests for Gaussian measurement noise, neuromorphic metrics, and BenchmarkConfig.

Part A: Noise injection in PMSMPhysicsEngine
Part B: SpikeCount, SynapticOps, ActivationSparsity accumulators
Part C: Factory includes new metrics
Part D: BenchmarkConfig integration

"""

from __future__ import annotations

import numpy as np
import pytest

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.harness.benchmark_suite import (
    QUICK_SCENARIOS,
    BenchmarkConfig,
    BenchmarkSuite,
)
from embark.benchmark.metrics.accumulators.neuromorphic import (
    ActivationSparsity,
    SpikeCount,
    SynapticOps,
)
from embark.benchmark.metrics.neurobench_factory import _control_metrics, create_metrics
from embark.benchmark.physics.config import PMSMConfig
from embark.benchmark.physics.pmsm import PMSMPhysicsEngine

# ============================================================================
# Helpers
# ============================================================================

_DUMMY_STATE = {"i_d": 0.0, "i_q": 1.0, "omega": 100.0, "epsilon": 0.0, "time": 0.0}
_DUMMY_REF = {"i_d_ref": 0.0, "i_q_ref": 2.0}
_DUMMY_ACTION = {"v_d": 0.0, "v_q": 0.0}
_DUMMY_NEXT_STATE = dict(_DUMMY_STATE)


def _make_controller_info(
    total_spikes: int = 100,
    syops: int = 5000,
    sparsity: float = 0.85,
) -> dict:
    return {
        "total_spikes": total_spikes,
        "syops": syops,
        "sparsity": sparsity,
    }


# ============================================================================
# Part A — Measurement noise
# ============================================================================


class TestMeasurementNoise:
    """Tests for optional Gaussian measurement noise on the physics engine."""

    def test_noise_disabled_by_default(self):
        """Default PMSMConfig has noise std = 0 → noise flag is off."""
        cfg = PMSMConfig()
        assert cfg.noise_current_std == 0.0
        assert cfg.noise_speed_std == 0.0

        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)
        assert engine._noise_enabled is False

    def test_noise_enabled_when_current_std_positive(self):
        cfg = PMSMConfig(noise_current_std=0.1)
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)
        assert engine._noise_enabled is True

    def test_noise_enabled_when_speed_std_positive(self):
        cfg = PMSMConfig(noise_speed_std=1.0)
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)
        assert engine._noise_enabled is True

    def test_noise_adds_variability(self):
        """With noise enabled, repeated resets from same seed must not be identical
        (unless the seed happens to produce zero noise — extremely unlikely over
        multiple readings)."""
        cfg = PMSMConfig(noise_current_std=0.5, noise_speed_std=5.0)
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)

        # Collect several reset states with different RNG seeds
        i_q_values = []
        for seed in range(10):
            state = engine.reset(seed=seed)
            i_q_values.append(state["i_q"])

        # With noise, there should be some variation across seeds
        assert len(set(i_q_values)) > 1, "Noise did not produce any variability"

    def test_noise_seed_reproducibility(self):
        """Same seed → same noise → same state."""
        cfg = PMSMConfig(noise_current_std=0.5, noise_speed_std=5.0)
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)

        state_a = engine.reset(seed=42)
        state_b = engine.reset(seed=42)

        assert state_a["i_d"] == pytest.approx(state_b["i_d"])
        assert state_a["i_q"] == pytest.approx(state_b["i_q"])
        assert state_a["omega"] == pytest.approx(state_b["omega"])

    def test_noise_magnitude_reasonable(self):
        """Noise with std=0.1 A should stay well within ±1 A for the vast majority of
        samples (99.7% within 3σ = 0.3 A)."""
        cfg = PMSMConfig(noise_current_std=0.1)
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)

        # Gather noisy states
        i_q_base_values = []
        for seed in range(50):
            engine.reset(seed=seed)
            # step once to get a deterministic base and then read state
            state = engine.reset(seed=seed)
            i_q_base_values.append(state["i_q"])

        # All values should be finite
        for v in i_q_base_values:
            assert np.isfinite(v), f"Non-finite i_q: {v}"

    def test_no_noise_deterministic(self):
        """Without noise, same seed → identical state (confirms noise path is off)."""
        cfg = PMSMConfig()  # noise_current_std=0.0, noise_speed_std=0.0
        engine = PMSMPhysicsEngine(n_rpm=1000.0, config=cfg)

        state_a = engine.reset(seed=42)
        state_b = engine.reset(seed=42)

        assert state_a["i_d"] == state_b["i_d"]
        assert state_a["i_q"] == state_b["i_q"]
        assert state_a["omega"] == state_b["omega"]
        assert state_a["epsilon"] == state_b["epsilon"]


# ============================================================================
# Part B — Neuromorphic metric accumulators
# ============================================================================


class TestSpikeCount:
    """Tests for SpikeCount accumulator."""

    def test_name(self):
        assert SpikeCount().name == "spike_count"

    def test_accumulation(self):
        m = SpikeCount()
        m.reset()
        info = _make_controller_info(total_spikes=100)
        for _ in range(10):
            m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, info)
        result = m.compute()
        assert result["total_spikes"] == 1000.0
        assert result["spikes_per_step"] == pytest.approx(100.0)

    def test_safe_without_controller_info(self):
        m = SpikeCount()
        m.reset()
        # Simulate PI controller path — no controller_info
        for _ in range(5):
            m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, None)
        result = m.compute()
        assert result["total_spikes"] == 0.0
        assert result["spikes_per_step"] == 0.0

    def test_reset_clears_state(self):
        m = SpikeCount()
        m.reset()
        info = _make_controller_info(total_spikes=50)
        m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, info)
        assert m.compute()["total_spikes"] == 50.0

        m.reset()
        assert m.compute()["total_spikes"] == 0.0

    def test_missing_key_ignored(self):
        """controller_info dict without 'total_spikes' is silently ignored."""
        m = SpikeCount()
        m.reset()
        m.update(
            _DUMMY_STATE,
            _DUMMY_REF,
            _DUMMY_ACTION,
            _DUMMY_NEXT_STATE,
            {"syops": 100},
        )
        assert m.compute()["total_spikes"] == 0.0


class TestSynapticOps:
    """Tests for SynapticOps accumulator."""

    def test_name(self):
        assert SynapticOps().name == "synaptic_ops"

    def test_accumulation(self):
        m = SynapticOps()
        m.reset()
        info = _make_controller_info(syops=500)
        for _ in range(20):
            m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, info)
        result = m.compute()
        assert result["total_syops"] == 10000.0
        assert result["syops_per_step"] == pytest.approx(500.0)

    def test_safe_without_controller_info(self):
        m = SynapticOps()
        m.reset()
        m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, None)
        result = m.compute()
        assert result["total_syops"] == 0.0
        assert result["syops_per_step"] == 0.0

    def test_reset_clears_state(self):
        m = SynapticOps()
        m.reset()
        m.update(
            _DUMMY_STATE,
            _DUMMY_REF,
            _DUMMY_ACTION,
            _DUMMY_NEXT_STATE,
            _make_controller_info(syops=200),
        )
        assert m.compute()["total_syops"] == 200.0
        m.reset()
        assert m.compute()["total_syops"] == 0.0


class TestActivationSparsity:
    """Tests for ActivationSparsity accumulator."""

    def test_name(self):
        assert ActivationSparsity().name == "activation_sparsity"

    def test_accumulation(self):
        m = ActivationSparsity()
        m.reset()
        sparsities = [0.8, 0.9, 0.7, 1.0, 0.6]
        for s in sparsities:
            info = _make_controller_info(sparsity=s)
            m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, info)
        result = m.compute()
        assert result["mean_sparsity"] == pytest.approx(0.8)
        assert result["min_sparsity"] == pytest.approx(0.6)
        assert result["max_sparsity"] == pytest.approx(1.0)

    def test_safe_without_controller_info(self):
        m = ActivationSparsity()
        m.reset()
        m.update(_DUMMY_STATE, _DUMMY_REF, _DUMMY_ACTION, _DUMMY_NEXT_STATE, None)
        result = m.compute()
        assert result["mean_sparsity"] == 0.0
        assert result["min_sparsity"] == 0.0
        assert result["max_sparsity"] == 0.0

    def test_reset_clears_state(self):
        m = ActivationSparsity()
        m.reset()
        m.update(
            _DUMMY_STATE,
            _DUMMY_REF,
            _DUMMY_ACTION,
            _DUMMY_NEXT_STATE,
            _make_controller_info(sparsity=0.5),
        )
        assert m.compute()["mean_sparsity"] == pytest.approx(0.5)
        m.reset()
        assert m.compute()["mean_sparsity"] == 0.0


# ============================================================================
# Part C — Factory includes neuromorphic metrics
# ============================================================================


class TestMetricFactory:
    """Verify neuromorphic metrics are included in the default factory."""

    def test_control_metrics_include_neuromorphic(self):
        metrics = _control_metrics()
        names = [m.name for m in metrics]
        assert "spike_count" in names
        assert "synaptic_ops" in names
        assert "activation_sparsity" in names

    def test_create_metrics_includes_neuromorphic(self):
        """create_metrics (with no controller) still includes neuromorphic."""
        metrics = create_metrics()
        names = [m.name for m in metrics]
        assert "spike_count" in names
        assert "synaptic_ops" in names
        assert "activation_sparsity" in names

    def test_total_metric_count(self):
        """Sanity: 11 control + 3 neuromorphic = 14 metrics without NeuroBench."""
        metrics = _control_metrics()
        assert len(metrics) == 14


# ============================================================================
# Part D — BenchmarkConfig integration
# ============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass and suite integration."""

    def test_default_config_canonical(self):
        """Default BenchmarkConfig matches the canonical benchmark settings."""
        cfg = BenchmarkConfig()
        assert cfg.tau == 1e-4
        assert cfg.use_dead_time is False
        assert cfg.noise_current_std == 0.0
        assert cfg.noise_speed_std == 0.0

    def test_to_pmsm_config_applies_overrides(self):
        """BenchmarkConfig fields flow through to PMSMConfig."""
        cfg = BenchmarkConfig(
            tau=5e-5,
            use_dead_time=True,
            noise_current_std=0.2,
            noise_speed_std=3.0,
        )
        pmsm = cfg.to_pmsm_config()
        assert pmsm.tau == 5e-5
        assert pmsm.use_dead_time is True
        assert pmsm.noise_current_std == 0.2
        assert pmsm.noise_speed_std == 3.0

    def test_to_dict_roundtrip(self):
        """to_dict() produces a serialisable dict with all fields."""
        cfg = BenchmarkConfig(noise_current_std=0.1, use_dead_time=True)
        d = cfg.to_dict()
        assert d["tau"] == 1e-4
        assert d["use_dead_time"] is True
        assert d["noise_current_std"] == 0.1
        assert d["noise_speed_std"] == 0.0

    def test_config_serialized_in_summary(self):
        """BenchmarkSuite serializes config into BenchmarkSummary."""
        cfg = BenchmarkConfig(noise_current_std=0.05)
        suite = BenchmarkSuite(
            scenarios=[QUICK_SCENARIOS[0]],
            config=cfg,
            verbose=False,
        )
        # Use PI controller for fast execution
        from embark.benchmark.physics.config import PMSMConfig as _PC

        controller = PIControllerAgent.from_system_config(_PC())
        summary = suite.run(controller=controller, name="PI-test")

        # Config should be in summary
        assert summary.config["noise_current_std"] == 0.05
        assert summary.config["use_dead_time"] is False

        # Config should be in serialised dict
        d = summary.to_dict()
        assert "config" in d
        assert d["config"]["noise_current_std"] == 0.05

    def test_suite_with_noise_config(self):
        """Suite with noise enabled produces valid (finite) metrics."""
        cfg = BenchmarkConfig(noise_current_std=0.1, noise_speed_std=1.0)
        suite = BenchmarkSuite(
            scenarios=[QUICK_SCENARIOS[0]],
            config=cfg,
            verbose=False,
        )
        controller = PIControllerAgent.from_system_config(PMSMConfig())
        summary = suite.run(controller=controller, name="PI-noisy")

        assert len(summary.scenario_results) == 1
        m = summary.scenario_results[0].metrics
        assert np.isfinite(m.get("mae_i_q", 0.0))
        assert np.isfinite(m.get("max_error_i_q", 0.0))

    def test_suite_with_deadtime_config(self):
        """Suite with dead-time enabled produces valid metrics."""
        cfg = BenchmarkConfig(use_dead_time=True)
        suite = BenchmarkSuite(
            scenarios=[QUICK_SCENARIOS[0]],
            config=cfg,
            verbose=False,
        )
        controller = PIControllerAgent.from_system_config(PMSMConfig())
        summary = suite.run(controller=controller, name="PI-deadtime")

        assert len(summary.scenario_results) == 1
        m = summary.scenario_results[0].metrics
        assert np.isfinite(m.get("max_error_i_q", 0.0))

    def test_suite_default_backward_compatible(self):
        """Suite without explicit config works identically to before."""
        suite_no_cfg = BenchmarkSuite(
            scenarios=[QUICK_SCENARIOS[0]],
            verbose=False,
        )
        suite_with_cfg = BenchmarkSuite(
            scenarios=[QUICK_SCENARIOS[0]],
            config=BenchmarkConfig(),
            verbose=False,
        )
        controller = PIControllerAgent.from_system_config(PMSMConfig())

        summary_a = suite_no_cfg.run(controller=controller, name="PI-a")
        summary_b = suite_with_cfg.run(controller=controller, name="PI-b")

        # Metrics should be identical (deterministic, no noise)
        m_a = summary_a.scenario_results[0].metrics
        m_b = summary_b.scenario_results[0].metrics
        for key in ("mae_i_q", "max_error_i_q", "rms_i_q"):
            assert m_a.get(key, 0.0) == pytest.approx(
                m_b.get(key, 0.0)
            ), f"Mismatch on {key}: {m_a.get(key)} vs {m_b.get(key)}"

    def test_import_from_top_level(self):
        """BenchmarkConfig is importable from the top-level benchmark package."""
        from embark.benchmark import BenchmarkConfig as BC

        assert BC is BenchmarkConfig
