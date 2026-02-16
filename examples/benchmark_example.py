"""
Embark Benchmark — Implementation-Oriented Usage Guide
=======================================================

This script demonstrates every major path through the benchmark framework.
Run it top-to-bottom, or jump to the section you need:

    1. PI baseline — single scenario, hand-picked metrics
    2. PI baseline — full suite (6 scenarios), default metric factory
    3. SNN controller — load a trained checkpoint, run quick suite
    4. Custom scenario — build your own reference & safety limits
    5. Low-level harness — manual control loop for debugging / logging
    6. Comparing controllers side-by-side
    7. BenchmarkConfig — opt-in noise, dead-time, and custom tau

Requirements:
    pip install embark            # core package
    pip install snntorch          # only for SNN examples (sections 3+)
    pip install gym-electric-motor  # physics backend (always needed)

Run:
    python examples/benchmark_example.py          # runs all sections
    python examples/benchmark_example.py --section 1   # run one section
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  PI controller baseline — single scenario, explicit metrics
# ---------------------------------------------------------------------------


def section_1_pi_single_scenario():
    """Minimal working example: one task, one controller, a few metrics."""

    from embark.benchmark import (
        ClosedLoopHarness,
        MaximumError,
        PIControllerAgent,
        PMSMCurrentControlTask,
        SettlingTime,
        TrackingITAE,
        TrackingMAE,
    )

    # --- Task ----------------------------------------------------------
    # PMSMCurrentControlTask wraps the GEM physics engine and a reference
    # generator.  from_config() is the easiest entry-point: it creates a
    # step-reference (0 → i_q_ref at t=0) at the requested speed.
    task = PMSMCurrentControlTask.from_config(
        n_rpm=1500,         # rotor speed [RPM]
        i_q_ref=2.0,        # q-axis current step [A]
        i_d_ref=0.0,        # d-axis current reference [A]
        max_steps=3000,      # episode length  (3000 × 0.1 ms = 0.3 s)
    )

    # --- Controller ----------------------------------------------------
    # PIControllerAgent is a classical FOC PI controller.  It implements
    # the unified Controller protocol directly — no adapter needed.
    # from_system_config() auto-tunes gains from the motor parameters.
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)

    # --- Metrics -------------------------------------------------------
    # Pick the metrics you care about.  Each is a MetricAccumulator that
    # receives (state, reference, action, next_state) every step, then
    # produces results via .compute() at the end.
    metrics = [
        TrackingMAE(tracked_keys=["i_q", "i_d"]),
        TrackingITAE(tracked_keys=["i_q", "i_d"], window_s=0.05),
        MaximumError(tracked_keys=["i_q", "i_d"]),
        SettlingTime(tracked_key="i_q", band_fraction=0.02, dwell_s=0.001),
    ]

    # --- Run -----------------------------------------------------------
    # ClosedLoopHarness owns the loop:
    #   state, ref = task.reset()
    #   controller.reset()
    #   while not done:
    #       action = controller(state, ref)
    #       state, ref, done = task.step(action)
    #       for m in metrics: m.update(...)
    harness = ClosedLoopHarness(task=task, controller=controller, metrics=metrics)
    results = harness.run()

    # --- Inspect results -----------------------------------------------
    # results is a flat dict: {"steps": int, "mae_i_q": float, ...}
    print("=== Section 1: PI single scenario ===")
    print(f"  Steps run    : {results['steps']}")
    print(f"  MAE i_q      : {results['mae_i_q']:.4f} A")
    print(f"  MAE i_d      : {results['mae_i_d']:.4f} A")
    print(f"  ITAE i_q     : {results['itae_i_q']:.6f} A*s^2")
    print(f"  Max error i_q: {results['max_error_i_q']:.4f} A")
    st = results.get("settling_time_i_q", float("inf"))
    print(f"  Settling time: {st:.4f} s" if st < float("inf") else "  Settling time: N/A")
    print()


# ---------------------------------------------------------------------------
# 2.  PI baseline — full 6-scenario benchmark suite
# ---------------------------------------------------------------------------


def section_2_pi_full_suite():
    """Run PI controller through all standard scenarios with default metrics."""

    from embark.benchmark import BenchmarkSuite, PIControllerAgent, STANDARD_SCENARIOS

    # BenchmarkSuite manages the scenario loop.  By default it uses
    # STANDARD_SCENARIOS (6 scenarios) and the default metric factory
    # which includes MAE, ITAE, MaxError, SettlingTime, Overshoot,
    # SteadyStateRMS, and InferenceLatency.
    suite = BenchmarkSuite(
        scenarios=STANDARD_SCENARIOS,  # or pass None for the same default
        verbose=True,                  # prints progress per scenario
    )

    # The PI controller doesn't need an adapter — it already speaks
    # Controller protocol: __call__(state, reference) -> ActionDict.
    # But it *does* need a config for auto-tuning, so BenchmarkSuite
    # will call controller.configure() if available.  PIControllerAgent
    # doesn't have configure(), so we use from_system_config() upfront.
    #
    # For PI, the suite internally creates a task per scenario and
    # re-instantiates the controller from scratch isn't needed — the
    # same controller works across speeds because the harness calls
    # controller.reset() per scenario.
    #
    # We use a generic config here; the suite will handle per-scenario
    # task creation.
    from embark.benchmark import PMSMConfig
    config = PMSMConfig()
    controller = PIControllerAgent.from_system_config(config)

    # Run all 6 scenarios
    summary = suite.run(controller=controller, name="PI-baseline")

    # Print formatted comparison table
    print(suite.format_summary(summary))

    # Programmatic access to per-scenario results:
    print(f"\nSafety violations: {summary.num_safety_violations}")
    print(f"Worst max error i_q: {summary.worst_max_error_iq:.4f} A")

    # Save to JSON (optional)
    output = Path("results/pi_baseline.json")
    output.parent.mkdir(exist_ok=True)
    suite.save_results(summary, output)
    print(f"Results saved to {output}")
    print()


# ---------------------------------------------------------------------------
# 3.  SNN controller — load trained checkpoint, run quick suite
# ---------------------------------------------------------------------------


def section_3_snn_quick_benchmark():
    """Load a trained SNN from a checkpoint and benchmark it."""

    try:
        import snntorch  # noqa: F401
        import torch
    except ImportError:
        print("=== Section 3: skipped (snntorch not installed) ===\n")
        return

    from embark.benchmark import (
        BenchmarkSuite,
        QUICK_SCENARIOS,
        TensorControllerAdapter,
    )
    from embark.benchmark.controllers.neural import SNNControllerWrapper
    from embark.benchmark.processors import (
        RateSNNActionProcessor,
        RateSNNStateProcessor,
    )

    # --- Model path ----------------------------------------------------
    # Point this to your trained .pt checkpoint.
    model_path = Path("tests/model/best_model.pt")
    if not model_path.exists():
        print(f"=== Section 3: skipped (no model at {model_path}) ===\n")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load checkpoint -----------------------------------------------
    # The checkpoint dict should contain at minimum:
    #   "state_dict"   : model weights
    #   "input_size"   : int  (e.g. 5, 12, or 13)
    #   "output_size"  : int  (typically 2 for v_d, v_q)
    #   "hidden_sizes" : list[int]
    #   "betas"        : list[float]  (LIF decay constants)
    #   "rate_steps"   : int  (temporal encoding steps)
    # Optional keys:
    #   "incremental_output" : bool
    #   "error_gain", "n_max", "delta_u_max", "version", ...
    checkpoint = torch.load(model_path, map_location=device)

    input_size = checkpoint.get("input_size", 12)
    incremental = checkpoint.get("incremental_output", False)
    print(f"=== Section 3: SNN quick benchmark ===")
    print(f"  Model input_size : {input_size}")
    print(f"  Output mode      : {'incremental' if incremental else 'absolute'}")

    # --- Rebuild model architecture ------------------------------------
    # You must reconstruct the same nn.Module that was used for training.
    # Here we import the architecture class from the test file as an example.
    # In production, import from your training code.
    sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
    from test_v10_end_to_end import FeedForwardRateSNNv10, load_v10_model

    model = load_v10_model(model_path, device=device)

    # --- State processor -----------------------------------------------
    # The processor must output exactly `input_size` features.
    # Choose flags to match what the model was trained on.
    if input_size == 5:
        # v5: currents(2) + errors(2) + speed(1)
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
        )
    elif input_size == 12:
        # v9/v10: currents(2) + errors(2) + speed(1) + derivatives(3) + EMAs(4)
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
    elif input_size == 13:
        # v12: currents(2) + refs(2) + errors(2) + speed(1) + prev_action(2) + EMAs(4)
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_references=True,
            include_errors=True,
            include_speed=True,
            include_prev_action=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
    else:
        raise ValueError(f"No processor template for input_size={input_size}")

    assert state_processor.output_dim == input_size, (
        f"Processor dim {state_processor.output_dim} != model input {input_size}"
    )

    # --- Action processor ----------------------------------------------
    action_processor = RateSNNActionProcessor(
        incremental=incremental,
        delta_max=checkpoint.get("delta_u_max", 0.2),
    )

    # --- Wrap everything -----------------------------------------------
    # SNNControllerWrapper adapts any nn.Module to the TensorController
    # protocol.  TensorControllerAdapter then bridges
    # TensorController + processors → unified Controller interface.
    wrapped = SNNControllerWrapper(model=model)

    controller = TensorControllerAdapter(
        controller=wrapped,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    # NOTE: do NOT call controller.configure() here — the BenchmarkSuite
    # calls it automatically for each scenario with the correct physics
    # config and task reference.

    # --- Run quick suite (2 scenarios) ---------------------------------
    suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=True)
    summary = suite.run(controller=controller, name="SNN-v12")

    print(suite.format_summary(summary))

    # Access the underlying model for diagnostics:
    print(f"\n  Underlying model type: {type(controller.model).__name__}")
    print()


# ---------------------------------------------------------------------------
# 4.  Custom scenario — build your own reference & safety limits
# ---------------------------------------------------------------------------


def section_4_custom_scenario():
    """Define a custom benchmark scenario with non-standard references."""

    from embark.benchmark import (
        BenchmarkSuite,
        PIControllerAgent,
        PMSMConfig,
        SafetyLimits,
        ScenarioDefinition,
    )
    from embark.benchmark.tasks.reference_generators import (
        MultiStepReference,
        SinusoidalReference,
    )

    config = PMSMConfig()

    # --- Custom reference generators -----------------------------------

    # Multi-step: ramp through several operating points
    ramp_ref = MultiStepReference(
        steps=[
            (0.00, 0.0, 0.0),    # t=0.00 s → i_d=0, i_q=0
            (0.05, 0.0, 1.0),    # t=0.05 s → i_q steps to 1 A
            (0.15, 0.0, 3.0),    # t=0.15 s → i_q steps to 3 A
            (0.30, 0.0, -1.0),   # t=0.30 s → i_q reverses to -1 A
        ]
    )

    # Sinusoidal: for tracking bandwidth characterisation
    sine_ref = SinusoidalReference(
        i_d_ref=0.0,
        i_q_amp=2.0,           # 2 A amplitude
        i_q_offset=0.0,
        frequency_hz=50.0,     # 50 Hz sinusoid
    )

    # --- Custom safety limits ------------------------------------------
    tight_limits = SafetyLimits(
        max_current_a=8.0,    # tighter than default 20 A
        max_voltage_v=24.0,   # half the default bus voltage
    )

    # --- Build scenario definitions ------------------------------------
    custom_scenarios = [
        ScenarioDefinition(
            name="ramp_1500rpm",
            description="Multi-step ramp at 1500 RPM",
            n_rpm=1500,
            reference_generator=ramp_ref,
            max_steps=5000,
            safety_limits=tight_limits,
        ),
        ScenarioDefinition(
            name="sine_tracking_1500rpm_50Hz",
            description="50 Hz sinusoidal tracking at 1500 RPM",
            n_rpm=1500,
            reference_generator=sine_ref,
            max_steps=10000,  # 1 s at 10 kHz
            safety_limits=None,  # use default limits
        ),
    ]

    # --- Run -----------------------------------------------------------
    controller = PIControllerAgent.from_system_config(config)
    suite = BenchmarkSuite(scenarios=custom_scenarios, verbose=True)
    summary = suite.run(controller=controller, name="PI-custom")

    print("=== Section 4: Custom scenarios ===")
    print(suite.format_summary(summary))
    print()


# ---------------------------------------------------------------------------
# 5.  Low-level harness — manual loop for debugging / custom logging
# ---------------------------------------------------------------------------


def section_5_manual_loop():
    """Use the harness components directly for full control over the loop."""

    from embark.benchmark import (
        PIControllerAgent,
        PMSMCurrentControlTask,
        TrackingMAE,
    )

    task = PMSMCurrentControlTask.from_config(
        n_rpm=1000,
        i_q_ref=2.0,
        max_steps=500,
    )
    controller = PIControllerAgent.from_system_config(task.physics_engine.config)
    metric = TrackingMAE(tracked_keys=["i_q"])

    # -- Manual control loop -------------------------------------------
    state, reference = task.reset()
    controller.reset()
    metric.reset()

    trajectory = []  # collect whatever you need

    done = False
    step = 0
    while not done and step < 500:
        action = controller(state, reference)

        next_state, next_reference, done = task.step(action)

        # Feed the metric
        metric.update(state, reference, action, next_state)

        # Custom logging — you have access to everything
        trajectory.append({
            "step": step,
            "time": state["time"],
            "i_q": state["i_q"],
            "i_q_ref": reference["i_q_ref"],
            "v_d": action["v_d"],
            "v_q": action["v_q"],
        })

        state = next_state
        reference = next_reference
        step += 1

    results = metric.compute()

    print("=== Section 5: Manual loop ===")
    print(f"  Steps: {step}")
    print(f"  Final i_q: {trajectory[-1]['i_q']:.4f} A  (ref: {trajectory[-1]['i_q_ref']:.1f} A)")
    print(f"  MAE i_q: {results['mae_i_q']:.4f} A")
    first5 = [f"{t['v_q']:.2f}" for t in trajectory[:5]]
    print(f"  First 5 v_q commands: {first5}")
    print()


# ---------------------------------------------------------------------------
# 6.  Comparing controllers side-by-side
# ---------------------------------------------------------------------------


def section_6_comparison():
    """Run two controllers through the same suite and compare."""

    from embark.benchmark import (
        BenchmarkSuite,
        PIControllerAgent,
        PMSMConfig,
        QUICK_SCENARIOS,
    )

    config = PMSMConfig()
    suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=False)

    # --- Controller A: aggressively-tuned PI ---------------------------
    ctrl_a = PIControllerAgent.from_system_config(config, tuning="technical_optimum")

    # --- Controller B: conservatively-tuned PI -------------------------
    ctrl_b = PIControllerAgent(kp_d=0.5, ki_d=50.0, kp_q=0.5, ki_q=50.0)

    summary_a = suite.run(controller=ctrl_a, name="PI-aggressive")
    summary_b = suite.run(controller=ctrl_b, name="PI-conservative")

    print("=== Section 6: Controller comparison ===")
    print(f"\n{'Metric':<25} {'PI-aggressive':>15} {'PI-conservative':>15}")
    print("-" * 55)

    for sa, sb in zip(summary_a.scenario_results, summary_b.scenario_results):
        print(f"\n  Scenario: {sa.scenario_name}")
        for key in ["mae_i_q", "max_error_i_q", "settling_time_i_q"]:
            va = sa.metrics.get(key, float("nan"))
            vb = sb.metrics.get(key, float("nan"))
            va_s = f"{va:.4f}" if va < float("inf") else "inf"
            vb_s = f"{vb:.4f}" if vb < float("inf") else "inf"
            print(f"    {key:<23} {va_s:>15} {vb_s:>15}")

    print(f"\n  Worst max error i_q:  {summary_a.worst_max_error_iq:>15.4f} {summary_b.worst_max_error_iq:>15.4f}")
    print(f"  Safety violations:    {summary_a.num_safety_violations:>15} {summary_b.num_safety_violations:>15}")
    print()


# ---------------------------------------------------------------------------
# 7.  BenchmarkConfig — opt-in noise, dead-time, custom tau
# ---------------------------------------------------------------------------


def section_7_benchmark_config():
    """Show how to enable measurement noise, dead-time, or change tau."""

    from embark.benchmark import (
        BenchmarkConfig,
        BenchmarkSuite,
        PIControllerAgent,
        PMSMConfig,
        QUICK_SCENARIOS,
    )

    config = PMSMConfig()
    controller = PIControllerAgent.from_system_config(config)

    # --- Default (canonical) benchmark ---------------------------------
    # BenchmarkSuite() without config= uses BenchmarkConfig() internally,
    # which is: tau=1e-4, no noise, no dead-time.  This is identical to
    # the behaviour before BenchmarkConfig was added.
    suite_default = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=False)
    summary_default = suite_default.run(controller=controller, name="PI-clean")

    # --- With Gaussian measurement noise -------------------------------
    # noise_current_std adds Gaussian noise to i_d and i_q (in Amperes)
    # noise_speed_std  adds Gaussian noise to omega     (in rad/s)
    # Noise is additive, applied after GEM denormalisation, and
    # seeded per-engine for reproducibility.
    noisy_config = BenchmarkConfig(
        noise_current_std=0.1,   # σ = 0.1 A on both i_d and i_q
        noise_speed_std=1.0,     # σ = 1 rad/s on omega
    )
    suite_noisy = BenchmarkSuite(
        scenarios=QUICK_SCENARIOS,
        config=noisy_config,
        verbose=False,
    )
    summary_noisy = suite_noisy.run(controller=controller, name="PI-noisy")

    # --- With inverter dead-time ---------------------------------------
    # Dead-time adds a realistic voltage distortion from the power stage.
    dt_config = BenchmarkConfig(use_dead_time=True)
    suite_dt = BenchmarkSuite(
        scenarios=QUICK_SCENARIOS,
        config=dt_config,
        verbose=False,
    )
    summary_dt = suite_dt.run(controller=controller, name="PI-deadtime")

    # --- Compare results -----------------------------------------------
    print("=== Section 7: BenchmarkConfig variants ===")
    print()
    print(f"{'Variant':<20} {'MAE_iq':>10} {'MaxErr_iq':>12} {'Settle':>10}")
    print("-" * 55)

    for summary in [summary_default, summary_noisy, summary_dt]:
        m = summary.scenario_results[0].metrics
        mae = m.get("mae_i_q", 0.0)
        max_err = m.get("max_error_i_q", 0.0)
        settle = m.get("settling_time_i_q", float("inf"))
        settle_s = f"{settle:.4f}" if settle < float("inf") else "N/A"
        print(f"{summary.controller_name:<20} {mae:>10.4f} {max_err:>12.4f} {settle_s:>10}")

    print()

    # The config is serialised into the summary for reproducibility:
    print("  Config stored in summary (noisy):")
    for k, v in summary_noisy.config.items():
        print(f"    {k}: {v}")

    # format_summary() shows active options in the header:
    print()
    print(suite_noisy.format_summary(summary_noisy))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

SECTIONS = {
    1: ("PI single scenario",        section_1_pi_single_scenario),
    2: ("PI full suite",             section_2_pi_full_suite),
    3: ("SNN quick benchmark",       section_3_snn_quick_benchmark),
    4: ("Custom scenarios",          section_4_custom_scenario),
    5: ("Manual control loop",       section_5_manual_loop),
    6: ("Controller comparison",     section_6_comparison),
    7: ("BenchmarkConfig variants",  section_7_benchmark_config),
}


def main():
    parser = argparse.ArgumentParser(
        description="Embark benchmark usage examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}: {v[0]}" for k, v in SECTIONS.items()),
    )
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run a specific section (1-7). Default: run all.",
    )
    args = parser.parse_args()

    if args.section is not None:
        if args.section not in SECTIONS:
            print(f"Unknown section {args.section}. Choose from: {list(SECTIONS.keys())}")
            return 1
        label, fn = SECTIONS[args.section]
        print(f"\n>>> Running section {args.section}: {label}\n")
        fn()
    else:
        for num, (label, fn) in SECTIONS.items():
            print(f"\n>>> Running section {num}: {label}\n")
            try:
                fn()
            except Exception as e:
                print(f"  [ERROR] {e}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
