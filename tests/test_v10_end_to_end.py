"""End-to-end test with trained SNN model from v10 notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add embark to path if needed
embark_root = Path(__file__).parent.parent
if str(embark_root) not in sys.path:
    sys.path.insert(0, str(embark_root))

from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.controllers.neural import SNNControllerWrapper
from embark.benchmark.harness import QUICK_SCENARIOS, STANDARD_SCENARIOS, BenchmarkSuite
from embark.benchmark.processors import RateSNNActionProcessor, RateSNNStateProcessor


# SNN architecture from v10 notebook
try:
    import snntorch as snn
    from snntorch import surrogate
    SNNTORCH_AVAILABLE = True
except ImportError:
    SNNTORCH_AVAILABLE = False
    print("Warning: snntorch not available, skipping SNN tests")


@pytest.mark.skipif(not SNNTORCH_AVAILABLE, reason="snntorch not installed")
class FeedForwardRateSNNv10(nn.Module):
    """
    SNN architecture from v10 notebook.
    Must match the architecture used during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        betas: list[float],
        rate_steps: int = 48,
        slope: float = 25.0,
        use_tanh: bool = False,
    ):
        super().__init__()
        if len(hidden_sizes) != len(betas):
            raise ValueError("hidden_sizes and betas must match")

        self.rate_steps = int(rate_steps)
        self.input_size = int(input_size)
        self.use_tanh = use_tanh

        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        prev = self.input_size * 2  # dual-population encoding

        spike_grad = surrogate.fast_sigmoid(slope=slope)
        for hs, beta in zip(hidden_sizes, betas):
            self.fcs.append(nn.Linear(prev, hs))
            self.lifs.append(snn.Leaky(beta=beta, spike_grad=spike_grad))
            prev = hs

        self.readout = nn.Linear(prev, output_size)

    def _encode_dual_population(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        x_clip = x.clamp(-1.0, 1.0)
        x_pos = torch.relu(x_clip)
        x_neg = torch.relu(-x_clip)
        if deterministic:
            return torch.cat([x_pos, x_neg], dim=-1)
        spk_pos = (torch.rand_like(x_pos) < x_pos).float()
        spk_neg = (torch.rand_like(x_neg) < x_neg).float()
        return torch.cat([spk_pos, spk_neg], dim=-1)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Forward pass returns only output (not spike rate for inference)."""
        batch_size = x.shape[0]
        device = x.device

        mems = [lif.init_leaky() for lif in self.lifs]
        spike_sum_last = torch.zeros(batch_size, self.fcs[-1].out_features, device=device)

        for _ in range(self.rate_steps):
            x_spk = self._encode_dual_population(x, deterministic=deterministic)
            h = x_spk
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                cur = fc(h)
                spk, mems[i] = lif(cur, mems[i])
                h = spk
            spike_sum_last = spike_sum_last + h

        hidden_rate = spike_sum_last / float(self.rate_steps)

        if self.use_tanh:
            y = torch.tanh(self.readout(hidden_rate))
        else:
            y = self.readout(hidden_rate)

        return y


def load_v10_model(model_path: Path, device: str = "cpu"):
    """
    Load trained v10 model from checkpoint.
    
    Args:
        model_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract architecture params
    input_size = checkpoint.get("input_size", 12)
    output_size = checkpoint.get("output_size", 2)
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 96, 64])
    betas = checkpoint.get("betas", [0.96, 0.90, 0.82])
    rate_steps = checkpoint.get("rate_steps", 48)
    slope = checkpoint.get("slope", 25.0)
    use_tanh = checkpoint.get("use_tanh", False)
    
    print(f"Loading v10 model:")
    print(f"  Architecture: {input_size} -> {hidden_sizes} -> {output_size}")
    print(f"  Rate steps: {rate_steps}")
    print(f"  Betas: {betas}")
    
    # Create model
    model = FeedForwardRateSNNv10(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        betas=betas,
        rate_steps=rate_steps,
        slope=slope,
        use_tanh=use_tanh,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Enable inference optimizations
    if device == "cuda":
        # Use TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("[OK] Model compiled with torch.compile for faster inference")
            except Exception:
                print("[INFO] torch.compile not available, using eager mode")
    
    print("[OK] Model loaded successfully")
    
    return model


def create_v10_controller(model_path: Path, device: str = "cpu"):
    """
    Create a complete controller from trained model for benchmarking.
    
    Automatically detects model configuration from checkpoint and creates
    appropriate processors.
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to run on
        
    Returns:
        TensorControllerAdapter ready for benchmarking
    """
    # Load checkpoint to get configuration
    checkpoint = torch.load(model_path, map_location=device)
    
    input_size = checkpoint.get("input_size", 12)
    incremental_output = checkpoint.get("incremental_output", False)
    error_gain = checkpoint.get("error_gain", 4.0)
    n_max = checkpoint.get("n_max", 3000.0)
    delta_u_max = checkpoint.get("delta_u_max", 0.2)
    version = checkpoint.get("version", "unknown")
    
    print(f"Model version: {version}")
    print(f"Input features: {input_size}")
    print(f"Output mode: {'incremental' if incremental_output else 'absolute'}")
    print(f"Error gain: {error_gain}, n_max: {n_max:.1f}")
    
    # Load model
    model = load_v10_model(model_path, device=device)
    
    # Wrap model
    wrapped_model = SNNControllerWrapper(model=model)
    
    # Create state processor based on input size
    if input_size == 12:
        # v10 configuration: 12 features
        state_processor = RateSNNStateProcessor(
            include_currents=True,       # i_d, i_q (2)
            include_errors=True,          # e_d, e_q (2)
            include_speed=True,           # n (1)
            include_derivatives=True,     # de_d, de_q, dn (3)
            include_ema_slow=True,        # EMA slow (2)
            include_ema_fast=True,        # EMA fast (2)
            include_references=False,     
            include_prev_action=False,    
            error_gain=error_gain,
            n_max=n_max,
            ema_alpha_slow=0.98,
            ema_alpha_fast=0.70,
        )
    elif input_size == 13:
        # v12 configuration: 13 features from train_snn_v12.py
        # Features: i_d, i_q, i_d_ref, i_q_ref, e_d, e_q, n, u_d_prev, u_q_prev,
        #           e_d_ema_slow, e_q_ema_slow, e_d_ema_fast, e_q_ema_fast
        state_processor = RateSNNStateProcessor(
            include_currents=True,       # i_d, i_q (2)
            include_errors=True,          # e_d, e_q (2)
            include_speed=True,           # n (1)
            include_references=True,      # i_d_ref, i_q_ref (2)
            include_prev_action=True,     # u_d_prev, u_q_prev (2)
            include_derivatives=False,    # NOT included in v12
            include_ema_slow=True,        # EMA slow (2)
            include_ema_fast=True,        # EMA fast (2)
            error_gain=error_gain,
            n_max=n_max,
            ema_alpha_slow=0.98,
            ema_alpha_fast=0.70,
        )
    else:
        raise ValueError(f"Unsupported input size: {input_size}")
    
    print(f"State processor: {state_processor.output_dim} features")
    assert state_processor.output_dim == input_size, f"Mismatch: processor={state_processor.output_dim}, model={input_size}"
    
    # Create action processor
    action_processor = RateSNNActionProcessor(
        incremental=incremental_output,
        delta_max=delta_u_max if incremental_output else 1.0,
    )
    
    # Create controller adapter
    controller = TensorControllerAdapter(
        controller=wrapped_model,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    
    print("[OK] Controller created successfully")
    
    return controller


@pytest.mark.skipif(not SNNTORCH_AVAILABLE, reason="snntorch not installed")
class TestV10ModelEndToEnd:
    """End-to-end tests with trained v10 model."""

    @pytest.fixture
    def model_path(self):
        """Path to trained model."""
        return Path(__file__).parent / "model" / "best_model.pt"

    @pytest.fixture
    def device(self):
        """Device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_model_exists(self, model_path):
        """Test that model file exists."""
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_model_loads(self, model_path, device):
        """Test that model can be loaded."""
        model = load_v10_model(model_path, device=device)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_forward_pass(self, model_path, device):
        """Test that model can perform forward pass."""
        model = load_v10_model(model_path, device=device)

        # Read input_size from checkpoint so the test works for any model version
        checkpoint = torch.load(model_path, map_location=device)
        input_size = checkpoint.get("input_size", 12)
        output_size = checkpoint.get("output_size", 2)

        # Create dummy input matching the model's expected input dimensions
        x = torch.randn(2, input_size, device=device)

        # Forward pass
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, output_size), f"Expected shape (2, {output_size}), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    def test_controller_creation(self, model_path, device):
        """Test that controller can be created from model."""
        controller = create_v10_controller(model_path, device=device)
        assert controller is not None

    def test_controller_single_step(self, model_path, device):
        """Test controller can perform single control step."""
        controller = create_v10_controller(model_path, device=device)

        # Configure controller with physics parameters (required before use)
        class _ConfigStub:
            i_max = 10.8
            u_max = 12.0
            tau = 1e-4

        class _TaskStub:
            pass

        controller.configure(_ConfigStub(), _TaskStub())

        # Create state dict matching PMSMPhysicsEngine output
        state = {
            "i_d": 0.0,
            "i_q": 1.0,
            "omega": 157.0,
            "epsilon": 0.0,
            "time": 0.0,
        }

        # Create reference dict matching ReferenceGenerator output
        reference = {
            "i_d_ref": 0.0,
            "i_q_ref": 2.0,
        }

        # Reset controller
        controller.reset()

        # Call controller using the unified interface: controller(state, reference)
        action = controller(state, reference)

        assert "v_d" in action, f"Expected 'v_d' in action, got keys: {list(action.keys())}"
        assert "v_q" in action, f"Expected 'v_q' in action, got keys: {list(action.keys())}"
        assert isinstance(action["v_d"], float), f"v_d should be float, got {type(action['v_d'])}"
        assert isinstance(action["v_q"], float), f"v_q should be float, got {type(action['v_q'])}"

    @pytest.mark.slow
    def test_single_scenario_benchmark(self, model_path, device):
        """Test running a single scenario with v10 model."""
        controller = create_v10_controller(model_path, device=device)
        
        # Use primary reference scenario (mid speed)
        suite = BenchmarkSuite(
            scenarios=[STANDARD_SCENARIOS[1]],
            verbose=True
        )
        
        # Run benchmark
        summary = suite.run(controller=controller, name="v10-test")
        
        # Verify results
        assert len(summary.scenario_results) == 1
        result = summary.scenario_results[0]
        
        assert result.scenario_name == "step_mid_speed_1500rpm_2A"
        assert "mae_i_q" in result.metrics
        assert "mae_i_d" in result.metrics
        
        # Check that controller produces reasonable results
        assert result.metrics["mae_i_q"] < 5.0, f"MAE too high: {result.metrics['mae_i_q']}"
        assert not result.safety_terminated, "Controller violated safety limits"
        
        print(f"\nSingle scenario results:")
        print(f"  MAE i_q: {result.metrics['mae_i_q']:.4f} A")
        print(f"  MAE i_d: {result.metrics['mae_i_d']:.4f} A")
        if "settling_time" in result.metrics:
            print(f"  Settling time: {result.metrics['settling_time']:.4f} s")

    @pytest.mark.slow
    def test_quick_scenarios_benchmark(self, model_path, device):
        """Test running quick scenarios with v10 model."""
        controller = create_v10_controller(model_path, device=device)
        
        # Run quick scenarios (2 scenarios)
        suite = BenchmarkSuite(scenarios=QUICK_SCENARIOS, verbose=True)
        summary = suite.run(controller=controller, name="v10-quick")
        
        # Verify results
        assert len(summary.scenario_results) == 2
        assert summary.num_safety_violations == 0, "Safety violations detected"
        
        print(f"\nQuick scenarios results:")
        for result in summary.scenario_results:
            print(f"  {result.scenario_name}:")
            print(f"    MAE i_q: {result.metrics['mae_i_q']:.4f} A")
            print(f"    Max error: {result.metrics.get('max_error_i_q', 0):.4f} A")
        
        print(f"\nAggregate metrics:")
        print(f"  Worst max error: {summary.worst_max_error_iq:.4f} A")

    @pytest.mark.slow
    @pytest.mark.full_benchmark
    def test_full_benchmark_suite(self, model_path, device):
        """Test running full 6-scenario benchmark with v10 model."""
        controller = create_v10_controller(model_path, device=device)
        
        # Run full suite
        suite = BenchmarkSuite(verbose=True)
        summary = suite.run(controller=controller, name="v10-full")
        
        # Verify results
        assert len(summary.scenario_results) == 6
        
        print(f"\n{'='*80}")
        print(f"Full Benchmark Results: v10")
        print(f"{'='*80}")
        
        for result in summary.scenario_results:
            status = "FAIL" if result.safety_terminated else "OK"
            print(f"\n{result.scenario_name}:")
            print(f"  Status: {status}")
            print(f"  MAE i_q: {result.metrics.get('mae_i_q', 0):.4f} A")
            print(f"  MAE i_d: {result.metrics.get('mae_i_d', 0):.4f} A")
            if "settling_time" in result.metrics:
                st = result.metrics['settling_time']
                st_str = f"{st:.4f}" if st < float('inf') else "N/A"
                print(f"  Settling time: {st_str} s")
            if "overshoot" in result.metrics:
                print(f"  Overshoot: {result.metrics['overshoot']:.2f} %")
        
        print(f"\n{'='*80}")
        print(f"Aggregate Metrics:")
        print(f"  Worst max error: {summary.worst_max_error_iq:.4f} A")
        print(f"  Safety violations: {summary.num_safety_violations}")
        print(f"{'='*80}")

        # Assert reasonable performance
        assert summary.num_safety_violations == 0, "Safety violations detected"
        assert summary.worst_max_error_iq < 20.0, f"Max error too high: {summary.worst_max_error_iq}"


def main():
    """Run end-to-end test standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test v10 model with benchmark suite")
    parser.add_argument("--model", type=str, default="tests/model/best_model.pt",
                        help="Path to trained model")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "quick", "full"],
                        help="Test mode: single scenario, quick (2), or full (6)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detect if not specified)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: reduce max_steps by 10x for quick testing")
    
    args = parser.parse_args()
    
    # Setup
    model_path = Path(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing v10 model: {model_path}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    if args.fast:
        print("Fast mode: ENABLED (10x fewer steps)")
    print()
    
    if not SNNTORCH_AVAILABLE:
        print("ERROR: snntorch not installed")
        print("Install with: pip install snntorch")
        return 1
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    try:
        # Create controller
        controller = create_v10_controller(model_path, device=device)
        
        # Select scenarios
        if args.mode == "single":
            scenarios = [STANDARD_SCENARIOS[1]]  # Primary reference
        elif args.mode == "quick":
            scenarios = QUICK_SCENARIOS
        else:  # full
            scenarios = STANDARD_SCENARIOS
        
        # Apply fast mode if requested
        if args.fast:
            from embark.benchmark.harness import ScenarioDefinition
            scenarios = [
                ScenarioDefinition(
                    name=s.name,
                    description=s.description,
                    n_rpm=s.n_rpm,
                    reference_generator=s.reference_generator,
                    max_steps=max(100, s.max_steps // 10),  # 10x reduction
                    safety_limits=s.safety_limits,
                )
                for s in scenarios
            ]
            print(f"Fast mode: Reduced steps to ~{sum(s.max_steps for s in scenarios)} total\n")
        
        # Run benchmark
        suite = BenchmarkSuite(scenarios=scenarios, verbose=True)
        summary = suite.run(controller=controller, name="v10")
        
        # Print summary
        suite.print_summary(summary)
        
        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        mode_suffix = f"{args.mode}_fast" if args.fast else args.mode
        output_file = output_dir / f"v10_benchmark_{mode_suffix}.json"
        suite.save_results(summary, output_file)
        print(f"\nResults saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
