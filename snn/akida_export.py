"""Akida/NIR export utilities for PMSM SNN controller.

This module provides tools for exporting the trained SNN controller to
neuromorphic hardware formats: NIR (Neuromorphic Intermediate Representation)
and Akida-compatible format via CNN2SNN.

Hardware targets include BrainChip Akida, Intel Loihi, and SpiNNaker.

Example:
    Export a trained model for hardware deployment::

        from snn.akida_export import export_to_nir, validate_akida_compatibility

        model = SimpleSNNController.load("snn/checkpoints/best_model.pt")

        # Check compatibility
        report = validate_akida_compatibility(model)
        print(report)

        # Export to NIR
        export_to_nir(model, "exports/pmsm_controller.nir")
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class AkidaCompatibilityReport:
    """Report on Akida hardware compatibility."""

    is_compatible: bool
    issues: list[str]
    warnings: list[str]
    recommendations: list[str]

    def __str__(self) -> str:
        lines = ["=" * 60, "AKIDA COMPATIBILITY REPORT", "=" * 60]
        lines.append(
            f"\nOverall: {'✅ COMPATIBLE' if self.is_compatible else '❌ NEEDS CHANGES'}\n"
        )

        if self.issues:
            lines.append("ISSUES (must fix):")
            for issue in self.issues:
                lines.append(f"  ❌ {issue}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️ {warning}")

        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  💡 {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_akida_compatibility(model: nn.Module) -> AkidaCompatibilityReport:
    """Check if the SNN model is compatible with Akida hardware.

    Akida constraints include: specific layer types (Dense, Conv2D, etc.),
    LIF neurons with specific parameters, INT8 quantized weights, and
    no arbitrary recurrence.

    Args:
        model: The SNN model to validate.

    Returns:
        Detailed compatibility report with issues, warnings, and recommendations.
    """
    issues = []
    warnings = []
    recommendations = []

    # Check for snnTorch Leaky neurons
    has_lif = False
    has_slow_leak = False
    has_no_reset = False

    for name, module in model.named_modules():
        # Check layer types
        if hasattr(module, "beta"):  # snnTorch neuron
            has_lif = True
            beta = module.beta
            if hasattr(beta, "item"):
                beta = beta.item()

            # Check for slow leak (may need special handling)
            if beta > 0.99:
                has_slow_leak = True
                warnings.append(
                    f"Layer '{name}' has slow leak (β={beta:.4f}). "
                    "May need membrane readout on host."
                )

            # Check reset mechanism
            if hasattr(module, "reset_mechanism") and module.reset_mechanism == "none":
                has_no_reset = True
                warnings.append(
                    f"Layer '{name}' has reset_mechanism='none'. "
                    "Akida uses reset-by-subtraction by default."
                )

        # Check for unsupported layers
        if isinstance(module, nn.RNN | nn.LSTM | nn.GRU):
            issues.append(f"Recurrent layer '{name}' not supported on Akida")

        if isinstance(module, nn.BatchNorm1d):
            warnings.append(
                f"BatchNorm '{name}' will be folded into weights during quantization"
            )

    if not has_lif:
        issues.append("No LIF neurons found - model may not be a valid SNN")

    # Check weight dtype
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            warnings.append(
                f"Parameter '{name}' is {param.dtype}, will be quantized to INT8"
            )

    # Check for quantization
    recommendations.append(
        "Quantize weights to INT8 before deployment (use quantization-aware training)"
    )

    if has_slow_leak:
        recommendations.append(
            "For slow-leak output: Consider adding a spike decoder or "
            "reading membrane on host for continuous control output"
        )

    if has_no_reset:
        recommendations.append(
            "Modify output layer to use reset-by-subtraction, or "
            "handle accumulation on host processor"
        )

    # Overall compatibility
    is_compatible = len(issues) == 0

    return AkidaCompatibilityReport(
        is_compatible=is_compatible,
        issues=issues,
        warnings=warnings,
        recommendations=recommendations,
    )


def export_to_nir(
    model: nn.Module,
    output_path: str,
    sample_input: torch.Tensor | None = None,
) -> bool:
    """Export SNN model to NIR (Neuromorphic Intermediate Representation) format.

    NIR is a universal format that can be converted to Akida, Loihi,
    SpiNNaker, and Speck hardware.

    Args:
        model: Trained SNN model.
        output_path: Path to save the .nir file.
        sample_input: Sample input for tracing. If None, uses default shape [1, 4].

    Returns:
        True if export succeeded.
    """
    try:
        import nir
        from snntorch import export as snn_export
    except ImportError as e:
        print("NIR export requires: pip install nir")
        print("snnTorch NIR support requires: pip install snntorch>=0.9.0")
        raise ImportError("Install NIR support: pip install nir snntorch>=0.9.0") from e

    if sample_input is None:
        # Default: [batch=1, features=4] for [i_d, i_q, e_d, e_q]
        sample_input = torch.randn(1, 4)

    # Ensure model is in eval mode
    model.eval()

    # Export to NIR
    print("Exporting model to NIR format...")
    nir_graph = snn_export.to_nir(model, sample_input)

    # Save NIR file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nir.write(str(output_path), nir_graph)
    print(f"✅ NIR graph saved to: {output_path}")
    print(f"   Nodes: {len(nir_graph.nodes)}")
    print(f"   Edges: {len(nir_graph.edges)}")

    return True


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    sample_input: torch.Tensor | None = None,
) -> bool:
    """Export SNN model to ONNX format (intermediate step for some hardware).

    Note: ONNX doesn't natively support spiking neurons, so this exports
    the computational graph. Useful for CNN2SNN workflows.

    Args:
        model: Trained SNN model.
        output_path: Path to save the .onnx file.
        sample_input: Sample input for tracing.

    Returns:
        True if export succeeded.
    """
    if sample_input is None:
        sample_input = torch.randn(1, 4)

    model.eval()

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        input_names=["state"],
        output_names=["voltage"],
        dynamic_axes={
            "state": {0: "batch"},
            "voltage": {0: "batch"},
        },
        opset_version=14,
    )

    print(f"✅ ONNX model saved to: {output_path}")
    return True


# =============================================================================
# Alternative Output Architectures for Akida
# =============================================================================


class RateCodingOutput(nn.Module):
    """
    Alternative output layer using rate coding.

    Instead of reading membrane potential, count spikes over a time window
    and decode to continuous values. More Akida-compatible.

    Output = (spike_count / time_steps) * scale - offset
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        time_steps: int = 10,
        scale: float = 2.0,
        offset: float = 1.0,
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.time_steps = time_steps
        self.scale = scale
        self.offset = offset

        # Import snnTorch
        import snntorch as snn
        from snntorch import surrogate

        # Standard LIF for spike output
        self.lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            reset_mechanism="subtract",  # Akida-compatible
        )

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass accumulating spikes over time_steps.

        Args:
            x: Input from previous layer [batch, features].
            mem: Previous membrane state.

        Returns:
            Tuple of (output, mem) where output is decoded continuous
            value [batch, output_size] and mem is updated membrane state.
        """
        batch_size = x.shape[0]
        device = x.device

        if mem is None:
            mem = torch.zeros(batch_size, self.fc.out_features, device=device)

        # Accumulate spikes
        spike_count = torch.zeros(batch_size, self.fc.out_features, device=device)

        for _ in range(self.time_steps):
            cur = self.fc(x)
            spk, mem = self.lif(cur, mem)
            spike_count += spk

        # Decode: rate → continuous value
        rate = spike_count / self.time_steps
        output = rate * self.scale - self.offset

        return output, mem


class PopulationCodingOutput(nn.Module):
    """Alternative output layer using population coding.

    Multiple neurons per output dimension, each tuned to a different value.
    The population activity is decoded to a continuous value.
    More robust and Akida-compatible than membrane readout.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        neurons_per_output: int = 10,
        value_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.output_size = output_size
        self.neurons_per_output = neurons_per_output
        self.value_range = value_range

        # Total output neurons
        total_neurons = output_size * neurons_per_output
        self.fc = nn.Linear(input_size, total_neurons)

        # Preferred values for each neuron (evenly spaced)
        values = torch.linspace(value_range[0], value_range[1], neurons_per_output)
        self.register_buffer("preferred_values", values)

        import snntorch as snn
        from snntorch import surrogate

        self.lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            reset_mechanism="subtract",
        )

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with population decoding.
        """
        batch_size = x.shape[0]
        device = x.device

        if mem is None:
            mem = torch.zeros(
                batch_size, self.output_size * self.neurons_per_output, device=device
            )

        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)

        # Reshape to [batch, output_size, neurons_per_output]
        spk_reshaped = spk.view(batch_size, self.output_size, self.neurons_per_output)

        # Population decoding: weighted average of preferred values
        # Normalize spike activity per output dimension
        spk_sum = spk_reshaped.sum(dim=-1, keepdim=True) + 1e-8
        weights = spk_reshaped / spk_sum

        # Decode
        output = (weights * self.preferred_values).sum(dim=-1)

        return output, mem


# =============================================================================
# Quantization Utilities
# =============================================================================


def quantize_model(
    model: nn.Module,
    calibration_data: torch.Tensor | None = None,
    bits: int = 8,
) -> nn.Module:
    """Quantize model weights to INT8 for Akida deployment.

    This is a simple post-training quantization. For best results,
    use quantization-aware training.

    Args:
        model: Model to quantize.
        calibration_data: Data for calibrating activation ranges.
        bits: Number of bits (default: 8 for Akida).

    Returns:
        Quantized model.
    """
    # This is a placeholder - full quantization would use:
    # - torch.quantization for PyTorch native
    # - cnn2snn for BrainChip's toolchain
    # - brevitas for QAT

    print(f"⚠️ Full INT{bits} quantization requires BrainChip's CNN2SNN toolkit")
    print("  Install: pip install cnn2snn akida")
    print("  Or use: torch.quantization for PyTorch native quantization")

    # For now, just return the model with a warning
    return model


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for Akida export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export SNN to Akida/NIR format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="snn/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/pmsm_controller.nir",
        help="Output path for NIR file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate compatibility, don't export",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["nir", "onnx"],
        default="nir",
        help="Export format",
    )

    args = parser.parse_args()

    # Load model
    from snn.models import SimpleSNNController

    print(f"Loading model from: {args.checkpoint}")
    model = SimpleSNNController.load(args.checkpoint)

    # Validate compatibility
    report = validate_akida_compatibility(model)
    print(report)

    if args.validate:
        return

    # Export
    if args.format == "nir":
        export_to_nir(model, args.output)
    elif args.format == "onnx":
        export_to_onnx(model, args.output)


if __name__ == "__main__":
    main()
