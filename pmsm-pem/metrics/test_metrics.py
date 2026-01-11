"""
Test script for benchmark metrics.
Run this to verify the metrics framework works with your simulation data.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from metrics import (
    run_benchmark,
    compute_accuracy_metrics,
    compute_dynamics_metrics,
    compute_efficiency_metrics,
    compute_safety_metrics,
    PMSMParameters,
    DEFAULT_MOTOR,
)


def test_with_real_data():
    """Test metrics with actual simulation data."""
    
    # Find a test CSV
    export_dir = Path(__file__).parent.parent / "export" / "train"
    csv_files = list(export_dir.glob("sim_*.csv"))
    
    if not csv_files:
        print("No CSV files found in export/train/")
        return False
    
    # Load first file
    csv_path = csv_files[0]
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Time range: {df['time'].min():.4f}s to {df['time'].max():.4f}s")
    print(f"Speed: {df['n'].mean():.0f} rpm")
    print(f"i_d_ref: {df['i_d_ref'].iloc[-1]:.2f} A")
    print(f"i_q_ref: {df['i_q_ref'].iloc[-1]:.2f} A")
    
    # Run full benchmark
    print("\n" + "="*70)
    print("RUNNING BENCHMARK")
    print("="*70)
    
    result = run_benchmark(
        df,
        controller_name="GEM Standard PI",
        operating_point=f"id={df['i_d_ref'].iloc[-1]:.1f}A, iq={df['i_q_ref'].iloc[-1]:.1f}A",
    )
    
    print(result.summary())
    
    # Export to dict
    print("\n" + "-"*70)
    print("METRICS AS DICT (for CSV export)")
    print("-"*70)
    metrics_dict = result.to_dict()
    for key, value in list(metrics_dict.items())[:20]:  # First 20
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("  ...")
    
    return True


def test_with_synthetic_data():
    """Test metrics with synthetic step response data."""
    
    print("\n" + "="*70)
    print("TESTING WITH SYNTHETIC STEP RESPONSE")
    print("="*70)
    
    # Create synthetic step response
    dt = 1e-4  # 100 µs
    t_max = 0.2  # 200 ms
    time = np.arange(0, t_max, dt)
    n = len(time)
    
    # Step at t=0.01s
    step_time = 0.01
    i_q_ref = np.where(time >= step_time, 5.0, 0.0)
    i_d_ref = np.zeros(n)
    
    # Simulate second-order response with overshoot
    # i_q ≈ i_q_ref * (1 - exp(-t/τ) * (cos(ωt) + sin(ωt)))
    tau = 0.005  # 5 ms time constant
    omega_n = 500  # rad/s natural frequency
    zeta = 0.5  # damping ratio
    
    t_from_step = np.maximum(0, time - step_time)
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    response = np.where(
        time >= step_time,
        5.0 * (1 - np.exp(-zeta * omega_n * t_from_step) * 
               (np.cos(omega_d * t_from_step) + 
                zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t_from_step))),
        0.0
    )
    
    i_q = response + np.random.normal(0, 0.01, n)  # Add noise
    i_d = np.random.normal(0, 0.02, n)
    
    # Voltages (approximate)
    u_q = 10 + np.random.normal(0, 0.1, n)
    u_d = -2 + np.random.normal(0, 0.1, n)
    
    # Speed
    speed = np.full(n, 1500.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'i_d': i_d,
        'i_q': i_q,
        'n': speed,
        'u_d': u_d,
        'u_q': u_q,
        'i_d_ref': i_d_ref,
        'i_q_ref': i_q_ref,
    })
    
    # Test individual metric functions
    print("\n1. ACCURACY METRICS:")
    accuracy = compute_accuracy_metrics(
        time, i_d, i_q, i_d_ref, i_q_ref
    )
    print(f"   ITAE_iq: {accuracy.ITAE_iq:.6f} A·s²")
    print(f"   MAE_iq: {accuracy.MAE_iq:.4f} A")
    print(f"   SS_error_iq: {accuracy.SS_error_iq:.4f} A")
    
    print("\n2. DYNAMICS METRICS:")
    dynamics = compute_dynamics_metrics(
        time, i_d, i_q, i_d_ref, i_q_ref, step_time
    )
    print(f"   Rise Time: {dynamics.rise_time_iq*1000:.2f} ms")
    print(f"   Settling Time: {dynamics.settling_time_iq*1000:.2f} ms")
    print(f"   Overshoot: {dynamics.overshoot_iq:.1f}%")
    
    print("\n3. EFFICIENCY METRICS:")
    efficiency = compute_efficiency_metrics(
        time, i_d, i_q, u_d, u_q, speed
    )
    print(f"   P_copper_mean: {efficiency.P_copper_mean:.2f} W")
    print(f"   eta_mean: {efficiency.eta_mean:.1f}%")
    
    print("\n4. SAFETY METRICS:")
    safety = compute_safety_metrics(
        time, i_d, i_q, u_d, u_q
    )
    print(f"   Current violations: {safety.current_violations}")
    print(f"   Max |I|: {efficiency.i_magnitude_max:.2f} A")
    
    # Full benchmark
    print("\n" + "-"*70)
    result = run_benchmark(df, "Synthetic Test", "Step Response Test", step_time)
    print(result.summary())
    
    return True


def test_neuromorphic_metrics():
    """Test neuromorphic metrics with synthetic spike data."""
    from metrics import compute_neuromorphic_metrics_from_spikes
    
    print("\n" + "="*70)
    print("TESTING NEUROMORPHIC METRICS")
    print("="*70)
    
    # Create synthetic SNN data
    num_neurons = 100
    num_timesteps = 1000
    dt_snn = 1e-4  # 100 µs
    
    # Sparse spike trains (10% average firing rate)
    spike_probability = 0.1
    spike_trains = (np.random.random((num_neurons, num_timesteps)) < spike_probability).astype(float)
    
    # Random weight matrix with 50% sparsity
    weights = np.random.randn(num_neurons, num_neurons)
    weights[np.random.random(weights.shape) < 0.5] = 0
    
    # Compute metrics
    neuro = compute_neuromorphic_metrics_from_spikes(
        spike_trains=spike_trains,
        weights=weights,
        dt_snn=dt_snn,
        platform_energy_per_syop=23e-12,  # 23 pJ (Loihi 2)
    )
    
    print(f"\nNetwork: {num_neurons} neurons, {num_timesteps} timesteps")
    print(f"\n1. SPIKE STATISTICS:")
    print(f"   Total spikes: {neuro.total_spikes}")
    print(f"   Mean spike rate: {neuro.spike_rate_mean:.1f} Hz")
    print(f"   Spikes per inference: {neuro.spikes_per_inference:.1f}")
    
    print(f"\n2. SPARSITY:")
    print(f"   Activation sparsity: {neuro.activation_sparsity*100:.1f}%")
    print(f"   Temporal sparsity: {neuro.temporal_sparsity*100:.1f}%")
    print(f"   Connection sparsity: {neuro.connection_sparsity*100:.1f}%")
    
    print(f"\n3. COMPUTATIONAL EFFICIENCY:")
    print(f"   Total SyOps: {neuro.total_syops:,}")
    print(f"   SyOps/timestep: {neuro.syops_per_timestep:.0f}")
    print(f"   MAC reduction factor: {neuro.mac_reduction_factor:.1f}×")
    
    print(f"\n4. ENERGY:")
    print(f"   Energy/inference: {neuro.energy_per_inference*1e9:.2f} nJ")
    print(f"   Dynamic power: {neuro.dynamic_power*1e3:.2f} mW")
    
    print(f"\n5. LATENCY:")
    print(f"   Inference latency: {neuro.inference_latency_mean*1e6:.0f} µs")
    
    print(f"\n6. MEMORY:")
    print(f"   Weight memory: {neuro.weight_memory_bytes/1024:.1f} KB")
    print(f"   State memory: {neuro.state_memory_bytes} bytes")
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK METRICS TEST SUITE")
    print("="*70)
    
    # Test with synthetic data first
    test_with_synthetic_data()
    
    # Test neuromorphic metrics
    test_neuromorphic_metrics()
    
    # Test with real data
    test_with_real_data()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
