"""Controller interface for PMSM benchmark.

This module defines the standard interface that any controller must implement
to be tested with the PMSM current control benchmark. The interface is
NeuroBench-compatible while remaining simple for any control approach.

Example:
    Test your controller with the benchmark::

        from benchmark.controller_interface import run_benchmark
        from benchmark.pmsm_env import PMSMEnv

        env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0)
        results = run_benchmark(your_controller, env)

    Implement a custom controller::

        class MyController(ControllerAgent):
            def __init__(self):
                self.my_model = load_my_model()
                self._state = None

            def __call__(self, state: np.ndarray) -> np.ndarray:
                action = self.my_model.predict(state)
                return np.clip(action, -1.0, 1.0)

            def reset(self):
                self._state = None

            def get_info(self) -> dict:
                return {"model_type": "my_controller", "params": 1234}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ControllerInterface(Protocol):
    """Protocol defining the minimum interface for a controller.

    Uses Python's Protocol for structural typing - any class that implements
    these methods is automatically compatible, no inheritance needed.
    """

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute control action from state.

        Args:
            state: Normalized state vector [i_d, i_q, e_d, e_q] in range [-1, 1].
                i_d, i_q are currents normalized by i_max.
                e_d, e_q are errors normalized by i_max.

        Returns:
            Normalized action [u_d, u_q] in range [-1, 1], voltages / u_max.
        """
        ...

    def reset(self) -> None:
        """Reset controller state (integrators, hidden states, etc.)."""
        ...


class ControllerAgent(ABC):
    """Abstract base class for controllers in the PMSM benchmark.

    Provides a common interface for both classical and neural network controllers.
    Subclass this and implement the abstract methods for your controller.
    Follows the NeuroBench agent interface pattern.
    """

    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute control action from state.

        Parameters
        ----------
        state : np.ndarray
            Normalized state [i_d, i_q, e_d, e_q] in [-1, 1]

        Returns
        -------
        np.ndarray
            Normalized action [u_d, u_q] in [-1, 1]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state for new episode."""
        pass

    def get_info(self) -> dict[str, Any]:
        """
        Return controller metadata for benchmark reporting.

        Override to provide controller-specific information.

        Returns
        -------
        dict
            Controller metadata (name, parameters, architecture, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "type": "unknown",
        }

    def get_neuromorphic_metrics(self) -> dict[str, Any] | None:
        """
        Return neuromorphic metrics if available.

        Override for SNN controllers to provide spike counts, sparsity, etc.

        Returns
        -------
        dict or None
            Neuromorphic metrics if available:
            - total_spikes: Total spike count
            - spikes_per_step: Average spikes per control step
            - sparsity: Activation sparsity (0-1)
            - num_neurons: Neuron count
            - num_synapses: Synapse count
            - inference_latency: Time per inference (seconds)
        """
        return None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Operating point
    n_rpm: float = 1000.0  # Motor speed [rpm]
    i_d_ref: float = 0.0  # d-axis current reference [A]
    i_q_ref: float = 5.0  # q-axis current reference [A]

    # Simulation
    max_steps: int = 2000  # Maximum steps per episode
    num_episodes: int = 1  # Number of episodes to run

    # Motor limits (for reference)
    i_max: float = 10.8  # Maximum current [A]
    u_max: float = 48.0  # Maximum voltage [V]
    tau: float = 1e-4  # Control period [s] (10 kHz)


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    # Metadata
    controller_name: str
    timestamp: str
    config: BenchmarkConfig

    # Accuracy metrics
    rmse_id: float  # RMSE of i_d tracking [A]
    rmse_iq: float  # RMSE of i_q tracking [A]
    mae_id: float  # Mean absolute error i_d [A]
    mae_iq: float  # Mean absolute error i_q [A]
    final_error_id: float  # Final tracking error i_d [A]
    final_error_iq: float  # Final tracking error i_q [A]

    # Dynamics metrics
    settling_time_id: float  # Settling time i_d [s]
    settling_time_iq: float  # Settling time i_q [s]
    overshoot_id: float  # Overshoot i_d [%]
    overshoot_iq: float  # Overshoot i_q [%]
    rise_time_iq: float  # Rise time i_q [s]

    # Stability metrics
    total_variation: float  # Control signal variation
    is_stable: bool  # No oscillation or divergence

    # Neuromorphic metrics (SNN only)
    total_spikes: int | None = None
    spikes_per_step: float | None = None
    sparsity: float | None = None
    num_neurons: int | None = None
    num_synapses: int | None = None
    inference_latency_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "controller_name": self.controller_name,
            "timestamp": self.timestamp,
            "config": {
                "n_rpm": self.config.n_rpm,
                "i_d_ref": self.config.i_d_ref,
                "i_q_ref": self.config.i_q_ref,
                "max_steps": self.config.max_steps,
            },
            "accuracy": {
                "rmse_id": self.rmse_id,
                "rmse_iq": self.rmse_iq,
                "mae_id": self.mae_id,
                "mae_iq": self.mae_iq,
                "final_error_id": self.final_error_id,
                "final_error_iq": self.final_error_iq,
            },
            "dynamics": {
                "settling_time_id": self.settling_time_id,
                "settling_time_iq": self.settling_time_iq,
                "overshoot_id": self.overshoot_id,
                "overshoot_iq": self.overshoot_iq,
                "rise_time_iq": self.rise_time_iq,
            },
            "stability": {
                "total_variation": self.total_variation,
                "is_stable": self.is_stable,
            },
            "neuromorphic": {
                "total_spikes": self.total_spikes,
                "spikes_per_step": self.spikes_per_step,
                "sparsity": self.sparsity,
                "num_neurons": self.num_neurons,
                "num_synapses": self.num_synapses,
                "inference_latency_mean": self.inference_latency_mean,
            }
            if self.total_spikes is not None
            else None,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== Benchmark Results: {self.controller_name} ===",
            f"Operating Point: {self.config.i_q_ref:.1f}A @ {self.config.n_rpm:.0f} rpm",
            "",
            "Accuracy:",
            f"  RMSE i_q: {self.rmse_iq*1000:.2f} mA",
            f"  RMSE i_d: {self.rmse_id*1000:.2f} mA",
            f"  Final error: {np.sqrt(self.final_error_id**2 + self.final_error_iq**2)*1000:.2f} mA",
            "",
            "Dynamics:",
            f"  Settling time i_q: {self.settling_time_iq*1000:.1f} ms",
            f"  Rise time i_q: {self.rise_time_iq*1000:.1f} ms",
            f"  Overshoot i_q: {self.overshoot_iq:.1f}%",
            "",
            "Stability:",
            f"  Total variation: {self.total_variation:.2f}",
            f"  Stable: {'Yes' if self.is_stable else 'No'}",
        ]

        if self.total_spikes is not None:
            lines.extend(
                [
                    "",
                    "Neuromorphic:",
                    f"  Total spikes: {self.total_spikes:,}",
                    f"  Spikes/step: {self.spikes_per_step:.1f}",
                    f"  Sparsity: {self.sparsity*100:.1f}%",
                    f"  Neurons: {self.num_neurons}",
                    f"  Synapses: {self.num_synapses:,}",
                    f"  Latency: {self.inference_latency_mean*1e6:.1f} µs"
                    if self.inference_latency_mean
                    else "",
                ]
            )

        return "\n".join(lines)


def run_benchmark(
    controller: ControllerInterface,
    env,  # PMSMEnv
    config: BenchmarkConfig | None = None,
    verbose: bool = True,
) -> BenchmarkResults:
    """
    Run benchmark for a controller.

    This is the main entry point for benchmarking any controller.

    Parameters
    ----------
    controller : ControllerInterface
        Any object implementing __call__(state) -> action and reset()
    env : PMSMEnv
        PMSM environment instance
    config : BenchmarkConfig, optional
        Benchmark configuration
    verbose : bool
        Print progress

    Returns
    -------
    BenchmarkResults
        Complete benchmark results

    Example
    -------
        from benchmark.controller_interface import run_benchmark
        from benchmark.pmsm_env import PMSMEnv
        from benchmark.agents import PIControllerAgent

        env = PMSMEnv(n_rpm=1000, i_d_ref=0.0, i_q_ref=5.0, max_steps=2000)
        controller = PIControllerAgent()

        results = run_benchmark(controller, env)
        print(results.summary())
    """
    if config is None:
        config = BenchmarkConfig(
            n_rpm=env.n_rpm,
            i_d_ref=env.i_d_ref,
            i_q_ref=env.i_q_ref,
            max_steps=env.max_steps,
        )

    # Validate controller interface
    if not isinstance(controller, ControllerInterface):
        raise TypeError(
            "Controller must implement ControllerInterface: "
            "__call__(state) -> action and reset()"
        )

    if verbose:
        print(f"Running benchmark for: {controller.__class__.__name__}")
        print(f"  Operating point: i_q_ref={config.i_q_ref}A @ {config.n_rpm} rpm")

    # Reset environment and controller
    state, info = env.reset()
    controller.reset()

    # Collect data
    i_d_list = []
    i_q_list = []
    i_d_ref_list = []
    i_q_ref_list = []
    u_d_list = []
    u_q_list = []
    time_list = []

    # Run episode
    for _ in range(config.max_steps):
        # Get action from controller
        action = controller(state)

        # Record data
        episode_data = env.get_episode_data()
        if episode_data:
            d = episode_data[-1] if len(episode_data) > 0 else None
            if d:
                i_d_list.append(d["i_d"])
                i_q_list.append(d["i_q"])
                i_d_ref_list.append(d["i_d_ref"])
                i_q_ref_list.append(d["i_q_ref"])
                time_list.append(d["time"])

        u_d_list.append(action[0] * config.u_max)
        u_q_list.append(action[1] * config.u_max)

        # Step environment
        state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break

    # Ensure we have the final data point
    episode_data = env.get_episode_data()
    if episode_data and len(i_d_list) < len(episode_data):
        d = episode_data[-1]
        i_d_list.append(d["i_d"])
        i_q_list.append(d["i_q"])
        i_d_ref_list.append(d["i_d_ref"])
        i_q_ref_list.append(d["i_q_ref"])
        time_list.append(d["time"])

    # Convert to arrays
    i_d = np.array(i_d_list)
    i_q = np.array(i_q_list)
    i_d_ref = np.array(i_d_ref_list)
    i_q_ref = np.array(i_q_ref_list)
    u_d = np.array(u_d_list[: len(i_d)])
    u_q = np.array(u_q_list[: len(i_q)])
    time = np.array(time_list)

    # Compute errors
    e_d = i_d_ref - i_d
    e_q = i_q_ref - i_q

    # Accuracy metrics
    rmse_id = float(np.sqrt(np.mean(e_d**2)))
    rmse_iq = float(np.sqrt(np.mean(e_q**2)))
    mae_id = float(np.mean(np.abs(e_d)))
    mae_iq = float(np.mean(np.abs(e_q)))
    final_error_id = float(np.abs(e_d[-1])) if len(e_d) > 0 else 0.0
    final_error_iq = float(np.abs(e_q[-1])) if len(e_q) > 0 else 0.0

    # Dynamics metrics
    settling_time_id = _compute_settling_time(time, i_d, i_d_ref)
    settling_time_iq = _compute_settling_time(time, i_q, i_q_ref)
    overshoot_id = _compute_overshoot(i_d, i_d_ref)
    overshoot_iq = _compute_overshoot(i_q, i_q_ref)
    rise_time_iq = _compute_rise_time(time, i_q, i_q_ref)

    # Stability metrics
    total_variation = float(np.sum(np.abs(np.diff(u_d))) + np.sum(np.abs(np.diff(u_q))))
    is_stable = not (np.isnan(rmse_iq) or rmse_iq > 5.0)  # Basic stability check

    # Neuromorphic metrics
    neuro_metrics = None
    if hasattr(controller, "get_neuromorphic_metrics"):
        neuro_metrics = controller.get_neuromorphic_metrics()
    elif hasattr(controller, "get_spike_statistics"):
        stats = controller.get_spike_statistics()
        if stats and "error" not in stats:
            neuro_metrics = {
                "total_spikes": stats.get("total_spikes", 0),
                "spikes_per_step": stats.get("spikes_per_control_step", 0),
                "sparsity": stats.get("mean_sparsity", 0),
                "num_neurons": stats.get("num_neurons", 0),
                "num_synapses": stats.get("num_synapses", 0),
                "inference_latency_mean": stats.get("inference_latency_mean_s", None),
            }

    # Get controller name
    controller_name = controller.__class__.__name__
    if hasattr(controller, "get_info"):
        info = controller.get_info()
        controller_name = info.get("name", controller_name)

    results = BenchmarkResults(
        controller_name=controller_name,
        timestamp=datetime.now().isoformat(),
        config=config,
        rmse_id=rmse_id,
        rmse_iq=rmse_iq,
        mae_id=mae_id,
        mae_iq=mae_iq,
        final_error_id=final_error_id,
        final_error_iq=final_error_iq,
        settling_time_id=settling_time_id,
        settling_time_iq=settling_time_iq,
        overshoot_id=overshoot_id,
        overshoot_iq=overshoot_iq,
        rise_time_iq=rise_time_iq,
        total_variation=total_variation,
        is_stable=is_stable,
        total_spikes=neuro_metrics.get("total_spikes") if neuro_metrics else None,
        spikes_per_step=neuro_metrics.get("spikes_per_step") if neuro_metrics else None,
        sparsity=neuro_metrics.get("sparsity") if neuro_metrics else None,
        num_neurons=neuro_metrics.get("num_neurons") if neuro_metrics else None,
        num_synapses=neuro_metrics.get("num_synapses") if neuro_metrics else None,
        inference_latency_mean=neuro_metrics.get("inference_latency_mean")
        if neuro_metrics
        else None,
    )

    if verbose:
        print(f"  Done! RMSE: {rmse_iq*1000:.2f} mA")

    return results


def _compute_settling_time(
    time: np.ndarray, y: np.ndarray, ref: np.ndarray, tolerance: float = 0.02
) -> float:
    """Compute settling time (time to stay within tolerance of final value)."""
    if len(y) == 0:
        return float("inf")

    final_ref = ref[-1]
    threshold = tolerance * abs(final_ref) if final_ref != 0 else tolerance

    # Find last time outside tolerance band
    within_band = np.abs(y - final_ref) <= threshold
    if not np.any(within_band):
        return float("inf")

    # Find first time that stays within band until end
    for i in range(len(within_band)):
        if np.all(within_band[i:]):
            return float(time[i])

    return float(time[-1])


def _compute_overshoot(y: np.ndarray, ref: np.ndarray) -> float:
    """Compute overshoot as percentage of step size."""
    if len(y) == 0:
        return 0.0

    final_ref = ref[-1]
    initial = y[0]
    step_size = final_ref - initial

    if abs(step_size) < 1e-6:
        return 0.0

    if step_size > 0:
        overshoot = np.max(y) - final_ref
    else:
        overshoot = final_ref - np.min(y)

    return float(max(0, overshoot / abs(step_size) * 100))


def _compute_rise_time(
    time: np.ndarray,
    y: np.ndarray,
    ref: np.ndarray,
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    """Compute rise time (10% to 90% of step)."""
    if len(y) < 2:
        return float("inf")

    final_ref = ref[-1]
    initial = y[0]
    step_size = final_ref - initial

    if abs(step_size) < 1e-6:
        return 0.0

    lower_val = initial + lower * step_size
    upper_val = initial + upper * step_size

    # Find crossings
    t_lower = None
    t_upper = None

    for i in range(1, len(y)):
        if t_lower is None and (
            (y[i - 1] <= lower_val <= y[i]) or (y[i - 1] >= lower_val >= y[i])
        ):
            t_lower = time[i]
        if t_upper is None and (
            (y[i - 1] <= upper_val <= y[i]) or (y[i - 1] >= upper_val >= y[i])
        ):
            t_upper = time[i]
        if t_lower is not None and t_upper is not None:
            break

    if t_lower is None or t_upper is None:
        return float("inf")

    return float(t_upper - t_lower)


def compare_controllers(
    controllers: list, env, config: BenchmarkConfig | None = None
) -> dict:
    """
    Compare multiple controllers on the same operating point.

    Parameters
    ----------
    controllers : list
        List of (name, controller) tuples
    env : PMSMEnv
        Environment instance
    config : BenchmarkConfig, optional
        Benchmark configuration

    Returns
    -------
    dict
        Results for each controller
    """
    results = {}

    for name, controller in controllers:
        print(f"\nBenchmarking: {name}")
        env.reset()
        result = run_benchmark(controller, env, config, verbose=True)
        results[name] = result

    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(
        f"{'Controller':<20} {'RMSE_iq [mA]':>12} {'Settling [ms]':>14} {'Overshoot [%]':>14}"
    )
    print("-" * 70)

    for name, result in results.items():
        print(
            f"{name:<20} {result.rmse_iq*1000:>12.2f} {result.settling_time_iq*1000:>14.1f} {result.overshoot_iq:>14.1f}"
        )

    return results
