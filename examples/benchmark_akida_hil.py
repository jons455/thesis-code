"""
TD-HIL (Time-Dilated Hardware-in-the-Loop) benchmark with Akida.

This example runs the EMBARK benchmark using a remote Akida inference server
(TCP). You can either:

  1. Run against **real Akida hardware**: point to your Akida server (host/port).
  2. Run against a **local echo server** (no hardware): use --use-echo to start
     a minimal TCP server that echoes the first 2 input floats as outputs, so
     you can validate the pipeline and latency metrics without hardware.
     (With echo, control performance is meaningless and safety violations are expected.)

The controller is built from:
  - RemoteAkidaPolicy (TCP client to Akida server)
  - TensorControllerAdapter + state/action processors (same interface as SNN
    benchmarks; processors must match what your deployed Akida model expects)

Requirements:
  - pip install embark gym-electric-motor
  - For real Akida: Akida inference server running on host:port with the same
    protocol (length-prefixed float32 in/out; optional trailing chip time).

Run:
  # With real Akida server (e.g. 192.168.1.100:5000)
  python examples/benchmark_akida_hil.py --host 192.168.1.100 --port 5000

  # With local echo server (no hardware)
  python examples/benchmark_akida_hil.py --use-echo

  # Quick suite (2 scenarios), save results
  python examples/benchmark_akida_hil.py --use-echo --quick --output results/akida_hil.json
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import struct
import sys
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional echo server (for testing without Akida hardware)
# ---------------------------------------------------------------------------


class _EchoServer:
    """
    Minimal TCP server that mimics the Akida inference protocol:
    receive length-prefixed float32 payload, return first 2 floats as action.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self.host, self.port = self._server.getsockname()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((self.host, self.port))
            s.close()
        except OSError:
            pass
        self._server.close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _serve(self) -> None:
        self._server.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                conn.settimeout(5.0)
                while not self._stop.is_set():
                    try:
                        len_buf = self._recv_all(conn, 4)
                        if len_buf is None:
                            break
                        n = struct.unpack("!I", len_buf)[0]
                        payload = self._recv_all(conn, n)
                        if payload is None:
                            break
                        arr = np.frombuffer(payload, dtype=np.float32)
                        out = np.zeros(2, dtype=np.float32)
                        out[: min(2, len(arr))] = arr[:2]
                        conn.sendall(struct.pack("!I", out.nbytes) + out.tobytes())
                    except (socket.timeout, ConnectionError, OSError):
                        break

    @staticmethod
    def _recv_all(sock: socket.socket, n: int) -> bytes | None:
        data = bytearray()
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)


# ---------------------------------------------------------------------------
# Imports (after optional echo server so script can be run without embark
# for --help)
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run EMBARK benchmark with Akida TD-HIL (remote or echo server)."
    )
    p.add_argument(
        "--host",
        type=str,
        default=os.environ.get("AKIDA_HOST", ""),
        help="Akida server host (or set AKIDA_HOST). Ignored if --use-echo.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AKIDA_PORT", "5000")),
        help="Akida server port (default: 5000, or AKIDA_PORT).",
    )
    p.add_argument(
        "--use-echo",
        action="store_true",
        help="Start a local TCP echo server and run benchmark against it (no hardware).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use QUICK_SCENARIOS (2 scenarios) instead of STANDARD_SCENARIOS (6).",
    )
    p.add_argument(
        "--processor",
        choices=("identity", "rate_snn_5", "rate_snn_12"),
        default="rate_snn_5",
        help="State/action processor set: identity (minimal), rate_snn_5 or rate_snn_12 to match trained SNN.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="Akida-HIL",
        help="Controller name in benchmark summary.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="If set, save BenchmarkSummary to this JSON path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress per scenario (default: True).",
    )
    p.add_argument(
        "--target-tau-us",
        type=float,
        default=None,
        help=(
            "Target control period in microseconds for direct-control feasibility "
            "(default: benchmark tau)."
        ),
    )
    p.add_argument(
        "--feasibility-percentile",
        type=int,
        choices=(95, 99),
        default=95,
        help="Use chip_p95_us or chip_p99_us for pass/fail check (default: 95).",
    )
    return p.parse_args()


def _build_controller(host: str, port: int, processor_kind: str):
    """Build RemoteAkidaPolicy + TensorControllerAdapter with chosen processors."""
    from embark.benchmark import TensorControllerAdapter
    from embark.benchmark.controllers.remote import RemoteAkidaPolicy
    from embark.benchmark.processors import (
        IdentityActionProcessor,
        IdentityStateProcessor,
        RateSNNActionProcessor,
        RateSNNStateProcessor,
    )

    policy = RemoteAkidaPolicy(
        host=host,
        port=port,
        output_shape=(2,),
        timeout_s=10.0,
    )

    if processor_kind == "identity":
        state_processor = IdentityStateProcessor(
            state_keys=["i_d", "i_q"],
            reference_keys=["i_d_ref", "i_q_ref"],
        )
        action_processor = IdentityActionProcessor(action_keys=["v_d", "v_q"])
    elif processor_kind == "rate_snn_5":
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
        )
        action_processor = RateSNNActionProcessor(incremental=False)
    elif processor_kind == "rate_snn_12":
        state_processor = RateSNNStateProcessor(
            include_currents=True,
            include_errors=True,
            include_speed=True,
            include_derivatives=True,
            include_ema_slow=True,
            include_ema_fast=True,
        )
        action_processor = RateSNNActionProcessor(incremental=False)
    else:
        raise ValueError(f"Unknown processor: {processor_kind}")

    adapter = TensorControllerAdapter(
        controller=policy,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    return adapter


def _print_direct_control_feasibility(summary, target_tau_us: float, percentile: int) -> None:
    """
    Print direct Akida->PMSM control feasibility from on-chip latency metrics.

    Uses per-scenario chip percentiles and reports:
    - primary scenario estimate (step_mid_speed_1500rpm_2A when present),
    - worst-case across all scenarios (conservative).
    """
    key = "chip_p95_us" if percentile == 95 else "chip_p99_us"

    chip_values: dict[str, float] = {}
    for scenario in summary.scenario_results:
        metric_value = scenario.metrics.get(key)
        if metric_value is None:
            continue
        value = float(metric_value)
        if value > 0.0:
            chip_values[scenario.scenario_name] = value

    if not chip_values:
        print("\nNo chip timing metrics found in summary; cannot evaluate direct-control feasibility.")
        return

    primary_name = "step_mid_speed_1500rpm_2A"
    if primary_name in chip_values:
        selected_name = primary_name
    else:
        # Fallback to first scenario with chip timing
        selected_name = next(iter(chip_values.keys()))

    selected_us = chip_values[selected_name]
    worst_us = max(chip_values.values())

    selected_khz = 1000.0 / selected_us
    worst_khz = 1000.0 / worst_us

    selected_pass = selected_us <= target_tau_us
    worst_pass = worst_us <= target_tau_us

    print("\n=== Direct PMSM Control Feasibility ===")
    print(f"Metric basis: {key} (on-chip only, no TCP round-trip)")
    print(f"Target control period: {target_tau_us:.1f} us")
    print(
        f"{selected_name}: {selected_us:.1f} us  ->  f_max ~ {selected_khz:.2f} kHz  "
        f"->  {'PASS' if selected_pass else 'FAIL'}"
    )
    print(
        f"Worst across scenarios: {worst_us:.1f} us  ->  f_max ~ {worst_khz:.2f} kHz  "
        f"->  {'PASS' if worst_pass else 'FAIL'}"
    )
    print("Note: this estimates direct Akida prediction budget, excluding host/TCP overhead.")


def main() -> None:
    args = _parse_args()

    if args.use_echo:
        server = _EchoServer(host="127.0.0.1", port=0)
        server.start()
        host, port = server.host, server.port
        print(f"Echo server listening on {host}:{port}")
    else:
        if not args.host:
            print("Error: --host or AKIDA_HOST required unless --use-echo.", file=sys.stderr)
            sys.exit(1)
        host, port = args.host, args.port
        server = None

    from embark.benchmark import (
        BenchmarkSuite,
        QUICK_SCENARIOS,
        STANDARD_SCENARIOS,
    )

    scenarios = QUICK_SCENARIOS if args.quick else STANDARD_SCENARIOS
    suite = BenchmarkSuite(scenarios=scenarios, verbose=args.verbose)

    controller = _build_controller(host, port, args.processor)

    summary = suite.run(controller=controller, name=args.name)

    if server is not None:
        server.stop()

    print(suite.format_summary(summary))

    if hasattr(controller, "controller") and hasattr(controller.controller, "latencies"):
        latencies = controller.controller.latencies
        if latencies:
            mean_ms = statistics.mean(latencies) * 1000
            print(f"\nInference latency (round-trip): mean = {mean_ms:.2f} ms, n = {len(latencies)}")
        chip = getattr(controller.controller, "chip_latencies", None)
        if chip and chip:
            mean_chip_ms = statistics.mean(chip) * 1000
            print(f"Chip inference time: mean = {mean_chip_ms:.2f} ms")

    target_tau_us = (
        float(args.target_tau_us)
        if args.target_tau_us is not None
        else float(summary.config.get("tau", 1e-4) * 1e6)
    )
    _print_direct_control_feasibility(
        summary=summary,
        target_tau_us=target_tau_us,
        percentile=args.feasibility_percentile,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suite.save_results(summary, out_path)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
