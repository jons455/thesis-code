"""
Akida inference server for HIL integration.

This server receives length-prefixed float32 inputs over TCP, runs Akida
inference, and returns:
    [action floats..., chip_inference_time_s]

`chip_inference_time_s` is measured on-device around `model.predict(...)`
for each request. This is the right metric for direct Akida->PMSM control
feasibility checks (p95/p99 against control period budget).
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
import traceback
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np


def _parse_shape(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Shape must have at least one dimension.")
    return tuple(int(p) for p in parts)


def _recv_all(sock: socket.socket, n_bytes: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _prepare_inputs(raw: bytes, input_shape: Tuple[int, ...] | None) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.float32)
    if not input_shape:
        return arr

    reshaped = arr.reshape(input_shape)
    # Common Akida model input expects explicit batch dimension.
    if reshaped.ndim == 3:
        reshaped = np.expand_dims(reshaped, axis=0)
    return reshaped


def _flatten_action(outputs: np.ndarray) -> np.ndarray:
    """Convert model outputs to a 1D float32 action vector."""
    out = np.asarray(outputs, dtype=np.float32)
    out = np.squeeze(out)
    if out.ndim == 0:
        out = out.reshape(1)
    return out.reshape(-1)


def _run_predict_once(model: object, inputs: np.ndarray) -> np.ndarray:
    outputs = model.predict(inputs)
    return np.asarray(outputs, dtype=np.float32)


def _run_inference_timed(
    model: object,
    inputs: np.ndarray,
    repeats: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Run inference and return (outputs, chip_inference_time_s).

    For repeats > 1, returns outputs from the last run and median latency.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    samples_s: list[float] = []
    outputs: np.ndarray | None = None

    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        outputs = _run_predict_once(model, inputs)
        t1 = time.perf_counter_ns()
        samples_s.append((t1 - t0) * 1e-9)

    assert outputs is not None
    chip_time_s = float(np.median(samples_s))
    return outputs, chip_time_s


@dataclass
class RunningStats:
    chip_times_s: list[float] = field(default_factory=list)
    request_count: int = 0

    def add(self, chip_time_s: float) -> None:
        self.request_count += 1
        self.chip_times_s.append(chip_time_s)

    def print_summary(self, target_tau_us: float | None = None) -> None:
        if not self.chip_times_s:
            print("No on-chip timings collected.")
            return

        arr = np.array(self.chip_times_s, dtype=np.float64)
        mean_us = float(np.mean(arr) * 1e6)
        p95_us = float(np.percentile(arr, 95) * 1e6)
        p99_us = float(np.percentile(arr, 99) * 1e6)
        max_us = float(np.max(arr) * 1e6)
        min_us = float(np.min(arr) * 1e6)

        print()
        print("=" * 56)
        print("On-Device Inference Summary")
        print("=" * 56)
        print(f"  Requests served: {len(arr)}")
        print(f"  Mean:           {mean_us:10.2f} us")
        print(f"  P95:            {p95_us:10.2f} us")
        print(f"  P99:            {p99_us:10.2f} us")
        print(f"  Max:            {max_us:10.2f} us")
        print(f"  Min:            {min_us:10.2f} us")
        if target_tau_us is not None:
            feasible = p95_us <= target_tau_us
            fmax_khz = 1000.0 / p95_us if p95_us > 0 else float("inf")
            print(f"  Target tau:     {target_tau_us:10.2f} us")
            print(f"  f_max(p95):     {fmax_khz:10.2f} kHz")
            print(f"  Feasible:       {'PASS' if feasible else 'FAIL'}")
        print("=" * 56)


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    model_path: str = "akida_model.fbz",
    echo: bool = False,
    input_shape: Tuple[int, ...] | None = (1, 1, -1),
    max_requests: Optional[int] = None,
    warmup_requests: int = 5,
    timing_repeats: int = 1,
    target_tau_us: float | None = None,
) -> None:
    model = None
    if not echo:
        try:
            import akida
            from akida import Model  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Akida SDK not available. Install it on the device or use --echo."
            ) from exc

        model = Model(model_path)
        devices = akida.devices()
        if devices:
            device = devices[0]
            print(f"Mapping model to Akida device: {device.desc}")
            try:
                model.map(device)
            except Exception as exc:
                print(f"WARNING: model.map() failed: {exc}")
                print("Continuing; inference may run in software mode.")
        else:
            print("WARNING: No Akida hardware detected (software mode).")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Listening on {host}:{port} (echo={echo})")

    conn, addr = server.accept()
    print(f"Connected by {addr}")

    stats = RunningStats()

    with conn:
        while True:
            try:
                len_bytes = _recv_all(conn, 4)
                if not len_bytes:
                    print("Client closed connection.")
                    break
                payload_len = struct.unpack("!I", len_bytes)[0]
                payload = _recv_all(conn, payload_len)
                if payload is None:
                    print("Client closed connection during payload read.")
                    break

                inputs = _prepare_inputs(payload, input_shape)

                if echo:
                    action = _flatten_action(inputs)
                    chip_time_s = 0.0
                else:
                    assert model is not None
                    # Optional warmup (not included in reported stats).
                    if stats.request_count < warmup_requests:
                        outputs = _run_predict_once(model, inputs)
                        action = _flatten_action(outputs)
                        chip_time_s = 0.0
                    else:
                        outputs, chip_time_s = _run_inference_timed(
                            model=model,
                            inputs=inputs,
                            repeats=timing_repeats,
                        )
                        action = _flatten_action(outputs)
                        stats.add(chip_time_s)

                action_bytes = action.astype(np.float32, copy=False).tobytes()
                chip_bytes = np.array([chip_time_s], dtype=np.float32).tobytes()
                response = action_bytes + chip_bytes
                conn.sendall(struct.pack("!I", len(response)) + response)

                if max_requests is not None and stats.request_count >= max_requests:
                    print(f"Reached max_requests={max_requests}, stopping.")
                    break

            except Exception as exc:
                print(f"Error processing request {stats.request_count + 1}: {exc}")
                traceback.print_exc()
                break

    stats.print_summary(target_tau_us=target_tau_us)
    server.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Akida on-device inference TCP server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-path", default="akida_model.fbz")
    parser.add_argument("--echo", action="store_true", help="Echo mode for quick testing.")
    parser.add_argument(
        "--input-shape",
        default="1,1,-1",
        help="Comma-separated model input shape, e.g. '1,1,-1'.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after N timed requests (warmup not counted).",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Number of warmup requests excluded from timing stats.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="Inference repeats per request; median is reported as chip time.",
    )
    parser.add_argument(
        "--target-tau-us",
        type=float,
        default=None,
        help="Optional direct-control budget for PASS/FAIL in summary.",
    )
    return parser


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(args=args)
    input_shape = _parse_shape(parsed.input_shape) if parsed.input_shape else None
    run_server(
        host=parsed.host,
        port=parsed.port,
        model_path=parsed.model_path,
        echo=parsed.echo,
        input_shape=input_shape,
        max_requests=parsed.max_requests,
        warmup_requests=parsed.warmup_requests,
        timing_repeats=parsed.timing_repeats,
        target_tau_us=parsed.target_tau_us,
    )


if __name__ == "__main__":
    main()
