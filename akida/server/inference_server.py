"""Akida inference server for HIL integration."""

from __future__ import annotations

import argparse
import socket
import struct
from typing import Iterable

import numpy as np


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("input_shape must have at least one dimension.")
    return tuple(int(p) for p in parts)


def _recv_all(sock: socket.socket, n_bytes: int) -> bytes | None:
    data = bytearray()
    while len(data) < n_bytes:
        chunk = sock.recv(n_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _run_inference(model: object, inputs: np.ndarray) -> np.ndarray:
    outputs = model.predict(inputs)
    return np.asarray(outputs, dtype=np.float32)


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    model_path: str = "akida_model.fbz",
    echo: bool = False,
    input_shape: tuple[int, ...] = (1, 1, -1),
    max_requests: int | None = None,
) -> None:
    model = None
    if not echo:
        try:
            from akida import Model  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Akida SDK not available. Install it on the Pi or use --echo."
            ) from exc
        model = Model(model_path)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"Listening on {host}:{port} (echo={echo})")

    conn, addr = server.accept()
    print(f"Connected by {addr}")
    request_count = 0

    with conn:
        while True:
            len_bytes = _recv_all(conn, 4)
            if not len_bytes:
                break
            payload_len = struct.unpack("!I", len_bytes)[0]
            payload = _recv_all(conn, payload_len)
            if payload is None:
                break

            inputs = np.frombuffer(payload, dtype=np.float32)
            if input_shape:
                inputs = inputs.reshape(input_shape)

            if echo:
                outputs = np.asarray(inputs, dtype=np.float32)
            else:
                outputs = _run_inference(model, inputs)

            response = outputs.astype(np.float32, copy=False).tobytes()
            conn.sendall(struct.pack("!I", len(response)) + response)

            request_count += 1
            if max_requests is not None and request_count >= max_requests:
                break

    server.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Akida inference TCP server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-path", default="akida_model.fbz")
    parser.add_argument("--echo", action="store_true", help="Echo inputs for testing.")
    parser.add_argument(
        "--input-shape",
        default="1,1,-1",
        help="Comma-separated input shape, e.g. '1,1,-1'.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after N requests (useful for tests).",
    )
    return parser


def main(args: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(args=args)
    input_shape = _parse_shape(parsed.input_shape)
    run_server(
        host=parsed.host,
        port=parsed.port,
        model_path=parsed.model_path,
        echo=parsed.echo,
        input_shape=input_shape,
        max_requests=parsed.max_requests,
    )


if __name__ == "__main__":
    main()
