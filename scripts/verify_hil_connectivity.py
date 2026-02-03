"""Verify Akida HIL TCP connectivity with an echo server."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import torch

from embark.benchmark.controllers.remote import RemoteAkidaPolicy


def _wait_for_port(host: str, port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Server on {host}:{port} did not start in time.")


def main() -> None:
    host = "127.0.0.1"
    port = 5001

    server_proc = subprocess.Popen(
        [
            sys.executable,
            "akida/server/inference_server.py",
            "--host",
            host,
            "--port",
            str(port),
            "--echo",
            "--max-requests",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_port(host, port)
        policy = RemoteAkidaPolicy(host=host, port=port, output_shape=(1, 4))
        policy.reset()

        observation = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        action = policy.forward(observation)

        if not torch.allclose(observation, action):
            raise RuntimeError("Echo verification failed: response differs from input.")

        print("HIL echo connectivity verified.")
    finally:
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
