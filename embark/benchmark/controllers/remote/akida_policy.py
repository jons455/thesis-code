"""Remote Akida controller client implementation."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from embark.benchmark.interfaces import TensorController


@dataclass
class RemoteAkidaPolicy(TensorController):
    """TCP client that forwards observations to a remote Akida server."""

    host: str
    port: int = 5000
    timeout_s: float | None = 10.0
    output_shape: tuple[int, ...] | None = None
    _socket: socket.socket | None = field(default=None, init=False, repr=False)
    _latencies: list[float] = field(default_factory=list, init=False, repr=False)
    _chip_latencies: list[float] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        """Reconnect to ensure a clean session."""
        self._close()
        self._connect()
        self._latencies.clear()
        self._chip_latencies.clear()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Serialize observation, send to server, return action tensor."""
        if self._socket is None:
            self._connect()

        obs_np = observation.detach().cpu().numpy().astype(np.float32, copy=False)
        payload = obs_np.tobytes()

        t0 = time.perf_counter()

        # Send length-prefixed payload
        self._socket.sendall(struct.pack("!I", len(payload)) + payload)

        # Receive response length
        len_bytes = self._recv_all(4)
        response_len = struct.unpack("!I", len_bytes)[0]

        # Receive response data
        response_data = self._recv_all(response_len)

        t1 = time.perf_counter()
        self._latencies.append(t1 - t0)

        # Server appends 1 extra float32 with on-chip inference time.
        # Split: all-but-last float32 = action, last float32 = chip time.
        all_floats = np.frombuffer(response_data, dtype=np.float32).copy()
        if len(all_floats) > 0 and self.output_shape is not None:
            expected_action_elems = int(np.prod(self.output_shape))
            if len(all_floats) == expected_action_elems + 1:
                chip_time = float(all_floats[-1])
                action_np = all_floats[:-1]
                self._chip_latencies.append(chip_time)
            else:
                # Fallback: no chip timing appended (old server)
                action_np = all_floats
        else:
            action_np = all_floats

        if self.output_shape is not None:
            action_np = action_np.reshape(self.output_shape)

        return torch.from_numpy(action_np)

    @property
    def last_info(self) -> dict[str, Any] | None:
        """Return timing info from the most recent forward() call."""
        if not self._latencies:
            return None
        info: dict[str, Any] = {"inference_latency_s": self._latencies[-1]}
        if self._chip_latencies:
            info["chip_inference_time_s"] = self._chip_latencies[-1]
        return info

    @property
    def latencies(self) -> list[float]:
        """Full history of per-step round-trip latencies in seconds."""
        return list(self._latencies)

    @property
    def chip_latencies(self) -> list[float]:
        """Full history of per-step on-chip inference times in seconds."""
        return list(self._chip_latencies)

    def get_state(self) -> dict[str, Any]:
        """Serialize minimal state for checkpointing."""
        return {"host": self.host, "port": self.port, "output_shape": self.output_shape}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore minimal state and reconnect."""
        self.host = state.get("host", self.host)
        self.port = state.get("port", self.port)
        self.output_shape = state.get("output_shape", self.output_shape)
        self.reset()

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.timeout_s is not None:
            sock.settimeout(self.timeout_s)
        sock.connect((self.host, self.port))
        self._socket = sock

    def _close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def _recv_all(self, n_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < n_bytes:
            chunk = self._socket.recv(n_bytes - len(data))
            if not chunk:
                raise ConnectionError("Server closed connection.")
            data.extend(chunk)
        return bytes(data)

    def __del__(self) -> None:
        self._close()
