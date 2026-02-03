"""Remote Akida controller client implementation."""

from __future__ import annotations

import socket
import struct
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

    def reset(self) -> None:
        """Reconnect to ensure a clean session."""
        self._close()
        self._connect()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Serialize observation, send to server, return action tensor."""
        if self._socket is None:
            self._connect()

        obs_np = observation.detach().cpu().numpy().astype(np.float32, copy=False)
        payload = obs_np.tobytes()

        # Send length-prefixed payload
        self._socket.sendall(struct.pack("!I", len(payload)) + payload)

        # Receive response length
        len_bytes = self._recv_all(4)
        response_len = struct.unpack("!I", len_bytes)[0]

        # Receive response data
        response_data = self._recv_all(response_len)
        action_np = np.frombuffer(response_data, dtype=np.float32)
        if self.output_shape is not None:
            action_np = action_np.reshape(self.output_shape)

        return torch.from_numpy(action_np)

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
