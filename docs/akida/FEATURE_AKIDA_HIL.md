# Feature: Akida Hardware-in-the-Loop (HIL) Integration

This document outlines the plan for integrating the Akida neuromorphic processor into the `embark` benchmarking framework using a "Time-Dilated" Hardware-in-the-Loop (HIL) approach.

## Overview

The integration treats the Akida hardware (hosted on a Raspberry Pi) as a "Remote Brain". The simulation runs on the PC, pauses to send observations to the Pi, and waits for the inference result. This architecture leverages the synchronous nature of the `ClosedLoopHarness` to avoid complex real-time synchronization issues.

---

## Architecture: The "Universal Socket"

By adhering to the `TensorController` protocol, the remote Akida policy becomes drop-in compatible with the simulation harness.

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Harness** | PC | Runs physics simulation, metrics, and orchestrates the loop. |
| **Client Policy** | PC | Implements `TensorController`. Serializes tensors and sends them over TCP. |
| **Server** | Raspberry Pi | Listens for tensors, runs Akida inference, returns actions. |
| **Akida Chip** | Raspberry Pi | Performs the actual neuromorphic inference. |

---

## Implementation Plan

### Phase 1: The "Echo" Test (Connectivity Verification)

**Goal:** Verify PC-Pi communication without ML complexity.

1.  **Server (Raspberry Pi)**: `echo_server.py`
    *   Listens on Port 5000.
    *   Receives a tensor.
    *   Returns the *same* tensor (or specific pattern).
2.  **Client (PC)**: `RemoteEchoPolicy`
    *   Connects to Pi.
    *   Implements `forward()` using `socket.send()` / `socket.recv()`.
3.  **Validation**:
    *   Run benchmark with `RemoteEchoPolicy`.
    *   Verify simulation runs (slowly) and logs valid trajectories.

### Phase 2: The "Remote Brain" (Akida Integration)

**Goal:** Replace echo logic with actual Akida inference.

1.  **Preparation**:
    *   Train Keras model on PC.
    *   Quantize using `quantizeml`.
    *   Convert to `.fbz` (Akida binary).
    *   Transfer `.fbz` to Pi.
2.  **Server (Raspberry Pi)**: `inference_server.py`
    *   Load `.fbz` model.
    *   Loop: Receive input -> `model.predict()` -> Send output.
3.  **Client (PC)**: `RemoteAkidaPolicy`
    *   Ensure data shape matches Akida expectations (Batch, Time, Features).

### Phase 3: The Benchmark

**Goal:** Run the formal comparison.

1.  **Configuration**: Use `StepResponseTask` or standard benchmark suite.
2.  **Execution**:
    *   Start server on Pi.
    *   Run `run_benchmark.py` on PC with `RemoteAkidaPolicy`.
3.  **Metrics**:
    *   Collect same metrics as simulation (RMSE, etc.).
    *   Compare Sim results vs. HIL results (should be near-identical).

---

## Reference Implementation

### Remote Policy (PC Side)

```python
import socket
import struct
import torch
import numpy as np
from embark.benchmark.interfaces import TensorController

class RemoteAkidaPolicy(TensorController):
    def __init__(self, host: str, port: int = 5000):
        self.host = host
        self.port = port
        self.socket = None

    def reset(self):
        # Reconnect on reset to ensure clean state
        if self.socket:
            self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        # Optional: Send reset signal to server if model is stateful

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1. Serialize
        data = obs.detach().cpu().numpy().astype(np.float32).tobytes()
        
        # 2. Send (Length + Data)
        self.socket.sendall(struct.pack('!I', len(data)) + data)
        
        # 3. Receive Response Length
        len_bytes = self.recv_all(4)
        if not len_bytes:
            raise ConnectionError("Server closed connection")
        response_len = struct.unpack('!I', len_bytes)[0]
        
        # 4. Receive Response Data
        response_data = self.recv_all(response_len)
        
        # 5. Deserialize
        action_np = np.frombuffer(response_data, dtype=np.float32)
        return torch.from_numpy(action_np)

    def recv_all(self, n: int) -> bytes:
        data = b''
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
```

### Inference Server (Pi Side)

```python
import socket
import struct
import numpy as np
# from akida import Model # Uncomment on Pi

def run_server(host='0.0.0.0', port=5000, model_path='model.fbz'):
    # model = Model(model_path) # Load Akida model
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"Listening on {host}:{port}...")
    
    conn, addr = server.accept()
    print(f"Connected by {addr}")
    
    while True:
        # Receive header
        len_bytes = recv_all(conn, 4)
        if not len_bytes: break
        length = struct.unpack('!I', len_bytes)[0]
        
        # Receive data
        data = recv_all(conn, length)
        inputs = np.frombuffer(data, dtype=np.float32)
        
        # INFERENCE
        # inputs = inputs.reshape(1, 1, -1) # Reshape for Akida
        # outputs = model.predict(inputs)
        # response_data = outputs.astype(np.float32).tobytes()
        
        # Mock Response (Echo)
        response_data = inputs.tobytes() 
        
        # Send back
        conn.sendall(struct.pack('!I', len(response_data)) + response_data)
        
    conn.close()

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

if __name__ == "__main__":
    run_server()
```
