"""
Tests for Time-Dilated Hardware-in-the-Loop (TD-HIL) pipeline.

Covers:
    1. RemoteAkidaPolicy timing instrumentation (last_info, latencies, reset)
    2. Latency flow through TensorControllerAdapter
    3. End-to-end: echo server -> RemoteAkidaPolicy -> adapter -> harness -> InferenceLatency
    4. akida/core/deploy.py CLI arg parsing

Uses a real threaded TCP echo server to test the full network round trip
without requiring Akida hardware.

"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

try:
    import tensorflow  # noqa: F401
except ImportError:
    tensorflow = None

from embark.benchmark.adapters import TensorControllerAdapter
from embark.benchmark.controllers.remote.akida_policy import RemoteAkidaPolicy
from embark.benchmark.harness import ClosedLoopHarness
from embark.benchmark.interfaces import ActionDict, ActionProcessor, SystemConfig
from embark.benchmark.metrics.accumulators import InferenceLatency, TrackingMAE
from embark.benchmark.processors import IdentityStateProcessor

# ---------------------------------------------------------------------------
# Echo server fixture
# ---------------------------------------------------------------------------


class _EchoServer:
    """
    Minimal TCP echo server that mimics the Akida inference server protocol.

    Receives length-prefixed float32 payloads and returns the first 2 floats (simulating
    a model with 2 outputs: v_d, v_q).

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
        # Connect to unblock accept()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect((self.host, self.port))
            sock.close()
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
                conn.settimeout(2.0)
                while not self._stop.is_set():
                    try:
                        len_bytes = self._recv_all(conn, 4)
                        if len_bytes is None:
                            break
                        payload_len = struct.unpack("!I", len_bytes)[0]
                        payload = self._recv_all(conn, payload_len)
                        if payload is None:
                            break

                        inputs = np.frombuffer(payload, dtype=np.float32)
                        # Return first 2 values (or zeros if input < 2)
                        output = np.zeros(2, dtype=np.float32)
                        output[: min(2, len(inputs))] = inputs[:2]
                        response = output.tobytes()
                        conn.sendall(struct.pack("!I", len(response)) + response)
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


@pytest.fixture
def echo_server():
    """Start an echo server on a random free port; tear it down after the test."""
    server = _EchoServer()
    server.start()
    yield server
    server.stop()


# ---------------------------------------------------------------------------
# Dummy task / processors for harness integration
# ---------------------------------------------------------------------------


@dataclass
class _DummyConfig:
    i_max: float = 10.0
    u_max: float = 10.0
    tau: float = 1e-4


class _DummyPhysics:
    def __init__(self) -> None:
        self.config = _DummyConfig()
        self._time = 0.0

    @property
    def state_keys(self):
        return {"i_d", "i_q", "time"}

    @property
    def action_keys(self):
        return {"v_d", "v_q"}

    def reset(self, seed=None):
        self._time = 0.0
        return {"i_d": 0.0, "i_q": 0.0, "time": 0.0}

    def step(self, action):
        self._time += self.config.tau
        return {
            "i_d": action.get("v_d", 0.0) * 0.1,
            "i_q": action.get("v_q", 0.0) * 0.1,
            "time": self._time,
        }, {}

    def close(self):
        pass


class _DummyTask:
    def __init__(self, max_steps: int = 5) -> None:
        self.physics_engine = _DummyPhysics()
        self.max_steps = max_steps
        self._step = 0

    @property
    def reference_keys(self):
        return {"i_d_ref", "i_q_ref"}

    def reset(self, seed=None):
        self._step = 0
        state = self.physics_engine.reset()
        return state, {"i_d_ref": 0.0, "i_q_ref": 1.0}

    def step(self, action):
        self._step += 1
        state, _ = self.physics_engine.step(action)
        done = self._step >= self.max_steps
        return state, {"i_d_ref": 0.0, "i_q_ref": 1.0}, done


@dataclass
class _SimpleActionProcessor(ActionProcessor):
    """Converts a 2-element tensor to v_d/v_q action dict."""

    u_max: float = 10.0

    def configure(self, physics_config: SystemConfig) -> None:
        self.u_max = getattr(physics_config, "u_max", self.u_max)

    def __call__(
        self, action: torch.Tensor, physics_config: SystemConfig
    ) -> ActionDict:
        flat = action.detach().cpu().flatten().tolist()
        return {"v_d": flat[0] * self.u_max, "v_q": flat[1] * self.u_max}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRemoteAkidaPolicyTiming:
    """Unit tests for the timing instrumentation in RemoteAkidaPolicy."""

    def test_forward_records_latency(self, echo_server):
        """Each forward() call should append one latency measurement."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )

        obs = torch.randn(1, 5, dtype=torch.float32)
        action = policy.forward(obs)

        assert action.shape == (2,)
        assert len(policy.latencies) == 1
        assert policy.latencies[0] > 0

    def test_last_info_returns_latency_dict(self, echo_server):
        """last_info should return a dict with inference_latency_s."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )

        assert policy.last_info is None  # No calls yet

        policy.forward(torch.randn(1, 5, dtype=torch.float32))

        info = policy.last_info
        assert info is not None
        assert "inference_latency_s" in info
        assert info["inference_latency_s"] > 0

    def test_multiple_forwards_accumulate(self, echo_server):
        """Multiple forward() calls should accumulate latencies."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )

        for _ in range(5):
            policy.forward(torch.randn(1, 5, dtype=torch.float32))

        assert len(policy.latencies) == 5
        assert all(lat > 0 for lat in policy.latencies)
        # last_info should reflect the most recent call
        assert policy.last_info["inference_latency_s"] == policy.latencies[-1]

    def test_reset_clears_latencies(self, echo_server):
        """Reset() should clear the latency history."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )

        policy.forward(torch.randn(1, 5, dtype=torch.float32))
        assert len(policy.latencies) == 1

        policy.reset()
        assert len(policy.latencies) == 0
        assert policy.last_info is None

    def test_latencies_is_copy(self, echo_server):
        """The latencies property should return a copy, not the internal list."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )

        policy.forward(torch.randn(1, 5, dtype=torch.float32))
        external = policy.latencies
        external.clear()
        assert len(policy.latencies) == 1  # Internal list unaffected


class TestLatencyThroughAdapter:
    """Test that last_info flows from RemoteAkidaPolicy through
    TensorControllerAdapter."""

    def test_adapter_exposes_last_info(self, echo_server):
        """TensorControllerAdapter.last_info should proxy
        RemoteAkidaPolicy.last_info."""
        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )
        state_proc = IdentityStateProcessor(
            state_keys=["i_d", "i_q"],
            reference_keys=["i_q_ref"],
        )
        action_proc = _SimpleActionProcessor()

        task = _DummyTask(max_steps=1)
        adapter = TensorControllerAdapter(
            controller=policy,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        adapter.configure(task.physics_engine.config, task)

        # Before any call, last_info should be None
        assert adapter.last_info is None

        # Run one step through the adapter
        state, reference = task.reset()
        adapter.reset()
        adapter(state, reference)

        # Now last_info should contain latency
        info = adapter.last_info
        assert info is not None
        assert "inference_latency_s" in info
        assert info["inference_latency_s"] > 0


class TestEndToEndHIL:
    """Full pipeline: echo server -> policy -> adapter -> harness -> latency metric."""

    def test_harness_collects_latency_metrics(self, echo_server):
        """ClosedLoopHarness should pass latency info to InferenceLatency
        accumulator."""
        n_steps = 10
        task = _DummyTask(max_steps=n_steps)

        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )
        state_proc = IdentityStateProcessor(
            state_keys=["i_d", "i_q"],
            reference_keys=["i_q_ref"],
        )
        action_proc = _SimpleActionProcessor()
        adapter = TensorControllerAdapter(
            controller=policy,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        adapter.configure(task.physics_engine.config, task)

        latency_metric = InferenceLatency()
        tracking_metric = TrackingMAE(tracked_keys=["i_q"])
        harness = ClosedLoopHarness(
            task=task,
            controller=adapter,
            metrics=[latency_metric, tracking_metric],
        )

        results = harness.run()

        # Latency metrics should be present and non-zero
        assert results["mean_latency_ms"] > 0
        assert results["p95_latency_ms"] > 0
        assert results["p99_latency_ms"] > 0
        assert results["max_latency_ms"] > 0
        assert results["total_inference_time_s"] > 0

        # Percentile ordering: mean <= p95 <= p99 <= max
        assert results["mean_latency_ms"] <= results["p95_latency_ms"] + 1e-9
        assert results["p95_latency_ms"] <= results["p99_latency_ms"] + 1e-9
        assert results["p99_latency_ms"] <= results["max_latency_ms"] + 1e-9

        # Steps should match
        assert results["steps"] == n_steps

        # Tracking metric should also be present (harness handles both)
        assert "mae_i_q" in results

    def test_latency_resets_between_episodes(self, echo_server):
        """Running the harness twice should give independent latency stats."""
        task = _DummyTask(max_steps=5)

        policy = RemoteAkidaPolicy(
            host=echo_server.host,
            port=echo_server.port,
            output_shape=(2,),
            timeout_s=5.0,
        )
        state_proc = IdentityStateProcessor(
            state_keys=["i_d", "i_q"],
            reference_keys=["i_q_ref"],
        )
        action_proc = _SimpleActionProcessor()
        adapter = TensorControllerAdapter(
            controller=policy,
            state_processor=state_proc,
            action_processor=action_proc,
        )
        adapter.configure(task.physics_engine.config, task)

        latency_metric = InferenceLatency()
        harness = ClosedLoopHarness(
            task=task, controller=adapter, metrics=[latency_metric]
        )

        results1 = harness.run()
        latencies1 = list(latency_metric._latencies)
        results2 = harness.run()
        latencies2 = list(latency_metric._latencies)

        # Both runs should produce valid latency data
        assert results1["mean_latency_ms"] > 0
        assert results2["mean_latency_ms"] > 0
        # Latency list should reset between runs (not accumulated)
        assert len(latencies1) == results1["steps"]
        assert len(latencies2) == results2["steps"]


@pytest.mark.skipif(tensorflow is None, reason="akida/tensorflow not installed")
class TestDeployArgParsing:
    """Test that evaluation/snn_keras/deploy.py CLI is importable and argument parser
    works."""

    def test_parser_accepts_required_args(self):
        from evaluation.akida.core.deploy import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--model-path",
                "model.h5",
                "--calibration-path",
                "calib.npy",
            ]
        )
        assert args.model_path == "model.h5"
        assert args.calibration_path == "calib.npy"
        assert args.output_path == "models/motor_control_akida.fbz"  # default

    def test_parser_accepts_all_options(self):
        from evaluation.akida.core.deploy import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--model-path",
                "model.keras",
                "--calibration-path",
                "calib.npy",
                "--output-path",
                "out.fbz",
                "--akida-version",
                "v1",
                "--weight-bits",
                "4",
                "--activation-bits",
                "4",
            ]
        )
        assert args.akida_version == "v1"
        assert args.weight_bits == 4
        assert args.activation_bits == 4
        assert args.output_path == "out.fbz"

    def test_deploy_functions_importable(self):
        """The public API functions should be importable."""
        from evaluation.akida.core.deploy import (
            convert_to_akida,
            deploy,
            main,
            quantize_model,
        )

        assert callable(quantize_model)
        assert callable(convert_to_akida)
        assert callable(deploy)
        assert callable(main)
