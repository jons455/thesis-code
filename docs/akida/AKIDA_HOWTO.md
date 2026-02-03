# How-To (Quick Start)

This section documents the concrete steps and file locations used by the current
implementation.

### 1) Start the inference server on the Pi

Server code lives in `akida/server/inference_server.py`.

Echo mode (connectivity test):

```
python akida/server/inference_server.py --host 0.0.0.0 --port 5000 --echo
```

Akida model mode:

```
python akida/server/inference_server.py --host 0.0.0.0 --port 5000 --model-path akida_model.fbz
```

### 2) Use the remote policy on the PC

Client policy lives in `embark/benchmark/controllers/remote/akida_policy.py` and
implements `TensorController`. It is used by the harness via
`TensorControllerAdapter`.

### 3) Verify connectivity locally (optional)

Run the local echo verification script:

```
python scripts/verify_hil_connectivity.py
```
