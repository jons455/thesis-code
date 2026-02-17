# Plan: Framework-Agnostic Benchmark Interface

> **⚠️ STATUS: PLANNING DOCUMENT — NOT YET IMPLEMENTED**  
> This document describes a planned refactoring to make the benchmark framework-agnostic.  
> **Current state:** The benchmark still uses PyTorch (`torch.Tensor`) in `TensorControllerAdapter` and processors.  
> **Last updated:** See verification checklist at bottom — items are unchecked, indicating this plan has not been implemented.

## Motivation

Right now, to benchmark an SNN (or any neural controller) with EMBARK, you
**must** use PyTorch.  The `TensorController` protocol expects `torch.Tensor`,
every processor creates or consumes `torch.Tensor`, and the only model wrapper
(`SNNControllerWrapper`) requires `torch.nn.Module`.

This gatekeeps the benchmark.  Researchers using **Norse**, **Lava/Loihi**,
**Brian2**, **NEST**, **JAX/Flax**, **snnTorch-on-JAX**, **Nengo**,
**pure-numpy SNN simulators**, or **custom C/FPGA implementations** cannot
participate without first porting their model to PyTorch — which defeats the
purpose of a benchmark that should measure the *controller*, not the framework.

The fix is surgical because **the core benchmark is already
framework-agnostic**.  Only a thin adapter layer (protocols, processors,
wrappers) is coupled to PyTorch.


---

## Current Architecture — What Touches PyTorch and What Doesn't

### Already framework-agnostic (NO changes needed)

| Component | Data types | File(s) |
|---|---|---|
| `Controller` protocol | `dict[str, float]` in/out | `interfaces/controller.py` |
| `ClosedLoopHarness` | Works with any `Controller` | `harness/closed_loop.py` |
| `BenchmarkSuite` | Works with any `Controller` | `harness/benchmark_suite.py` |
| `PMSMCurrentControlTask` | `dict[str, float]` | `tasks/pmsm_current_control.py` |
| All metrics (MAE, ITAE, etc.) | `dict[str, float]` | `metrics/accumulators/*.py` |
| All reference generators | `dict[str, float]` | `tasks/reference_generators.py` |
| `PMSMPhysicsEngine` | numpy internally, dict API | `physics/pmsm.py` |
| `PIControllerAgent` | Pure dict-based | `agents.py` |
| `BenchmarkConfig` | Primitive types only | `harness/benchmark_suite.py` |

### Coupled to PyTorch (changes needed)

| Component | How it uses PyTorch | Coupling level | File(s) |
|---|---|---|---|
| `TensorController` protocol | `torch.Tensor` in type hints (forward ref) | **Type hints only** | `interfaces/controller.py` |
| `StateProcessor` protocol | Returns `torch.Tensor` (forward ref) | **Type hints only** | `interfaces/processors.py` |
| `ActionProcessor` protocol | Accepts `torch.Tensor` (forward ref) | **Type hints only** | `interfaces/processors.py` |
| `TensorControllerAdapter` | Stores tensor refs, type hints | **Type hints only** | `adapters/tensor_adapter.py` |
| `RateSNNStateProcessor` | `torch.tensor(features, ...)` on last line | **One line** | `processors/rate_snn.py` |
| `RateSNNActionProcessor` | `action.detach().cpu().flatten().tolist()` | **One line** | `processors/rate_snn.py` |
| `IdentityStateProcessor` | `torch.tensor(values, ...)` on last line | **One line** | `processors/identity.py` |
| `IdentityActionProcessor` | `action.detach().cpu().flatten().tolist()` | **One line** | `processors/identity.py` |
| `PWMActionProcessor` | `action.detach().cpu().flatten().tolist()` | **One line** | `processors/pwm.py` |
| `SNNControllerWrapper` | `torch.nn.Module`, calls model directly | **Runtime import** | `controllers/neural/snn_wrapper.py` |
| `RemoteAkidaPolicy` | Converts torch↔numpy for TCP | **Runtime import** | `controllers/remote/akida_policy.py` |
| `neurobench_factory.py` | `isinstance(model, torch.nn.Module)` | **One check** | `metrics/neurobench_factory.py` |
| `SNNControllerAgent` | Heavy torch usage (devices, tensors) | **Runtime import** | `agents.py` |

### Critical insight

The processors compute everything in **plain Python floats** (EMA filters,
derivatives, normalization, clipping).  They only touch PyTorch at the very
boundary:

- **State processors:** Build a `list[float]`, then `torch.tensor(list)` on the last line.
- **Action processors:** Call `action.detach().cpu().flatten().tolist()` on the first line, then work in Python.

There is **no autograd, no GPU computation, no backpropagation** — this is
pure inference.  The tensors are just a data shuttle between the processors
and the neural controller's `forward()` method.


---

## Design Decision: numpy as the Lingua Franca

**Why numpy?**

1. **Already a dependency** — the physics engine uses it internally.
2. **Universal interop** — every ML framework has zero-copy conversion to/from numpy:
   - PyTorch: `torch.from_numpy(arr)` / `tensor.numpy()`
   - JAX: `jnp.array(arr)` / `np.asarray(jax_arr)`
   - TensorFlow: `tf.constant(arr)` / `arr.numpy()`
   - Lava, Brian2, NEST, Nengo: all numpy-native
   - C/FPGA: numpy arrays can be created from raw buffers
3. **No new dependency** — numpy is lighter than torch and already required.
4. **Consistent with the physics layer** — `PMSMPhysicsEngine` already uses
   numpy internally before converting to dicts.

**Alternative considered: `Any` type / duck typing.**  
Rejected because it provides no contract.  `np.ndarray` is concrete: you
know the shape, dtype, and can validate inputs.

### Array Contract (explicit invariants)

To avoid subtle cross-framework bugs, the benchmark should define a strict
array boundary contract:

1. **Dtype:** `np.float32` at controller boundaries unless explicitly
   documented otherwise.
2. **Layout:** C-contiguous arrays preferred for predictable interop
   (`np.asarray(x, dtype=np.float32, order="C")`).
3. **Shape convention:** support both:
   - Single sample: `(features,)`
   - Batched input (optional wrapper support): `(batch, features)`
4. **Output convention:** model output must be convertible to an action array
   that existing action processors can flatten deterministically.
5. **No autograd semantics at boundary:** boundary arrays are inference-only.

This keeps the core deterministic and framework-neutral while still allowing
framework-specific internals inside wrappers.


---

## Do We Need Wrappers for Every Framework?

**No.**  That would be an infinite maintenance burden and we'd always be
behind.

We provide:

1. **The protocol** — a 4-method interface where `forward()` takes and
   returns `np.ndarray`.  This is the contract.  Any framework can satisfy
   it.

2. **Two reference wrappers** that we ship and maintain:
   - `TorchModelWrapper` — for PyTorch / snntorch / Norse models
   - `NumpyModelWrapper` — zero-overhead passthrough for numpy-native models

3. **Documentation + examples** showing how to write your own wrapper.
   A wrapper is ~10-15 lines of code.

Users of other frameworks write their own thin adapter.  Here's how trivial
it is for each framework:

```python
# JAX — ~8 lines
class JaxWrapper:
    def __init__(self, params, apply_fn):
        self.params, self.apply_fn = params, apply_fn
    def forward(self, obs: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        out = self.apply_fn(self.params, jnp.array(obs))
        return np.asarray(out)
    def reset(self): pass
    def get_state(self): return {}
    def set_state(self, s): pass

# Lava — ~10 lines
class LavaWrapper:
    def __init__(self, process):
        self.process = process
    def forward(self, obs: np.ndarray) -> np.ndarray:
        self.process.input_port.send(obs)
        return self.process.output_port.recv()
    def reset(self): self.process.reset()
    def get_state(self): return {}
    def set_state(self, s): pass

# Brian2 — ~8 lines
class Brian2Wrapper:
    def __init__(self, network, input_group, output_group):
        self.net, self.inp, self.out = network, input_group, output_group
    def forward(self, obs: np.ndarray) -> np.ndarray:
        self.inp.rates = obs * brian2.Hz  # map to firing rates
        self.net.run(1 * brian2.ms)
        return np.array(self.out.v)  # read membrane voltages
    def reset(self): self.net.restore()
    def get_state(self): return {}
    def set_state(self, s): pass

# ONNX Runtime — ~8 lines
class OnnxWrapper:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
    def forward(self, obs: np.ndarray) -> np.ndarray:
        inputs = {self.session.get_inputs()[0].name: obs.astype(np.float32)}
        return self.session.run(None, inputs)[0]
    def reset(self): pass
    def get_state(self): return {}
    def set_state(self, s): pass

# Pure numpy SNN — 0 lines (it already returns np.ndarray)
```

The key insight: **the wrapper is the user's responsibility** because only
they know their framework's API.  Our job is to make the contract so simple
that writing a wrapper is trivial.


---

## Implementation Plan — Step by Step

Each step below is self-contained and can be verified independently.
**Backward compatibility is preserved throughout** — existing PyTorch code
continues to work with the new `TorchModelWrapper`.


### Step 1: Generalize the Protocols

**Files to modify:**
- `embark/benchmark/interfaces/controller.py`
- `embark/benchmark/interfaces/processors.py`

**What to do:**

Replace `torch.Tensor` type hints with `np.ndarray` in the three protocols:

```python
# interfaces/controller.py — BEFORE
import torch  # in TYPE_CHECKING block

class TensorController(Protocol):
    def forward(self, observation: "torch.Tensor") -> "torch.Tensor": ...

# interfaces/controller.py — AFTER
import numpy as np

class TensorController(Protocol):
    def forward(self, observation: np.ndarray) -> np.ndarray: ...
```

```python
# interfaces/processors.py — BEFORE
class StateProcessor(Protocol):
    def __call__(self, state: StateDict, reference: ReferenceDict) -> "torch.Tensor": ...

class ActionProcessor(Protocol):
    def __call__(self, action: "torch.Tensor", physics_config: SystemConfig) -> ActionDict: ...

# interfaces/processors.py — AFTER
class StateProcessor(Protocol):
    def __call__(self, state: StateDict, reference: ReferenceDict) -> np.ndarray: ...

class ActionProcessor(Protocol):
    def __call__(self, action: np.ndarray, physics_config: SystemConfig) -> ActionDict: ...
```

Remove the `if TYPE_CHECKING: import torch` blocks from both files.  Add
`import numpy as np` instead (top-level, not conditional).

**Verification:** Run `python -c "from embark.benchmark.interfaces import TensorController, StateProcessor, ActionProcessor"` — should import cleanly without torch installed.


### Step 2: Update All Processor Implementations

**Files to modify:**
- `embark/benchmark/processors/rate_snn.py`
- `embark/benchmark/processors/identity.py`
- `embark/benchmark/processors/pwm.py`

**What to do — state processors (dict → array):**

Each state processor builds a Python `list[float]` and wraps it at the end.
Change the final line:

```python
# BEFORE
import torch
return torch.tensor(features, dtype=torch.float32)

# AFTER
import numpy as np
return np.array(features, dtype=np.float32)
```

**What to do — action processors (array → dict):**

Each action processor extracts a Python list on the first line.  Change it:

```python
# BEFORE
action_list = action.detach().cpu().flatten().tolist()

# AFTER
action_list = np.asarray(action).flatten().tolist()
```

Using `np.asarray()` is defensive — it handles both `np.ndarray` (no-op)
and any array-like input.

**Important:** Update all type annotations in these files:
- Method signatures: `torch.Tensor` → `np.ndarray`
- `import torch` → `import numpy as np` (most files already import numpy
  for the physics engine; check and deduplicate)

**What to do — `pwm.py`:**

`PWMConverter` already uses numpy (`np.clip`, `np.sign`).  Only the
`PWMActionProcessor.__call__` method needs the same `detach().cpu()` →
`np.asarray()` change.

**Verification:** Write a quick test that creates a `RateSNNStateProcessor`,
calls it with a state dict, and asserts the return type is `np.ndarray`.
Same for action processors.


### Step 3: Update the TensorControllerAdapter

**File to modify:**
- `embark/benchmark/adapters/tensor_adapter.py`

**What to do:**

1. Remove `if TYPE_CHECKING: import torch` block.
2. Add `import numpy as np` (or `from numpy.typing import NDArray`).
3. Change stored field types:
   ```python
   # BEFORE
   _last_observation: "torch.Tensor | None" = field(default=None, repr=False)
   _last_action_tensor: "torch.Tensor | None" = field(default=None, repr=False)

   # AFTER
   _last_observation: np.ndarray | None = field(default=None, repr=False)
   _last_action_tensor: np.ndarray | None = field(default=None, repr=False)
   ```
4. Update property return type annotations:
   ```python
   # BEFORE
   @property
   def last_observation(self) -> "torch.Tensor | None": ...

   # AFTER
   @property
   def last_observation(self) -> np.ndarray | None: ...
   ```
5. The `__call__` method body **does not change** — it just passes objects
   between processor and controller.  The types flowing through are now
   `np.ndarray` instead of `torch.Tensor` but the orchestration is identical.

**Verification:** The adapter should work without torch installed (assuming
the controller and processors also don't need torch).


### Step 4: Create Framework-Specific Model Wrappers

**Files to create:**
- `embark/benchmark/controllers/neural/torch_wrapper.py` (new)
- `embark/benchmark/controllers/neural/numpy_wrapper.py` (new)

**Files to modify:**
- `embark/benchmark/controllers/neural/snn_wrapper.py` (keep for backward compat, deprecate)
- `embark/benchmark/controllers/neural/__init__.py`
- `embark/benchmark/controllers/__init__.py`
- `embark/benchmark/__init__.py`

#### 4a. `torch_wrapper.py` — PyTorch model wrapper

```python
"""Wrapper for PyTorch models to match the numpy-based TensorController protocol."""
from __future__ import annotations
from typing import Any
import numpy as np

class TorchModelWrapper:
    """
    Wraps any torch.nn.Module into a numpy-based TensorController.

    Handles torch.Tensor ↔ np.ndarray conversion, torch.no_grad() context,
    device management, and tuple-return (action, info) unpacking.

    This is the recommended wrapper for PyTorch / snntorch / Norse models.

    Args:
        model: A torch.nn.Module.  Must accept a tensor and return either
               a single action tensor or a (action_tensor, info_dict) tuple.
        device: Device string ("cpu", "cuda", "cuda:0", etc.).

    Example::

        import torch.nn as nn
        model = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 2))
        wrapper = TorchModelWrapper(model, device="cpu")
        action = wrapper.forward(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        # action is np.ndarray

    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        import torch
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_info = None
        if hasattr(self.model, "reset"):
            self.model.reset()

    def forward(self, observation: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(observation, dtype=np.float32))
            t = t.to(self.device)
            output = self.model(t)

            if isinstance(output, tuple) and len(output) >= 2:
                action, info = output[0], output[1]
                if isinstance(info, dict):
                    self.last_info = info
            else:
                action = output

            return action.detach().cpu().numpy()

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass
```

**Key details:**
- `import torch` is inside `__init__` and `forward`, not at module level.
  This means the *module* can be imported without torch installed (for
  discoverability), but *instantiation* requires torch.
- Handles tuple returns (action, info) just like the old `SNNControllerWrapper`.
- `model` type hint is `Any` (not `torch.nn.Module`) to avoid importing
  torch at module level.  The actual type check happens implicitly when
  `model.to(device)` is called.
- Keep model in inference mode (`model.eval()`) and always execute in
  `torch.no_grad()` to enforce benchmark-inference semantics.
- If torch is missing at instantiation time, raise a clear install hint:
  `ImportError("TorchModelWrapper requires PyTorch. Install with: pip install torch")`.
- Normalize incoming observations defensively:
  `obs = np.asarray(observation, dtype=np.float32, order="C")`.

#### 4b. `numpy_wrapper.py` — Zero-overhead passthrough

```python
"""Wrapper for numpy-native models (zero overhead passthrough)."""
from __future__ import annotations
from typing import Any, Protocol
import numpy as np


class NumpyCallable(Protocol):
    """Any callable that takes np.ndarray and returns np.ndarray."""
    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


class NumpyModelWrapper:
    """
    Wraps any callable(np.ndarray) -> np.ndarray as a TensorController.

    Zero conversion overhead.  Use this for:
    - Pure numpy SNN implementations
    - Brian2 / NEST / Nengo models that output numpy
    - ONNX Runtime sessions (wrap the run() call)
    - C extensions that return numpy arrays
    - Any framework that can produce np.ndarray output

    Args:
        model_fn: Any callable that takes an np.ndarray observation and
                  returns an np.ndarray action.
        reset_fn: Optional callable to reset model state.

    Example::

        def my_snn(obs: np.ndarray) -> np.ndarray:
            # your custom SNN logic here
            return np.tanh(obs @ weights)

        wrapper = NumpyModelWrapper(my_snn)
        action = wrapper.forward(np.array([1.0, 2.0, 3.0]))

    """

    def __init__(
        self,
        model_fn: NumpyCallable,
        reset_fn: Any = None,
    ) -> None:
        self.model_fn = model_fn
        self._reset_fn = reset_fn
        self.last_info: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_info = None
        if self._reset_fn is not None:
            self._reset_fn()

    def forward(self, observation: np.ndarray) -> np.ndarray:
        result = self.model_fn(observation)
        if isinstance(result, tuple) and len(result) >= 2:
            action, info = result[0], result[1]
            if isinstance(info, dict):
                self.last_info = info
            return np.asarray(action, dtype=np.float32)
        return np.asarray(result, dtype=np.float32)

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        pass
```

#### 4c. Deprecate old `SNNControllerWrapper`

Don't delete `snn_wrapper.py` — add a deprecation warning that points to
`TorchModelWrapper`:

```python
# At the top of SNNControllerWrapper.__init__:
import warnings
warnings.warn(
    "SNNControllerWrapper is deprecated, use TorchModelWrapper instead. "
    "It has the same functionality but uses numpy arrays at the interface, "
    "making the benchmark framework-agnostic.",
    DeprecationWarning,
    stacklevel=2,
)
```

Then make the old wrapper delegate to `TorchModelWrapper` internally, or
simply keep it working as-is (both the old torch-based path and the new
numpy-based path work since `TensorControllerAdapter` doesn't care about
the internal array type — it just passes objects through).

**Simplest backward-compat approach:** Make `SNNControllerWrapper` a thin
subclass of `TorchModelWrapper` that issues the deprecation warning.

#### 4d. Update `__init__.py` exports

Add `TorchModelWrapper` and `NumpyModelWrapper` to the public API:

```python
# embark/benchmark/controllers/neural/__init__.py
from .snn_wrapper import SNNControllerWrapper    # deprecated, keep for compat
from .torch_wrapper import TorchModelWrapper
from .numpy_wrapper import NumpyModelWrapper

# embark/benchmark/__init__.py — add to imports and __all__:
from .controllers import TorchModelWrapper, NumpyModelWrapper
```

#### 4e. Add a shared wrapper protocol + validator utility

**Files to create:**
- `embark/benchmark/interfaces/model_wrapper.py` (new)
- `embark/benchmark/validation/wrapper_validation.py` (new)

Add a protocol that formalizes what any wrapper must implement:

```python
class ModelWrapper(Protocol):
    def forward(self, observation: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
```

Then add `validate_model_wrapper(wrapper)` to provide a friendly runtime error
for missing methods or invalid `forward()` outputs.  Use this in docs/examples
and optionally in adapter construction checks.


### Step 5: Update `RemoteAkidaPolicy`

**File to modify:**
- `embark/benchmark/controllers/remote/akida_policy.py`

**What to do:**

This file already does torch↔numpy conversion internally!  It converts the
torch observation to numpy for TCP serialization, then converts the numpy
response back to torch.  After this refactoring, it becomes simpler because
the protocol is already numpy:

```python
# BEFORE
def forward(self, observation: torch.Tensor) -> torch.Tensor:
    obs_np = observation.detach().cpu().numpy().astype(np.float32, copy=False)
    ...
    return torch.from_numpy(action_np)

# AFTER
def forward(self, observation: np.ndarray) -> np.ndarray:
    obs_np = np.asarray(observation, dtype=np.float32)
    ...
    return action_np  # already numpy!
```

Remove `import torch` from this file entirely.  It no longer needs it.


### Step 6: Update `neurobench_factory.py`

**File to modify:**
- `embark/benchmark/metrics/neurobench_factory.py`

**What to do:**

The `_controller_has_model()` function does:
```python
return isinstance(model, torch.nn.Module)
```

Change to a duck-type check that doesn't require torch:

```python
def _controller_has_model(controller: Any | None) -> bool:
    """True when the controller exposes a model object for metric hooks."""
    if controller is None:
        return False
    model = getattr(controller, "model", None)
    if model is None:
        return False
    return hasattr(model, "forward") or callable(model)
```

If NeuroBench adapters specifically need a `torch.nn.Module`, that check
should live inside the NeuroBench adapter code (which already conditionally
imports torch), not in the factory.

Remove `import torch` from this file.


### Step 7: Update `SNNControllerAgent` and `SNNControllerTorchAgent`

**File to modify:**
- `embark/benchmark/agents.py`

**What to do:**

These are the legacy full-agent classes that bundle model + processing.
They heavily use torch internally.  Two options:

**Option A (recommended):** Keep them as-is.  They implement the
`Controller` protocol (dict in, dict out), so the harness doesn't care
what they use internally.  They're self-contained PyTorch agents.  Mark
them as "PyTorch-specific" in their docstrings.

**Option B:** Refactor to use `TorchModelWrapper` internally.  More work,
higher risk, low benefit since these are legacy classes.

**Go with Option A.**  No changes needed to `agents.py`.  Just update the
docstring to note they require PyTorch.


### Step 8: Update Tests

**Files to modify:**
- Any test that directly creates torch tensors and passes them to processors
- Any test that checks `isinstance(result, torch.Tensor)`

**What to do:**

Search all test files for:
- `torch.tensor(` → replace with `np.array(`
- `isinstance(result, torch.Tensor)` → `isinstance(result, np.ndarray)`
- `action.detach()` → `np.asarray(action)`

The most affected tests will be:
- `tests/test_processors.py`
- `tests/test_rate_snn_processors.py`
- `tests/test_metric_adapters.py`
- `tests/test_v10_end_to_end.py` (SNN integration tests)

Add a dedicated conformance test file for wrapper contracts:
- `tests/test_model_wrapper_conformance.py`

Minimum conformance cases:
- Wrapper exposes required methods (`forward/reset/get_state/set_state`)
- `forward()` returns `np.ndarray`
- Output dtype is `float32` (or explicitly normalized to it)
- Single-sample input path works (`shape == (features,)`)
- Batched input path behavior is explicit (supported or clear error)
- Non-contiguous input is handled (`np.ascontiguousarray` or equivalent)

**Verification:** All tests pass.  Run the full suite:
```
python -m pytest tests/ -v
```


### Step 9: Update `benchmark_example.py`

**File to modify:**
- `examples/benchmark_example.py`

**What to do:**

Section 3 (SNN quick benchmark) currently uses `SNNControllerWrapper`.
Update it to show `TorchModelWrapper` and add a comment showing the
numpy wrapper alternative:

```python
# Wrap the PyTorch model for the numpy-based benchmark interface
wrapped = TorchModelWrapper(model=model, device=device)

# If your model is numpy-native instead of PyTorch, use:
# wrapped = NumpyModelWrapper(model_fn=my_numpy_snn)
```


### Step 10: Update Documentation

**Files to modify:**
- `docs/benchmark/BENCHMARK_USER_GUIDE.md` — add "Using non-PyTorch models" section
- `docs/benchmark/RATE_SNN_BENCHMARK_INTERFACE.md` — update type references
- `docs/reference/FUTURE_WORK.md` — mark Section 2 (Multi-Framework Support) as
  "done" for the core numpy adapter; update remaining items
- `README.md` — mention framework-agnostic support in features list
- `docs/MIGRATION_V2.md` (new) — migration guide for existing users

**Examples to add (complete workflows, not templates only):**
- `examples/frameworks/jax_lif_network.py`
- `examples/frameworks/onnx_exported_model.py`
- `examples/frameworks/brian2_coba_network.py`

**New content to add** (to user guide or a new "Framework Integration" page):

1. The 3-level integration diagram:
   ```
   Your Model (any framework)
       ↓ writes a thin wrapper (~10 lines)
   TensorController protocol (numpy arrays)
       ↓ handled by existing processors + adapter
   Controller protocol (Python dicts)
       ↓ handled by harness
   BenchmarkSuite (results)
   ```

2. Copy-paste wrapper templates for PyTorch, JAX, ONNX, Lava, Brian2
   (the examples from the "Do We Need Wrappers for Every Framework?"
   section above).

3. Processor reuse note: The existing `RateSNNStateProcessor` and
   `RateSNNActionProcessor` work identically regardless of what framework
   the model uses — they convert dicts to numpy and numpy to dicts.  The
   model wrapper is the only framework-specific piece.


---

## File Change Summary

| File | Change type | Risk |
|---|---|---|
| `interfaces/controller.py` | Type hints: `torch.Tensor` → `np.ndarray` | Very low |
| `interfaces/processors.py` | Type hints: `torch.Tensor` → `np.ndarray` | Very low |
| `adapters/tensor_adapter.py` | Type hints only | Very low |
| `processors/rate_snn.py` | `torch.tensor()` → `np.array()`, `detach().cpu()` → `np.asarray()` | Very low |
| `processors/identity.py` | Same mechanical change | Very low |
| `processors/pwm.py` | Same mechanical change | Very low |
| `controllers/neural/torch_wrapper.py` | **New file** | Low |
| `controllers/neural/numpy_wrapper.py` | **New file** | Low |
| `controllers/neural/snn_wrapper.py` | Add deprecation warning | Very low |
| `controllers/remote/akida_policy.py` | Remove torch conversion, simplify | Low |
| `metrics/neurobench_factory.py` | Remove `isinstance(model, torch.nn.Module)` | Low |
| `controllers/neural/__init__.py` | Add exports | Very low |
| `controllers/__init__.py` | Add exports | Very low |
| `__init__.py` | Add exports | Very low |
| `agents.py` | Docstring update only | None |
| Test files | `torch.Tensor` → `np.ndarray` assertions | Low |
| `examples/benchmark_example.py` | Use new wrapper names | Very low |
| Docs (multiple) | New sections, updated references | None |


---

## Backward Compatibility Strategy

1. **`SNNControllerWrapper`** — kept, with deprecation warning.  Existing
   code that uses it continues to work.

2. **`SNNControllerAgent` / `SNNControllerTorchAgent`** — unchanged.  They
   implement `Controller` (dict-based), so the harness doesn't care.

3. **Processor return types** — changed from `torch.Tensor` to `np.ndarray`.
   This is a **breaking change** for anyone who directly calls a processor
   and then does torch-specific operations on the result (e.g.,
   `result.requires_grad_()`).  However:
   - Processors are inference-only; nobody should be doing autograd on them.
   - The adapter (`TensorControllerAdapter`) is the only consumer, and it
     just passes the object through.
   - If anyone truly needs a torch tensor, `torch.from_numpy(result)` is
     a zero-copy operation they can add in their wrapper.

4. **Import paths** — all existing import paths continue to work.  New
   wrappers are additive.

5. **Naming compatibility** — keep `TensorController` for now to avoid churn,
   and optionally add a later alias/deprecation path (`ArrayController`) in a
   future release if clearer naming is desired.


### Optional rollout safety switch (time-boxed)

If maintainers want an emergency escape hatch during rollout, add a temporary
`EMBARK_LEGACY_TORCH=true` mode for processors.  This should be:
- Off by default
- Documented as temporary
- Removed after 1-2 release cycles

Do not let this become permanent API surface; the target steady state remains
numpy at the interface boundary.


---

## What This Enables

After this change, the benchmark supports:

| Framework | How to integrate | Wrapper needed? |
|---|---|---|
| PyTorch / snntorch | `TorchModelWrapper(model)` | Ships with embark |
| Norse (PyTorch-based) | `TorchModelWrapper(model)` | Ships with embark |
| JAX / Flax | User writes ~8-line wrapper | User-provided |
| TensorFlow / Keras | User writes ~8-line wrapper | User-provided |
| Lava (Intel Loihi) | User writes ~10-line wrapper | User-provided |
| Brian2 | User writes ~8-line wrapper | User-provided |
| NEST | User writes ~8-line wrapper | User-provided |
| Nengo | User writes ~8-line wrapper | User-provided |
| ONNX Runtime | User writes ~8-line wrapper | User-provided |
| Pure numpy | `NumpyModelWrapper(fn)` | Ships with embark |
| C / FPGA (via ctypes) | `NumpyModelWrapper(fn)` | Ships with embark |
| Remote hardware (TCP) | `RemoteAkidaPolicy` (updated) | Ships with embark |

The key principle: **the benchmark contract is `np.ndarray` → `np.ndarray`**.
Any framework that can produce a numpy array can participate.  We don't
gatekeep — we provide the protocol and two reference implementations.
Users bring their own models.


---

## Verification Checklist

After implementation, verify each item:

- [ ] `python -c "from embark.benchmark import TorchModelWrapper, NumpyModelWrapper"` works
- [ ] `python -c "from embark.benchmark import BenchmarkSuite"` works **without torch installed**
- [ ] `python -m pytest tests/` passes
- [ ] `python examples/benchmark_example.py --section 1` (PI baseline, no torch needed)
- [ ] `python examples/benchmark_example.py --section 2` (PI full suite, no torch needed)
- [ ] `python examples/benchmark_example.py --section 3` (SNN, needs torch — uses TorchModelWrapper)
- [ ] A pure-numpy "model" can run through the full benchmark via NumpyModelWrapper
- [ ] Wrapper conformance tests pass for shipped wrappers and one user-style wrapper
- [ ] Non-contiguous input arrays are normalized at wrapper boundary
- [ ] Single-sample and batched-shape behavior is documented and tested
- [ ] Old code using `SNNControllerWrapper` still works (with deprecation warning)
- [ ] `RemoteAkidaPolicy` works without torch import at module level
