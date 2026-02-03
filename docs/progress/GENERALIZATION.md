# Universal Control Framework Architecture

**Goal:** Create a domain-agnostic benchmarking framework where **Physics Engines** (Plants) and **Controllers** (Agents) are fully interchangeable plugins. The framework orchestrates the interaction, enforces safety, and computes standardized metrics regardless of the underlying domain (e.g., Motors, Gimbals, Chemical Processes).

---

## I. High-Level Architecture: "The Universal Loop"

The core of the system is the `ClosedLoopHarness`. It acts as a universal adapter, connecting any *Controller* to any *Task* via standardized protocols.

### The Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ClosedLoopHarness                              │
│                                                                         │
│   1. Get State      2. Normalize      3. Compute      4. Denormalize    │
│   (Physics Units)    (Tensor)          (Tensor)        (Physics Units)  │
│                                                                         │
│      ┌────┐          ┌────┐             ┌────┐             ┌────┐       │
│      │Task│─────────▶│Proc│────────────▶│Ctrl│────────────▶│Proc│       │
│      └────┘          └────┘             └────┘             └────┘       │
│         ▲                                                      │        │
│         └──────────────────────────────────────────────────────┘        │
│                            5. Step Physics                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
      │                                                         │
      ▼                                                         ▼
┌────────────┐                                           ┌─────────────┐
│Safety Guard│ (Checks Limits)                           │Metric Engine│
└────────────┘                                           └─────────────┘

```

---

## II. Core Components (The Plugins)

### 1. The Physics Engine (The Plant)

**Responsibility:** Simulates the physical world. It knows *nothing* about control or tasks.

* **Input:** Physical Action (e.g., Volts, Force, Heat).
* **Output:** Physical State (e.g., Amps, Angle, Temperature).
* **Contract:** Must implement `PhysicsEngine` protocol.

**Examples:**

* `PMSMPhysicsEngine`: Wraps GEM simulator.
* `GimbalPhysicsEngine` (Future): Simulates 2-axis mechanical gimbal.
* `FOPDTPhysicsEngine` (Future): Simulates First-Order Plus Dead Time (chemical process).

### 2. The Task (The Objective)

**Responsibility:** Defines "What implies success?" It combines a Physics Engine with a Reference Generator.

* **Responsibility:** Generates reference trajectories (targets).
* **Responsibility:** Defines Safety Limits (e.g., "Stop if Temp > 100°C").
* **Contract:** Must implement `ClosedLoopTask` protocol.

**Examples:**

* `PMSMCurrentControlTask`: Tracks  current.
* `GimbalPointingTask`: Tracks azimuth/elevation angles.

### 3. The Controller (The Agent)

**Responsibility:** Computes the control signal.

* **Neural (Tensor-based):** SNNs, ANNs. Wrapped in `TensorControllerAdapter`.
* **Classical (Dict-based):** PIDs, MPCs. Implements `Controller` directly.

---

## III. Standardization Strategy

To ensure a PID on a Gimbal is evaluated exactly like an SNN on a PMSM, we use **Semantic Abstraction**.

### 1. Metric Registry (The Rosetta Stone)

Instead of hardcoding "current error", we calculate "tracking error" and map it via config.

* **Generic Metric:** `TrackingRMSE` (Calculates ).
* **PMSM Config:** Maps `y`  `i_q`, `y_{ref}`  `i_q_{ref}`.
* **Gimbal Config:** Maps `y`  `theta`, `y_{ref}`  `theta_{ref}`.

### 2. Universal Processors

We standardized on `Processors` to handle the mismatch between "Physical Reality" and "Neural Inputs".

* **`StateProcessor`**: Converts `{i_q: 2.0A}`  `Tensor([0.5])`.
* **`ActionProcessor`**: Converts `Tensor([1.0])`  `{v_q: 24.0V}`.

---

## IV. Directory Structure

This structure separates the *Framework* (Core) from the *Implementations* (Plugins).

```text
embark/
├── benchmark/
│   ├── core/                  # THE FRAMEWORK (Domain Agnostic)
│   │   ├── harness.py         # ClosedLoopHarness
│   │   ├── interfaces.py      # Protocols (Physics, Task, Controller)
│   │   └── metrics.py         # Generic Accumulators (RMSE, SyOps)
│   │
│   ├── adapters/              # THE GLUE
│   │   ├── tensor_adapter.py  # Connects Neural Nets to Harness
│   │   └── safety.py          # Universal Safety Guard
│   │
│   ├── implementations/       # THE PLUGINS (Domain Specific)
│   │   ├── physics/
│   │   │   ├── pmsm_gem.py    # GEM Wrapper
│   │   │   └── fopdt.py       # Future Chemical Plant
│   │   │
│   │   ├── tasks/
│   │   │   ├── pmsm_tasks.py  # Current Control
│   │   │   └── generic_tasks.py # Step Response Generator
│   │   │
│   │   └── controllers/
│   │       ├── baselines/     # PI, PID
│   │       └── neural/        # SNN Wrappers
│   │
│   └── configs/               # THE MAPS
│       └── pmsm_default.py    # Maps 'Tracking' -> 'i_q'

```

---

## V. Example: Swapping the World

The power of this architecture is that changing the domain requires changing *only* the Task configuration.

### Scenario A: PMSM Control (Current)

```python
# 1. Define World
physics = PMSMPhysicsEngine(config=GEMConfig(...))
task = PMSMTask(physics, ref_gen=StepGen(target=2.0))

# 2. Define Brain
controller = SNNController(...)

# 3. Run
harness = ClosedLoopHarness(task, controller)
harness.run()

```

### Scenario B: Gimbal Control (Future)

```python
# 1. Define World
physics = GimbalPhysicsEngine(config=GimbalConfig(inertia=...))
task = GimbalTask(physics, ref_gen=SinusoidGen(amp=45_deg))

# 2. Define Brain (SAME CONTROLLER CLASS!)
# We just retrain weights; the code interface is identical.
controller = SNNController(...) 

# 3. Run (SAME HARNESS!)
harness = ClosedLoopHarness(task, controller)
harness.run()

```

---

## VI. Migration Plan (Future Work)

This section outlines the concrete steps required to fully decouple the framework from the PMSM domain, enabling support for other systems (e.g., Gimbals, Chemical Plants).

### Step 1: Decouple System Configuration
**Current Issue:** `SystemConfig` protocol includes PMSM-specific fields (`i_max`, `u_max`, `tau`).
**Fix:** Split configuration into a base protocol and domain-specific dataclasses.

1.  **Define Base Protocol:**
    ```python
    class SystemConfig(Protocol):
        """Marker protocol for physical configurations."""
        pass
    ```
2.  **Define PMSM Config:**
    ```python
    @dataclass
    class PMSMConfig(SystemConfig):
        i_max: float
        u_max: float
        tau: float
        # ... other motor params
    ```
3.  **Update Usage:** Update `PMSMPhysicsEngine` to expect `PMSMConfig`.

### Step 2: Generalize Processors
**Current Issue:** `MinMaxProcessor` contains hardcoded `if key.startswith("i_")` logic for bounds.
**Fix:** Remove magic logic and require explicit bounds configuration.

1.  **Refactor `MinMaxProcessor`:**
    - Remove auto-detection in `configure()`.
    - Require `bounds` dictionary to be fully populated by the `Task`.
2.  **Update `PMSMTask`:**
    - The Task is the only component that knows the domain *and* the physics limits.
    - It should construct the `bounds` dict (e.g., `{"i_q": (-i_max, i_max)}`) and pass it to the processor.

### Step 3: Domain-Agnostic Metric Registry
**Current Issue:** Metrics are manually instantiated with hardcoded keys (e.g., `TrackingRMSE(tracked_keys=["i_d", "i_q"])`).
**Fix:** Introduce a registry to map semantic names to physical keys.

1.  **Create Registry:**
    ```python
    METRIC_KEY_MAP = {
        "tracking": ["i_d", "i_q"],  # PMSM default
        # "tracking": ["azimuth", "elevation"]  # Gimbal future
    }
    ```
2.  **Update Harness/Config:** Allow selecting metrics by semantic name (`"tracking"`) which resolves to specific keys at runtime based on the active task.

### Step 4: Directory Restructuring (Optional)
**Current Issue:** Implementations are mixed with core interfaces.
**Fix:** strict separation.

```text
embark/benchmark/
├── core/               # Interfaces, Harness, Generic Metrics
├── implementations/    # Domain-specific plugins
│   ├── pmsm/           # PMSM Physics, Tasks, Configs
│   └── gimbal/         # (Future) Gimbal components
└── adapters/           # TensorAdapters, SafetyGuards
```

### Estimated Effort
- **Step 1 & 2 (Core Decoupling):** ~2-4 hours. High value for purity.
- **Step 3 (Metric Registry):** ~2 hours. Useful mostly for multi-domain batch benchmarking.
- **Step 4 (Restructure):** ~2 hours. Purely organizational.
