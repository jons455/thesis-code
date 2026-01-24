# Methodology & Implementation Details

This document details the specific implementation choices, validation strategies, and the roadmap for refining the benchmark architecture. It serves as the bridge between the high-level design (`ARCHITECTURE.md`) and the actual code.

## 1. Implementation Reality

While `ARCHITECTURE.md` describes the logical layers, the actual implementation contains specific optimizations and adaptations for Neuromorphic Control.

### 1.1 Biological Integration (Implicit Integrator)
The initial design proposed a "Hybrid" architecture with an external mathematical integrator ($u_{acc} += kick$).
The **current implementation** (`snn/models.py` -> `SimpleSNNController`) achieves this *biologically* using **Slow-Leak LIF Neurons** in the output layer.

*   **Mechanism**: The output neurons have a very high decay factor ($\beta \approx 0.995$).
*   **Behavior**: They effectively integrate spikes over time and hold the membrane potential constant when input ceases (steady state).
*   **Benefit**: This removes the need for an external "PostProcessor" class to handle integration, making the SNN a self-contained "Black Box" that outputs continuous voltage directly.

### 1.2 Temporal Upsampling (Sub-stepping)
A critical difference between standard RL and this benchmark is the timescale mismatch.
*   **Control Frequency**: 10 kHz (100 µs per step).
*   **SNN Inference**: Needs time to settle.

The `SNNControllerAgent` implements **Temporal Upsampling**:
*   For every **1 Control Step** (Environment update), the SNN runs **$N$ Inference Steps** (e.g., $N=10$) internally.
*   Input is repeated for all $N$ steps.
*   The final membrane potential after $N$ steps is used as the action.

### 1.3 Implicit Processor Layer
Currently, the "Processor Layer" (Pre/Post processing) is **implicit**:
*   **Normalization (Pre)**: Hardcoded in `PMSMEnv._normalize_observation`.
*   **Denormalization (Post)**: Handled in `run_benchmark.py` or implicitly by the environment's action scaling.

*Refactoring Goal*: Make this explicit (see Section 3).

---

## 2. Validation Strategy: Custom Loop vs. NeuroBench

We employ a hybrid validation strategy to satisfy both Control Theory and Neuromorphic Engineering requirements.

### 2.1 Why a Custom Loop? (`benchmark/run_benchmark.py`)
Standard RL runners (including NeuroBench's default) are insufficient for strict Power Electronics validation because:
1.  **Physics-Based Metrics**: We need to calculate **ITAE** (Integral Time-Absolute Error) and **Total Variation** (Chattering). These require access to the full trajectory data, not just scalar rewards.
2.  **Safety & Constraints**: We need frame-by-frame checks for NaN divergence and voltage limit violations to abort dangerous runs immediately.
3.  **Debugging**: The custom loop allows inspection of internal states (`e_d`, `e_q`) at every microsecond, which is critical for diagnosing controller instability.

### 2.2 Role of NeuroBench
We strictly utilize NeuroBench for **Neuromorphic Efficiency Metrics**:
*   `SynapticOperations` (SyOps)
*   `ActivationSparsity`

We wrap our agents in a partial `TorchAgent` interface solely to allow NeuroBench's metric calculators to inspect the network graph. This ensures our efficiency scores are standardized and comparable to literature, while our control scores remain physically rigorous.

---

## 3. Refactoring Plan (Code Cleanup)

To align the codebase with the clean architecture, the following refactoring steps are planned:

### 3.1 Explicit Processor Layer
**Goal**: Move normalization/denormalization out of `PMSMEnv` and `run_benchmark.py` into dedicated classes.

1.  **Create `Processors`**:
    *   `IdentityPreprocessor`: For PI Controller (Pass-through).
    *   `NormalizationPreprocessor`: Handles `state / limit` logic.
    *   `DeltaEncodingPreprocessor`: For differential inputs (if needed in future).
2.  **Update `PMSMEnv`**:
    *   Remove `_normalize_observation`. Return raw physical state.
    *   Remove action scaling. Expect raw physical voltage.
3.  **Update `Agents`**:
    *   Agents should be purely mathematical/neural (working in normalized space).
    *   The `EpisodeRunner` (or loop) connects `Env -> Pre -> Agent -> Post -> Env`.

### 3.2 SNN Folder Structure
**Goal**: Standardize the `snn/` directory.

1.  **`snn/inference/`**: Move `SimpleSNNController` here (or keep in `models.py` but clean up imports).
2.  **`snn/training/`**: specific training scripts (currently `train.py` is root).
3.  **`snn/checkpoints/`**: formalized location for `.pt` files.

### 3.3 Configuration Management
**Goal**: Centralize "Magic Numbers" (Limits, Gains, Timesteps).

1.  **`config.py`**: Create a global config object that is shared by `PMSMEnv`, `PIController`, and `SNNController`.
2.  Ensure `dt` (100µs) is defined in **one place** and propagated everywhere.
