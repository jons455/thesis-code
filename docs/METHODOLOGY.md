# Methodology & Implementation Details

This document details the specific implementation choices, validation strategies, and the roadmap for refining the benchmark architecture. It serves as the bridge between the high-level design (`ARCHITECTURE.md`) and the actual code.

## 1. Implementation Reality

While `ARCHITECTURE.md` describes the logical layers, the actual implementation contains specific optimizations and adaptations for Neuromorphic Control.

### 1.1 Biological Integration (Implicit Integrator)
The initial design proposed a "Hybrid" architecture with an external mathematical integrator ($u_{acc} += kick$).
The **current implementation** (`evaluation/snn/models.py` -> `SimpleSNNController`) achieves this *biologically* using **Slow-Leak LIF Neurons** in the output layer.

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

### 2.1 Why a Custom Loop? (`embark/benchmark/run_benchmark.py`)
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

1.  **`evaluation/snn/inference/`**: Move `SimpleSNNController` here (or keep in `models.py` but clean up imports).
2.  **`evaluation/snn/training/`**: specific training scripts (currently `train.py` is root).
3.  **`models/checkpoints/`**: formalized location for `.pt` files.

### 3.3 Configuration Management (Single Source of Truth)
**Goal**: Centralize "Magic Numbers" (Limits, Gains, Timesteps) to prevent drift between Env, PI, and SNN.

1.  **`config.py`**: Create a global config object (or YAML loader).
2.  **Scope**:
    *   Motor Plant (R, L, Flux)
    *   Control Limits (I_max, U_max)
    *   Timing (dt, Simulation steps)
3.  **Benefit**: Changing `R_s` in one place updates the Simulation, the PI Tuning (Technical Optimum), and the SNN Normalization automatically.

### 3.4 Testing Consolidation
**Goal**: Centralize scattered tests.

1.  **Move**: `embark/benchmark/tests/` -> `embark/tests/benchmark/`
2.  **Move**: `embark/metrics/tests/` -> `embark/tests/metrics/`
3.  **Move**: `scripts/test_*.py` -> `embark/tests/integration/`
4.  **Benefit**: A single `pytest` command validates the entire thesis codebase.

### 3.5 Data Pipeline Standardization
**Goal**: Clear separation of raw vs. processed data.

1.  **`data/raw/`**: Simulation outputs (CSVs from GEM).
2.  **`data/processed/`**: Normalized datasets for PyTorch `DataLoader`.
3.  **`results/`**: Benchmark artifacts (plots, tables).

### 3.6 CLI Entry Points
**Goal**: Simplify execution.

1.  Create `main.py` (or `cli.py`) with `argparse` or `typer`.
2.  Commands:
    *   `python main.py train --epochs 100`
    *   `python main.py benchmark --agent pi`
    *   `python main.py visualize --run-id latest`
