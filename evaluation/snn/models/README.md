## Models Directory

This directory contains the SNN controller implementations for PMSM control.
Each model implements a different coding or architectural strategy.

### Available Models

1.  **MembraneSNNController (`membrane.py`)**
    -   **Approach:** Rate Coding with Non-Spiking Continuous Output.
    -   **Mechanism:** Uses slow-leak LIF neurons at the output to integrate spikes into continuous voltage.
    -   **Use case:** Baseline, high precision, simple implementation.

2.  **PopulationSNNController (`population.py`)**
    -   **Approach:** Population Coding.
    -   **Mechanism:** Multiple output neurons per dimension, each tuned to a specific value range. Output is weighted average.
    -   **Use case:** Robustness, Akida compatibility (neuromorphic hardware).

3.  **LearnedLinearSNNController (`learned_linear.py`)**
    -   **Approach:** Learned Linear Decoding.
    -   **Mechanism:** Large population of output spikes is mapped to voltage via a dense linear layer.
    -   **Use case:** Lower MSE than fixed population coding, still hardware-friendly.

4.  **DeltaSNNController (`delta.py`)**
    -   **Approach:** Delta (Incremental) Coding.
    -   **Mechanism:** Output neurons represent +dV and -dV. Voltage is integrated externally or by a counter.
    -   **Use case:** Matches integral control dynamics, extremely efficient for updates.

5.  **TTFSSNNController (`ttfs.py`)**
    -   **Approach:** Time-to-First-Spike (TTFS) Coding.
    -   **Mechanism:** Information encoded in the precise timing of the first spike within a window.
    -   **Use case:** Ultra-low latency, energy efficient (1 spike per cycle).

6.  **RecurrentSNNController (`recurrent.py`)**
    -   **Approach:** Recurrent SNN (RSNN).
    -   **Mechanism:** Hidden layers use full recurrence (all-to-all) to learn temporal dynamics implicitly.
    -   **Use case:** Learning complex temporal filters, solving temporal credit assignment.

### Usage

All models share a common configuration object `SNNConfig` and interface.

```python
from evaluation.snn.models import MembraneSNNController, SNNConfig

config = SNNConfig(hidden_size=64)
model = MembraneSNNController(config=config)
# ... training ...
```
