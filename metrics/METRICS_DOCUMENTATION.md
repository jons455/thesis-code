# Benchmark Metrics Rationale: Bridging Control Theory and Neuromorphic Engineering

This document is relevant for the Methodology chapter!

## 1. Introduction & Scientific Motivation
The evaluation of Spiking Neural Networks (SNNs) in closed-loop control systems requires a multi-dimensional metric framework. Standard Deep Learning metrics (e.g., Top-1 Accuracy) are insufficient because they ignore temporal dynamics and physical stability. Conversely, standard Control Theory metrics (e.g., Phase Margin) often fail to capture the discrete, event-driven nature of neuromorphic hardware.

This framework synthesizes metrics from both domains to quantify the **Pareto trade-off** between **Control Fidelity** (Physical Performance) and **Neuromorphic Efficiency** (Computational Cost).


## 2. Control Fidelity Metrics (Physical Performance)

These metrics assess the quality of the physical actuation. They are calculated based on the un-normalized, physical state variables (Amperes, Volts).

### 2.1 Tracking Accuracy: RMSE vs. ITAE
While Root Mean Square Error (RMSE) is the standard for regression tasks, it is often insufficient for evaluating SNN-based controllers due to the "Steady-State Drift" phenomenon.

#### **RMSE (Root Mean Square Error)**
$$RMSE = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (i_{ref}[t] - i_{meas}[t])^2}$$
* **Scientific Justification:** RMSE provides a statistical measure of the average deviation magnitude (tracking error). It represents the signal power of the error.
* **Limitation:** It treats early transient errors (which are expected) and late steady-state errors (which indicate failure) equally.

#### **ITAE (Integral of Time-weighted Absolute Error)**
$$ITAE = \int_{0}^{T} t \cdot |e(t)| \cdot dt$$
* **Scientific Justification:** ITAE introduces a time-weighting factor $t$. Errors occurring later in the simulation are penalized more heavily than initial errors.
* **Relevance to SNNs:** SNNs utilizing Leaky Integrate-and-Fire (LIF) neurons often suffer from **leak-induced drift** at steady state. A controller might reach the setpoint but slowly drift away as the membrane potential leaks. ITAE is the most sensitive metric for detecting this specific failure mode, acting as a "Drift Detector."

### 2.2 Signal Quality: Smoothness (Total Variation)
SNNs operate on discrete events (spikes), which can lead to high-frequency switching ("chattering") in the control signal if the gain is too high or the delta-threshold is too low.

#### **Total Variation (TV)**
$$TV(u) = \sum_{t=1}^{N} |u[t] - u[t-1]|$$
* **Scientific Justification:** TV measures the accumulated "Control Effort" or "Smoothness." Mathematically, it approximates the integral of the absolute derivative of the control signal ($\int |\dot{u}| dt$).
* **Physical Implication:** High TV indicates "Bang-Bang" behavior or excessive high-frequency noise. In a physical PMSM, this high-frequency content leads to:
    1.  **Mechanical Wear:** Increased stress on bearings and gearboxes due to torque ripple.
    2.  **Acoustic Noise:** Audible whining in the motor.
    3.  **Iron Losses:** Increased eddy current losses in the motor core.
* **Thesis Argument:** A "valid" SNN controller must achieve low RMSE *without* a significantly higher TV than the PI baseline.

### 2.3 Transient Dynamics: Overshoot & Settling Time
These metrics characterize the bandwidth and damping ratio of the closed-loop system.

#### **Overshoot ($M_p$)**
$$M_p = \frac{\max(y) - y_{ss}}{y_{ss}} \times 100\%$$
* **Scientific Justification:** Overshoot correlates inversely with the system's phase margin (damping).
* **Safety Criticality:** SNNs trained via surrogate gradients can learn aggressively high gains to minimize RMSE quickly. This often results in dangerous voltage spikes. Quantifying $M_p$ serves as a proxy safety constraint.



## 3. Neuromorphic Efficiency Metrics (Computational Cost)

These metrics assess the suitability of the algorithm for deployment on event-driven hardware (e.g., Intel Loihi, SpiNNaker).

### 3.1 Dynamic Energy Proxy: Synaptic Operations (SyOps)
Unlike ANNs, which consume constant energy per inference ($E \propto N_{neurons}$), SNN energy consumption is dynamic and data-dependent.

#### **SyOps (Synaptic Operations)**
$$SyOps = \sum_{l=1}^{L} N_{spikes}^{(l)} \times N_{fanout}^{(l)}$$
* **Scientific Justification:** In CMOS neuromorphic hardware, the dominant energy cost is the "broadcast" of a spike event to downstream neurons and the subsequent accumulation of weights. SyOps is a hardware-agnostic counter of these events.
* **Energy Estimate:** Literature suggests an energy cost of approximately **23 pJ per SyOp** on Loihi 2 (14nm process) [Davies et al., 2018].
* **The Thesis Goal:** Demonstrate that the SNN controller achieves parity in Control Fidelity (RMSE) while reducing SyOps by orders of magnitude compared to a standard ANN.

### 3.2 Bandwidth Efficiency: Activation Sparsity
$$Sparsity = 1 - \frac{\sum N_{spikes}}{N_{neurons} \times N_{timesteps}}$$
* **Scientific Justification:** Measures the "silence" of the network. High sparsity implies that the system utilizes temporal coding (information in *when* a spike happens) rather than rate coding (information in *how many* spikes).
* **Implication:** Higher sparsity directly reduces the communication bandwidth requirements of the neuromorphic interconnect (Network-on-Chip congestion).



## 4. References & Standards

This metric selection is grounded in established standards from both domains:

1.  **IEEE Standard 519 & Industrial Electronics Society:**
    * Source for *Rise Time, Settling Time, Overshoot* definitions.
    * Establishes *RMSE* as the baseline for current tracking accuracy.

2.  **The NeuroBench Framework (Yik et al., 2024):**
    * Source for *SyOps* and *Sparsity* definitions.
    * Establishes the standard for comparing algorithmic complexity across different neuromorphic architectures.

3.  **Modern Control Engineering (Ogata, 2010):**
    * Source for *ITAE* as the optimal performance index for systems requiring zero steady-state error.

4.  **Intel Loihi Architecture (Davies et al., 2018):**
    * Provides the empirical basis for linking *SyOps* to physical energy consumption (Joules).