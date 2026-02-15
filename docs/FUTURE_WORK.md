# Future Work & Roadmap

This document captures concrete extensions that would make EMBARK more
general, more realistic, and more useful for the neuromorphic motor control
community.  Items are grouped by theme and roughly ordered by impact within
each group.

---

## 1. Broader SNN Encoding Support

**Current state:** Rate-encoding SNNs only (`RateSNNStateProcessor` normalises
all features to continuous [-1, 1] scalars).

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Spike-timing encoding** | Encode features as inter-spike intervals or time-to-first-spike. Requires a temporal `SpikeTimingStateProcessor` that emits spike trains over `rate_steps` rather than scalar rates. | Medium |
| **Event-driven encoding** | Accept asynchronous event streams (e.g. from a DVS sensor or event camera). Needs a new `EventDrivenStateProcessor` with variable-length input buffers. | High |
| **Population coding** | Encode a single scalar across a population of neurons with tuning curves (Gaussian, cosine). Already partially present via dual-population encoding in the model, but not exposed at the processor level. | Low |
| **Poisson spike trains** | Stochastic rate-to-spike conversion (Bernoulli per timestep). Useful as a reference encoding for comparison. | Low |
| **Latency / phase coding** | Encode magnitude as spike latency relative to a reference oscillation. Relevant for ultra-low-latency control. | Medium |

**Design note:** All encodings should satisfy the existing `StateProcessor`
protocol (`configure` / `__call__` / `output_dim`) so they drop into
`TensorControllerAdapter` without harness changes.

---

## 2. Multi-Framework Support

**Current state:** PyTorch only (`TensorController.forward` requires
`torch.Tensor`).

| Extension | Description | Effort |
|-----------|-------------|--------|
| **NumPy / generic array protocol** | Add a `NumpyControllerAdapter` that converts between `np.ndarray` and the harness dicts. Most non-PyTorch frameworks (Lava, Brian2, NEST) can produce NumPy arrays. | Low |
| **TensorFlow / Keras** | Wrap `tf.function` models via a `TFControllerAdapter`. Needs `tf.Tensor` ↔ `torch.Tensor` bridging or a framework-agnostic observation format. | Medium |
| **JAX** | Similar to TF: wrap a JAX `apply_fn` with a thin adapter. JAX arrays convert to NumPy trivially. | Low-Medium |
| **Lava** | Intel's neuromorphic framework uses `Process` objects with `InPort`/`OutPort`. A `LavaControllerAdapter` would manage the Lava runtime and shuttle spikes/rates through ports. | High |
| **ONNX import** | Load an ONNX model and run inference via `onnxruntime`. Eliminates the need for the original training framework at benchmark time. | Medium |
| **SpiNNaker / Loihi bindings** | Native hardware backends that communicate over USB/Ethernet. Extend the existing `RemoteAkidaPolicy` pattern to a generic `RemoteHardwareController` with a common TCP/gRPC protocol. | High |

**Architecture approach:** Introduce an abstract `ArrayController` protocol
that uses `numpy.ndarray` and make `TensorController` a PyTorch-specific
subtype. The harness operates on dicts; only the adapter layer touches
framework-specific tensors.

---

## 3. Additional Motor Types & Topologies

**Current state:** PMSM only, via `gym_electric_motor`.

| Extension | Description | Effort |
|-----------|-------------|--------|
| **BLDC (trapezoidal)** | GEM already supports BLDC (`Cont-CC-BLDC-v0`). Needs a `BLDCPhysicsEngine` with 6-step commutation and a `BLDCConfig` dataclass. | Medium |
| **Induction motor (IM)** | Squirrel-cage IM with rotor flux estimation. GEM has `Cont-CC-SIM-v0`. Adds slip-frequency control as a benchmark dimension. | Medium |
| **Switched reluctance (SRM)** | Highly non-linear torque characteristic, interesting for neuromorphic control. GEM does not support SRM; would need a custom physics model or external solver. | High |
| **Multi-motor systems** | Two or more coupled motors (e.g. dual-drive traction). Needs a vectorised task that runs multiple physics engines in parallel. | High |
| **Motor parameter sweeps** | Same motor type but with randomised or swept parameters (L_d, R_s, psi_p) to test robustness. Could be a `ParameterSweepScenario` wrapper. | Low |

---

## 4. Outer Control Loops

**Current state:** Inner dq-frame current loop only; speed is held constant
by `ConstantSpeedLoad`.

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Speed control loop** | Replace `ConstantSpeedLoad` with an inertia + friction model. Add `omega_ref` to the reference dict. The SNN must now produce torque-producing current references. | Medium |
| **Position control loop** | Add an integrator on top of speed control. Relevant for servo drives and robotics. | Medium |
| **Cascaded architecture support** | Allow the benchmark to evaluate an SNN that replaces one, two, or all three cascade levels. Needs a configurable cascade harness. | High |
| **Torque control** | Direct torque control (DTC) as an alternative to FOC. Different state representation (flux, torque) and switching table. | High |

---

## 5. Realistic Simulation Enhancements

**Current state:** Deterministic, noise-free, fixed-step, no disturbances.

### 5.1 Measurement Noise & Sensor Models

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Gaussian current noise** | Additive white Gaussian noise on `i_d`, `i_q` measurements. Configurable SNR. | Low |
| **ADC quantisation** | Finite-resolution current/voltage sensing (e.g. 12-bit ADC over ±20 A range). | Low |
| **Encoder noise** | Rotor position `epsilon` noise or quantisation (e.g. 1024 ppr encoder). | Low |
| **Speed estimation error** | Simulated observer noise on `omega` (relevant when using sensorless control). | Medium |

### 5.2 Load & Disturbance Models

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Step load torque** | Sudden load change at a configurable time. Tests disturbance rejection. | Low |
| **Periodic load torque** | Sinusoidal or pulsating load (e.g. compressor, single-cylinder engine). | Low |
| **Friction model** | Coulomb + viscous friction instead of constant-speed assumption. | Medium |
| **Inertia variation** | Time-varying or uncertain moment of inertia. | Medium |

### 5.3 Parameter Uncertainty

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Randomised motor parameters** | Per-episode sampling of R_s, L_d, L_q, psi_p within tolerance bands. Tests robustness to manufacturing variation. | Low |
| **Temperature derating** | R_s increases with temperature; psi_p decreases. A thermal model could ramp temperature during an episode. | Medium |
| **Magnetic saturation** | L_d, L_q as functions of current magnitude. Non-linear inductance model. | High |

### 5.4 Solver & Timing

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Variable-step solver** | Adaptive RK45 with error tolerance. Better accuracy for stiff systems at high switching frequencies. | Medium |
| **Multirate simulation** | Different sample rates for electrical (10 kHz) and mechanical (1 kHz) subsystems. | Medium |
| **Jitter / non-uniform sampling** | Simulated interrupt jitter on the control loop timing. | Low |

---

## 6. Additional Metrics

**Current state:** MAE, ITAE, MaxError, SettlingTime, Overshoot,
SteadyStateRMS, InferenceLatency.

### 6.1 Power Quality & Efficiency

| Metric | Description | Effort |
|--------|-------------|--------|
| **Total Harmonic Distortion (THD)** | FFT of current waveform; ratio of harmonics to fundamental. Standard power quality metric. | Medium |
| **Current ripple (peak-to-peak)** | Max deviation from mean in steady state. Directly affects torque ripple. | Low |
| **Copper losses (I^2 R)** | Cumulative resistive loss over the episode. Requires R_s from config. | Low |
| **Switching losses** | Estimated from number of PWM transitions and device characteristics. | Medium |
| **Energy consumption** | Total electrical energy input (V*I integrated). | Low |
| **Power factor** | Ratio of real to apparent power. Relevant for grid-connected applications. | Medium |

### 6.2 Neuromorphic Efficiency (beyond NeuroBench)

| Metric | Description | Effort |
|--------|-------------|--------|
| **Spike count per control step** | Already tracked by `SNNControllerAgent.last_info`; expose as a first-class metric accumulator. | Low |
| **Synaptic operations (SyOps)** | Total multiply-accumulate equivalents. Currently in `last_info` but not aggregated as a metric. | Low |
| **Activation sparsity** | Fraction of silent neurons per layer. Available in `last_info`. | Low |
| **Memory footprint** | Parameter count + state buffer size. Could be computed once at start. | Low |
| **Energy-delay product** | Latency x estimated energy per inference. Composite efficiency metric. | Medium |

### 6.3 Robustness Metrics

| Metric | Description | Effort |
|--------|-------------|--------|
| **Gain margin / phase margin** | Linearise the closed loop around an operating point and compute stability margins. Requires a frequency-domain analysis pass. | High |
| **Sensitivity to noise** | Run the same scenario with and without noise; report metric degradation. | Low (once noise is added) |
| **Worst-case across parameter sweep** | Min/max/mean of any metric across a set of randomised motor parameters. | Low (once sweeps exist) |

---

## 7. Hardware Deployment & Integration

**Current state:** Akida HIL via TCP only.

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Generic remote controller protocol** | Abstract the `RemoteAkidaPolicy` TCP interface into a `RemoteController` base with JSON or protobuf messages. Any hardware backend (Loihi, SpiNNaker, FPGA) implements the same protocol. | Medium |
| **FPGA-in-the-loop** | UART/SPI bridge to run the SNN on an FPGA (e.g. Xilinx Zynq) with the physics simulation on the host. Measure real gate-level latency and power. | High |
| **Embedded C export** | Transpile a trained SNN to fixed-point C for deployment on STM32 or similar MCUs. Validate equivalence against the PyTorch reference. | High |
| **ONNX export** | Export the trained model to ONNX for deployment on edge accelerators (TensorRT, OpenVINO, Akida). Already partially possible via `torch.onnx.export`. | Low |
| **Quantisation-aware benchmarking** | Run the benchmark with INT8 or binary weights to measure accuracy degradation from quantisation. | Medium |

---

## 8. Benchmark Suite Enhancements

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Dynamic speed scenarios** | Speed ramp or step within an episode. Requires replacing `ConstantSpeedLoad` with a dynamic load model. | Medium |
| **Long-horizon scenarios** | 10+ second episodes with multiple reference changes, load steps, and speed transients. Tests long-term stability and memory. | Low |
| **Stochastic scenario sets** | Randomised reference profiles, speeds, and load torques per episode. Report statistics over N episodes. | Medium |
| **Multi-objective scoring** | Combine tracking performance, efficiency, latency, and sparsity into a single Pareto-aware score. | Medium |
| **Leaderboard / result registry** | Standardised JSON schema + a lightweight CLI to submit and compare results across groups. | Medium |
| **Parallel scenario execution** | Run independent scenarios in separate processes or threads. Straightforward since tasks and controllers are independent. | Low |
| **GPU-batched environments** | Vectorise the physics step across a batch of episodes for faster evaluation of stochastic scenarios. Requires a JAX/PyTorch-native physics engine. | High |

---

## 9. Documentation & Tooling

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Interactive notebook** | Jupyter notebook that walks through a complete benchmark run with plots (current tracking, voltage commands, spike rasters). | Low |
| **Plotting utilities** | Built-in `plot_episode(trajectory)` and `plot_comparison(summary_a, summary_b)` functions. Currently users must build their own plots. | Low |
| **CLI entry-point** | `python -m embark.benchmark run --model path/to/model.pt --scenarios quick` as a one-liner. | Low |
| **Sphinx API docs** | Auto-generated API reference from docstrings. The `docs/SPHINX_SETUP.md` stub exists but Sphinx is not wired up. | Medium |
| **CI / pre-commit** | Automated test runs, linting, and type-checking on push. | Low |

---

## 10. Research Directions

These are longer-term research questions that EMBARK could help answer once
the above engineering extensions are in place:

- **Encoding comparison:** Systematic evaluation of rate vs. temporal vs.
  population coding on the same motor control task.  Requires extensions
  from Section 1.

- **Spike-efficient control:** What is the minimum spike rate (sparsity)
  that still achieves PI-level tracking performance?  Needs per-layer
  sparsity metrics from Section 6.2.

- **Transfer across motors:** Train on one PMSM, deploy on another with
  different parameters.  Requires parameter sweeps from Section 3.

- **Online adaptation:** Can an SNN adapt its weights during an episode to
  handle parameter drift?  Requires the noise and uncertainty models from
  Section 5.

- **Real-time guarantees:** Characterise worst-case inference latency on
  neuromorphic hardware and determine whether it meets the 100 us control
  deadline.  Requires HIL from Section 7.

- **Hybrid architectures:** Combine a classical PI inner loop with an SNN
  outer loop (or vice versa).  Requires cascaded control support from
  Section 4.

---

## Priority Suggestions

For maximum near-term impact, the following items have the best
effort-to-value ratio:

1. **Measurement noise** (Section 5.1) -- low effort, big realism gain
2. **Spike count / SyOps as first-class metrics** (Section 6.2) -- data
   already in `last_info`, just needs an accumulator wrapper
3. **Parallel scenario execution** (Section 8) -- straightforward
   multiprocessing, cuts wall-clock time proportionally
4. **Plotting utilities** (Section 9) -- most-requested by users; every
   benchmark run currently ends with manual matplotlib code
5. **NumPy controller adapter** (Section 2) -- unlocks Lava, Brian2, NEST
   without a full framework integration effort
