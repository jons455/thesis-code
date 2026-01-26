# SNN Controller Architecture for PI Imitation Learning

This document outlines the **Pure SNN** approach for imitating a PI controller in PMSM current control. The architecture is specifically designed to address the challenges of learning the Integral (I) term.

---

## Why PI (not PID) for Motor Control?

In PMSM Field Oriented Control (FOC), **PI controllers are standard** — the Derivative (D) term is rarely used:

| Term | Purpose | Challenge for SNNs |
|------|---------|-------------------|
| **P (Proportional)** | "Error is big → output big voltage" (instant reaction) | Easy — feedforward mapping |
| **I (Integral)** | "Error has been small but persistent → slowly ramp up" (memory) | **Hard — requires temporal memory** |
| **D (Derivative)** | Rate of change → predict future | Not used (current sensors are noisy) |

The core challenge: **Standard feedforward neural networks have no memory.** To imitate the I-term, the SNN must "remember" past errors.

---

## The Integral Problem & SNN Solution

### The Mathematical Challenge

A classical PI controller computes:

```
u_d(t) = Kp_d × e_d(t) + Ki_d × ∫e_d(τ)dτ
u_q(t) = Kp_q × e_q(t) + Ki_q × ∫e_q(τ)dτ
```

The integral accumulates error over time — but neurons naturally **leak** (forget).

### The "Slow-Leak" Solution

We use **high-beta output neurons** that leak very slowly:

```
Membrane dynamics:  V(t+1) = β × V(t) + I(t)

β = 0.9   → τ ≈ 9.5 timesteps  (fast, for P-term)
β = 0.995 → τ ≈ 200 timesteps  (slow, acts as integrator)
β = 1.0   → τ = ∞              (perfect integrator, no leak)
```

With β=0.995, the output neuron **accumulates input** over time, effectively implementing the integral term.

---

## Pure SNN Architecture

### Network Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PURE SNN FOR PI IMITATION                        │
│                                                                     │
│  Input              Hidden Layers           Output Layer            │
│  [i_d, i_q,    ┌──────────────────────┐   ┌─────────────────┐      │
│   e_d, e_q] ──▶│ Dense → LIF (β=0.9)  │──▶│ Dense → LIF     │      │
│  (normalized)  │ Dense → LIF (β=0.9)  │   │ (β=0.995)       │      │
│                └──────────────────────┘   │ no reset        │      │
│                       ↓ spikes            │ ↓               │      │
│                                           │ Membrane = u_d,u_q     │
│                                           └─────────────────┘      │
│                                                                     │
│  P-TERM: Learned by hidden layers (fast dynamics)                  │
│  I-TERM: Implemented by slow-leak output neurons                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions (with Scientific Rationale)

| Decision | Choice | Rationale | Reference |
|----------|--------|-----------|-----------|
| **Recurrence** | Implicit via slow-leak | Creates temporal memory for I-term | [Zaidel2021] |
| **Hidden neurons** | LIF with β=0.9 | Fast response (P-term dynamics) | Standard snnTorch |
| **Output neurons** | LIF with β=0.995, no reset | Slow leak = built-in integrator | [Stroobants2022] |
| **Output readout** | Membrane potential | Continuous value, not spikes | [Zaidel2021] |
| **Multi-output** | Single network, 2 outputs | Learns [u_d, u_q] jointly | Multi-task learning |
| **Training target** | Absolute voltage | Direct behavioral cloning | Standard imitation |

### How Each PI Term is Learned

| PI Term | How SNN Learns It | Network Component |
|---------|-------------------|-------------------|
| **P (Kp × e)** | Hidden layers learn instant error-to-voltage mapping | LIF layers with β=0.9 |
| **I (Ki × ∫e)** | Output neurons accumulate input over time | LIF output with β=0.995, no reset |

### The "Infinite Leak" Trick (β → 1.0)

For the Integral term, setting the output layer to **very slow leak** (or no leak):

```python
# Standard neuron: voltage leaks quickly (good for P-term)
lif_hidden = snn.Leaky(beta=0.9)  # τ ≈ 9.5 steps

# Slow-leak neuron: voltage stays (perfect for I-term)  
lif_output = snn.Leaky(beta=0.995, reset_mechanism="none")  # τ ≈ 200 steps
```

The output neuron acts as a **storage** for the accumulated error (the integral).

---

## Data Flow

```
Environment State          SNN                    Action
[i_d, i_q, e_d, e_q] ──▶ MembraneSNNController ──▶ [u_d, u_q]
   (normalized)           (membrane readout)     (normalized)
   
Where:
  e_d = (i_d_ref - i_d) / i_max
  e_q = (i_q_ref - i_q) / i_max
```

### Input/Output Specification

| Signal | Range | Units | Description |
|--------|-------|-------|-------------|
| i_d | [-1, 1] | normalized | d-axis current / i_max |
| i_q | [-1, 1] | normalized | q-axis current / i_max |
| e_d | [-1, 1] | normalized | d-axis error / i_max |
| e_q | [-1, 1] | normalized | q-axis error / i_max |
| u_d | [-1, 1] | normalized | d-axis voltage / u_max |
| u_q | [-1, 1] | normalized | q-axis voltage / u_max |

### Implementation Status ✅

```
snn/
├── __init__.py
├── models.py         # Membrane/Population/LearnedLinear/Delta ✅
├── output_layers.py  # Output decoders (pop/learned/delta) ✅
├── dataset.py        # PMSMDataset (~310 lines) ✅
└── train.py          # Training script (~430 lines) ✅

benchmark/
├── agents.py         # SNNControllerAgent with multi-timestep ✅
├── controller_interface.py  # Clean benchmark API ✅
└── run_benchmark.py  # Full comparison runner ✅

scripts/
├── generate_training_data.py  # Creates clean PI trajectories ✅
├── validate_data.py           # Verifies data quality ✅
└── example_custom_controller.py  # How to test any controller ✅
```

### Training Configuration

| Aspect | Specification | Notes |
|--------|---------------|-------|
| Framework | snnTorch (PyTorch-based) | Surrogate gradients |
| Input | [i_d, i_q, e_d, e_q] normalized | [-1, 1] range |
| Target | [u_d, u_q] normalized | PI controller output |
| Loss | MSE(membrane, target) | On voltage, not spikes |
| Sequence | BPTT through time | 100-step windows |
| Batch size | 32 trajectories | ~950 batches/epoch |
| Learning rate | 1e-3 (Adam) | Cosine annealing |
| Data | `train_v2/` (1000 files) | 100% clean tracking |

### Multi-Timestep Inference (Option B)

Following van Breukelen (2025), the agent supports multiple SNN timesteps per control step:

```python
# In SNNControllerAgent
for _ in range(self.num_inference_steps):
    voltage, self._snn_state, spike_info = self.model(
        state_tensor, self._snn_state, return_spikes=True
    )
# Use final voltage as output
```

This allows proper spike integration before output readout.

---

## Output Coding Options (5 Total)

This repo now supports four implemented output strategies plus one conceptual option:

| Option | Output Type | Where Implemented | Akida Compatibility |
|--------|-------------|-------------------|---------------------|
| **Membrane Readout** | Membrane potential | `MembraneSNNController` | Low (host readout) |
| **Population Coding** | Spikes + fixed tuning curves | `PopulationSNNController` | High |
| **Learned Linear Decoding** | Spikes + learned dense readout | `LearnedLinearSNNController` | High |
| **Delta (Incremental) Coding** | Up/Down spikes per axis | `DeltaSNNController` | High |
| **Direct PWM** | Spike trains → duty cycles | Concept only | Experimental |

Notes:
- **Delta Coding** uses an internal accumulator during training and can be mapped to a host-side counter on Akida.
- **Learned Linear** trades fixed preferred values for a trainable decoder (often lower MSE).

## Alternative Architectures (If Pure SNN Has Issues)

### Comparison of Approaches

| Approach | Integration | Sparsity | Hardware | Status |
|----------|-------------|----------|----------|--------|
| **Pure SNN** (primary) | Slow-leak output neurons | Hidden sparse, output tonic | Fully on-chip | ✅ Implemented |
| **Hybrid SNN** | External integrator | All layers sparse | SNN on chip, integrator on host | 🔲 Optional |
| **Two Networks** | Separate d/q SNNs | Per-axis optimization | Parallel chips | 🔲 Optional |

### Option: Two Separate Networks

If the single network struggles to learn both outputs, split into two:

```python
# Matches classical PI structure exactly
class TwoAxisSNNController:
    def __init__(self):
        self.snn_d = MembraneSNNController(input_size=2, output_size=1)  # [i_d, e_d] → u_d
        self.snn_q = MembraneSNNController(input_size=2, output_size=1)  # [i_q, e_q] → u_q
    
    def __call__(self, state):
        i_d, i_q, e_d, e_q = state
        u_d = self.snn_d([i_d, e_d])
        u_q = self.snn_q([i_q, e_q])
        return [u_d, u_q]
```

Advantage: Decoupled training (each axis learns independently).

### Option: Hybrid SNN-Integrator

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HYBRID SNN CONTROLLER                          │
│                                                                     │
│  Input              SNN (all β=0.9)        External Integrator      │
│  [i,e] ──▶ DeltaEncoder ──▶ LIF Network ──▶ u += decoded_kick       │
│                                    ↓                ↓               │
│                              [kick_d, kick_q]  [u_d, u_q]           │
└─────────────────────────────────────────────────────────────────────┘
```

- SNN predicts **delta voltage** (Δu), not absolute voltage
- External integrator accumulates: u(t) = u(t-1) + Δu
- Advantage: SNN can be completely silent at steady state (maximum sparsity)
- Disadvantage: Integration happens outside neuromorphic hardware

### When to Use Alternatives

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Poor d-axis tracking | Coupled learning interferes | Two Networks |
| Voltage drift at steady-state | β not high enough | Increase to 0.999 or Hybrid |
| Oscillation/chattering | Output too sensitive | Add output smoothing |
| Slow training convergence | Task too complex | Two Networks |

---

## Implementation Status

### Phase 1: Pure SNN Foundation ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create folder structure | ✅ | `snn/` with 4 files |
| MembraneSNNController | ✅ | 440 lines, membrane readout |
| PMSMDataset | ✅ | 310 lines, windowed sequences |
| Training script | ✅ | 430 lines, CLI args |
| Clean training data | ✅ | `train_v2/` (1000 files, 0A error) |

### Phase 2: Closed-Loop Integration ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| SNNControllerAgent | ✅ | Multi-timestep inference |
| Benchmark API | ✅ | `controller_interface.py` |
| Spike tracking | ✅ | `get_spike_statistics()` |
| Test with PI baseline | ✅ | 78mA RMSE, 2.4ms settling |

### Phase 3: Training & Evaluation 🔲 READY

| Task | Status | Command |
|------|--------|---------|
| Full training run | 🔲 Ready | `poetry run python -m snn.train --epochs 100` |
| Closed-loop validation | 🔲 | `poetry run python -m benchmark.run_benchmark --full-metrics` |
| PI vs SNN comparison | 🔲 | `poetry run python -m benchmark.run_benchmark --compare` |
| Generate plots | 🔲 | After training |

### Phase 4: Extensions 🔲 FUTURE

| Task | Priority | Notes |
|------|----------|-------|
| Two Networks approach | Medium | If single network struggles |
| Quantization (Akida) | Low | After SNN works |
| NIR export | Low | For hardware portability |
| Multi-operating-point | Medium | Speed/load sweep |

---

## Implementation Details (Actual Code)

### MembraneSNNController (snn/models.py)

```python
class MembraneSNNController(nn.Module):
    """
    Pure SNN controller with built-in integration (membrane potential readout).
    
    Architecture:
        Input [4] → Dense → LIF (β=0.9) → Dense → LIF (β=0.9) → Dense → LIF (β=0.995)
                                                                          ↓
                                                                    Membrane = [u_d, u_q]
    """
    
    def __init__(self, config: SNNConfig = None, hidden_size: int = 64):
        # ...
        
        # Output layer - SLOW leak (built-in integration)
        self.fc_out = nn.Linear(hidden_size, 2)
        self.lif_out = snn.Leaky(
            beta=0.995,           # Slow decay = integrator
            reset_mechanism="none",  # Don't reset - accumulate!
        )
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, state=None, return_spikes=False):
        # Process through hidden layers (fast dynamics)
        for layer, neuron in zip(self.layers, self.neurons):
            cur = layer(spk)
            spk, mem = neuron(cur, hidden_mems[i])
        
        # Output layer - read membrane potential (not spikes!)
        cur_out = self.fc_out(spk)
        _, mem_out = self.lif_out(cur_out, mem_out)
        
        # Scale and clip to [-1, 1]
        voltage = torch.tanh(mem_out * self.output_scale)
        
        return voltage, new_state, spike_info
```

### SNNControllerAgent (benchmark/agents.py)

```python
class SNNControllerAgent:
    """
    SNN controller agent with multi-timestep inference.
    
    Parameters:
        checkpoint_path: Path to trained model
        num_inference_steps: SNN timesteps per control step (Option B)
    """
    
    def __init__(self, checkpoint_path, num_inference_steps=1):
        self.model = MembraneSNNController.load(checkpoint_path)
        self.num_inference_steps = num_inference_steps
        self._snn_state = None
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        # Run multiple internal timesteps per control step
        for _ in range(self.num_inference_steps):
            voltage, self._snn_state, spike_info = self.model(
                state_tensor, self._snn_state, return_spikes=True
            )
        
        return voltage.numpy().clip(-1, 1)
    
    def reset(self):
        self._snn_state = None
    
    def get_spike_statistics(self) -> dict:
        # Returns: total_spikes, sparsity, latency, etc.
        pass
```

### Benchmark API (benchmark/controller_interface.py)

```python
# The interface any controller must implement:
class ControllerInterface(Protocol):
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state=[i_d,i_q,e_d,e_q] → action=[u_d,u_q]"""
        ...
    
    def reset(self) -> None:
        """Reset internal state"""
        ...

# Run benchmark on any controller:
results = run_benchmark(my_controller, env)
print(results.summary())
```

---

## Success Criteria

### Current Baseline (PI Controller)

| Metric | PI Value | Notes |
|--------|----------|-------|
| RMSE i_q | 78 mA | Dominated by transient |
| Final error | 0.00 mA | Perfect steady-state |
| Settling time | 2.4 ms | Fast response |
| Overshoot | 73% | Technical Optimum tuning |

### Minimum Viable Product (MVP)

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Training converges | Loss < 0.01 | Training curve |
| Closed-loop stable | No NaN/explosion | Episode completes |
| Tracks reference | RMSE < 1.0 A | Benchmark metrics |
| Correct polarity | u_q positive when e_q positive | Manual check |
| Demonstrates sparsity | > 50% silent neurons | Activation logging |

### Good Result

| Criterion | Target | Comparison |
|-----------|--------|------------|
| Competitive tracking | RMSE < 500 mA | 6× of PI baseline |
| Fast response | Settling time < 20 ms | 8× of PI |
| High sparsity | > 80% silent | NeuroBench metrics |
| Smooth control | TV < 2× PI | No chattering |

### Excellent Result (Thesis Win)

| Criterion | Target | Comparison |
|-----------|--------|------------|
| Near-PI accuracy | RMSE < 200 mA | 3× of PI baseline |
| Fast settling | < 10 ms | 4× of PI |
| Very sparse | > 90% silent | Efficient |
| Energy advantage | < 1 mJ/inference | SyOps estimation |
| Akida-ready | Quantized model | 4-bit weights |

---

## Dependencies (Already Installed)

```toml
# In pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
torch = "^2.0"
snntorch = "^0.9"
gym-electric-motor = "^1.0"
gymnasium = "^0.29"
numpy = "^1.24"
pandas = "^2.0"
matplotlib = "^3.7"
```

Key packages:
- **snnTorch**: LIF neurons with surrogate gradients
- **gym-electric-motor (GEM)**: PMSM simulation
- **gymnasium**: Environment interface

---

## Risk Mitigation

| Risk | Symptom | Mitigation |
|------|---------|------------|
| Training data corrupted | Wrong polarity outputs | Use `train_v2/` (clean data) ✅ FIXED |
| Training doesn't converge | Loss plateaus high | Increase hidden size, tune beta |
| Closed-loop unstable | NaN, explosion | Gradient clipping, reduce lr |
| Poor steady-state | Final error > 0.5 A | Increase β → 0.999 |
| Voltage drift | Slow error accumulation | Add Hybrid integrator |
| Wrong d/q coupling | d-axis affects q-axis | Use Two Networks approach |
| Chattering/oscillation | High TV metric | Smooth output, increase β |

---

## Future Extensions

After Pure SNN works:

1. **Quantization**: Add QAT for Akida deployment (4-bit weights)
2. **NIR Export**: Export to neuromorphic intermediate representation
3. **Operating Points**: Test across speed/load conditions
4. **Disturbance Rejection**: Evaluate robustness
5. **Compare Architectures**: Pure SNN vs Two Networks vs Hybrid

---

## References

### PI-to-SNN Imitation Learning

1. **Stroobants, S. et al. (2022)**. "Parsimonious Neuromorphic PID for Quadrotor Altitude Control."
   - arXiv:2109.10199
   - Position-coded N-PI on Loihi, 93 neurons
   - 100× energy savings vs ARM Cortex-M4

2. **Zaidel, Y. et al. (2021)**. "Neuromorphic NEF-Based Inverse Kinematics and PID Control."
   - Front. Neurorobot., PMC7887770
   - Rate-coded PI with membrane potential readout
   - 250-500 neurons per axis

3. **Stroobants, S. et al. (2023)**. "Neuromorphic Control using Input-Weighted Threshold Adaptation."
   - arXiv:2304.08778
   - IWTA mechanism for precise integration
   - 10 neurons vs 30 for position-coded integrator

4. **van Breukelen Castillo, M.F. (2025)**. "SNNs for High-Speed Continuous Control."
   - IMAVS 2025, Paper 17
   - Multiple integration cycles per control step
   - Hybrid spike-rate decoding

### Key Insights from Literature

| Paper | Key Contribution | Applied Here |
|-------|------------------|--------------|
| [Stroobants2022] | Slow-leak neurons for integration | β=0.995 output layer |
| [Zaidel2021] | Membrane potential readout | reset_mechanism="none" |
| [Stroobants2023] | IWTA for precise integration | Future extension |
| [vanBreukelen2025] | Multi-timestep inference | num_inference_steps param |

---

*Last Updated: 2026-01-22*
