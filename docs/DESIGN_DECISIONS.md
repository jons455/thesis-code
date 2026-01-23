# Design Decisions Log

This document records key architectural and implementation decisions made during the project. Each decision includes the context, alternatives considered, and rationale.

Use this for the thesis Implementation and Discussion chapters.

---

## Decision Index

| # | Decision | Date | Impact |
|---|----------|------|--------|
| D1 | GEM (Python) over MATLAB for simulation | ~Dec 2025 | High |
| D2 | NeuroBench framework integration | 2026-01-13 | High |
| D3 | Gymnasium wrapper design | 2026-01-13 | Medium |
| D4 | Pure SNN vs Hybrid SNN-Integrator | 2026-01-20 | High |
| D5 | Slow-leak output neurons | 2026-01-20 | High |
| D6 | Direct voltage target (not Δu) | 2026-01-20 | Medium |
| D7 | snnTorch framework | 2026-01-20 | Medium |

---

## D1: GEM (Python) over MATLAB for Simulation (estimated)

**Date**: ~December 2025  
**Category**: Simulation Environment

### Context
Need a PMSM simulation environment for training and evaluating controllers. The benchmark should be easily accessible and reproducible.

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **MATLAB/Simulink** | Industry standard, validated | License required, mixed languages, not portable |
| **GEM (gym-electric-motor)** | Pure Python, Gym interface, open source | Less documentation |
| **Custom simulation** | Full control | Time-consuming, validation needed |

### Decision
**→ GEM (gym-electric-motor)**

### Rationale
1. **All-in-one language** — Entire benchmark pipeline in Python (no MATLAB dependency)
2. **Easily accessible** — Anyone can run the benchmark without licenses
3. **Shippable** — Complete package can be shared/published
4. **NeuroBench compatible** — OpenAI Gym interface works with benchmark framework
5. **Open source** — Reproducible by other researchers

### Key Insight
We don't need the "most perfect" simulation — we need an **accessible, reproducible benchmark**. GEM is validated and good enough for fair SNN vs PI comparison.

### Validation
GEM simulation matched MATLAB/Simulink with tracking error < 1e-11 A at steady state (used for validation only, not the main pipeline).

---

## D2: NeuroBench Framework Integration

**Date**: 2026-01-13  
**Category**: Benchmarking Framework

### Context
Need standardized metrics for neuromorphic controller evaluation.

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **NeuroBench** | Standardized, peer-reviewed, community | Newer, less documentation |
| **Custom metrics** | Full control | Not comparable to other work |
| **MLPerf** | Well-established | Not neuromorphic-focused |

### Decision
**→ NeuroBench (2025_GC branch)**

### Rationale
1. Specifically designed for neuromorphic computing
2. Includes closed-loop control support (BenchmarkClosedLoop)
3. Standardized metrics (SyOps, sparsity, etc.)
4. Published framework → citable in thesis
5. Enables comparison with other neuromorphic work

### Implementation
- Installed from `2025_GC` branch (commit c8dfd47)
- Created Gymnasium wrapper (PMSMEnv) for compatibility

---

## D3: Gymnasium Wrapper Design

**Date**: 2026-01-13  
**Category**: Interface Design

### Context
GEM environment needs to be wrapped for NeuroBench compatibility.

### Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Observation space | `[i_d, i_q, e_d, e_q]` normalized | Minimal state, includes error |
| Action space | `[u_d, u_q]` in [-1, 1] | Normalized for NN |
| Coordinate transform | dq internally | SNN works in dq frame |
| Reference handling | Inside env | Simplifies agent interface |

### Rationale
- Agent sees normalized values → no domain knowledge needed
- Error included in observation → agent doesn't need to compute
- Consistent with NeuroBench TorchAgent expectations

---

## D4: Pure SNN vs Hybrid SNN-Integrator

**Date**: 2026-01-20  
**Category**: SNN Architecture (Critical Decision)

### Context
At steady state, if error is constant (zero), a standard SNN would stop spiking and output would drift. Need a solution.

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Hybrid SNN-Integrator** | SNN outputs Δu, external integrator accumulates | Maximum sparsity | Integrator runs on host CPU |
| **Pure SNN (slow-leak)** | Output neurons with high β act as integrator | Fully on-chip | Slight leak at steady state |
| **Rate-coded output** | Continuous firing encodes voltage | Simple | No sparsity benefit |
| **Recurrent SNN** | Internal recurrence maintains state | Flexible | Hard to train |

### Decision
**→ Pure SNN with slow-leak output neurons**

### Rationale
1. **Simpler pipeline** — No external integrator code needed
2. **Fully on-chip** — Deployable entirely on Akida hardware
3. **Same training data** — Can use absolute voltage targets
4. **Fallback available** — Can add Hybrid later if needed

### Trade-offs
- Slight voltage drift possible (β=0.995 → 0.5% decay/step)
- Hidden layers still sparse, output less so
- But practical for 10kHz control (drift << control action)

### Hardware Consideration
BrainChip Akida supports LIF neurons with configurable time constants → slow-leak is native.

---

## D5: Slow-Leak Output Neuron Parameters

**Date**: 2026-01-20  
**Category**: Neuron Dynamics

### Context
Need to choose decay rates (β) for LIF neurons.

### Decision

| Layer | β Value | Time Constant | Behavior |
|-------|---------|---------------|----------|
| Hidden | 0.9 | ~10 steps | Fast, responsive |
| Output | 0.995 | ~200 steps | Slow, integrating |

### Rationale
- **Hidden β=0.9**: Fast response to input changes, forgets quickly
- **Output β=0.995**: At 10kHz, this gives τ ≈ 20ms — holds voltage across control cycles
- Balance between holding state and allowing corrections

### Mathematical Basis
```
τ = dt / (1 - β)
For β=0.995, dt=0.1ms: τ = 0.0001 / 0.005 = 20ms

Membrane after N steps: V(N) = V(0) * β^N
After 100 steps: V = V(0) * 0.995^100 ≈ 0.6 * V(0)
```

---

## D6: Direct Voltage Target (not Δu)

**Date**: 2026-01-20  
**Category**: Training Target

### Context
Hybrid approach uses Δu (voltage change) as target. What should Pure SNN use?

### Options

| Target | Formula | When to Use |
|--------|---------|-------------|
| **Absolute voltage** | `y = [u_d, u_q]` | Pure SNN (slow-leak integrates) |
| **Delta voltage** | `y = u[t] - u[t-1]` | Hybrid SNN (external integrator) |

### Decision
**→ Absolute voltage target**

### Rationale
1. Slow-leak output neurons naturally integrate input
2. Training on absolute values is more stable
3. Direct correspondence: membrane potential ≈ voltage command
4. Simpler loss function (MSE on voltage)

---

## D7: snnTorch Framework

**Date**: 2026-01-20  
**Category**: SNN Framework

### Context
Need a framework for implementing and training SNNs.

### Options Considered

| Framework | Pros | Cons |
|-----------|------|------|
| **snnTorch** | PyTorch-based, GPU support, active | Newer |
| **Norse** | Research-focused | Less documentation |
| **BindsNET** | Biologically detailed | Slower |
| **Brian2** | Simulation-focused | Not for training |
| **Lava (Intel)** | Loihi-native | Hardware-specific |

### Decision
**→ snnTorch**

### Rationale
1. Built on PyTorch → familiar, GPU-accelerated
2. Surrogate gradient methods for training
3. Good documentation and tutorials
4. Supports NIR export → hardware portability
5. Active development (v0.9.4)

---

## Template for Future Decisions

```markdown
## DX: [Decision Title]

**Date**: YYYY-MM-DD  
**Category**: [Architecture/Training/Evaluation/etc.]

### Context
[What problem needed solving?]

### Options Considered
| Option | Pros | Cons |
|--------|------|------|

### Decision
**→ [Chosen option]**

### Rationale
1. [Reason 1]
2. [Reason 2]

### Trade-offs
- [What we gave up]
- [Risks accepted]
```

---

*Last Updated: 2026-01-20*
