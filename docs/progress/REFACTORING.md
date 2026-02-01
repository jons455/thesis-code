## **ARCHITECTURAL REDESIGN REQUIREMENTS & QUESTIONS**

Based on your codebase analysis, here's a comprehensive breakdown of what you need to answer and plan before proceeding:

---

## **I. CLARIFICATION QUESTIONS**

### **A. Physics Engine Abstraction**

1. **State Space Definition**: 
   - Should the `PhysicsEngine` interface standardize on a **named state dictionary** (e.g., `{"i_d": float, "i_q": float, ...}`) or allow flexible state arrays with adapter-specific extraction?
   - How do you want to handle systems with different state dimensions? (PMSM has 4 control states, a gimbal might have 2 DoF with position/velocity)

2. **Action Space Definition**:
   - Should all physics engines accept **physical units** or **normalized actions**? Currently PMSM uses physical [u_d, u_q] in Volts.
   - Do you want voltage-based control for all systems, or should the interface support force/torque commands for mechanical systems?

3. **Coordinate Transforms**:
   - Currently GEM handles Park/Clarke transforms internally. Should the new interface abstract this as `PhysicsEngine.get_control_inputs()` and `PhysicsEngine.apply_control()` methods?
   - Or should each engine expose "native coordinates" and let users handle transforms?

4. **Operating Point Configuration**:
   - PMSM uses `(n_rpm, i_d_ref, i_q_ref)`. A gimbal might use `(angle_ref, rate_ref)`. Should the interface use a generic `ReferenceTrajectory` protocol?

### **B. Baseline Controller Abstraction**

5. **Controller Interface Standardization**:
   - Your PI controller currently uses Technical Optimum tuning specific to PMSM ($K_p = L/(2T_s)$). Should the baseline interface require:
     - **Auto-tuning methods** (e.g., `BaselineController.tune(system_params)`) 
     - **Manual parameter passing** (e.g., `BaselineController(kp, ki, kd)`)
     - Both?

6. **Baseline Types**:
   - What baseline controllers should be supported beyond PI?
     - PID (with derivative term)?
     - Model Predictive Control (MPC)?
     - Cascade controllers (outer position loop + inner velocity loop for gimbal)?

7. **Stateful Controllers**:
   - PI has integrator state. Should the interface require serialization methods (`get_state()`, `set_state()`) for mid-episode inspection?

### **C. Metrics Generalization**

8. **Domain-Specific vs Universal Metrics**:
   - Your current metrics are PMSM-specific (e.g., `RMSE_i_q`, `settling_time_iq`). Should you:
     - Create **abstract metric categories** (e.g., `TrackingError`, `SettlingTime`) with physics-engine-specific implementations?
     - Or standardize on **generic names** (e.g., `RMSE_primary`, `RMSE_secondary`) and document what "primary" means per system?

9. **Frequency-Dependent Metrics**:
   - Your PMSM runs at 10 kHz. A gimbal might run at 1 kHz. Should metrics like ITAE, Total Variation, and SyOps be:
     - **Time-normalized** (e.g., ITAE per second)?
     - **Step-normalized** (e.g., ITAE per 1000 steps)?
     - Both, with clear documentation?

10. **Neuromorphic Metrics Across Domains**:
    - Should spike statistics (total spikes, sparsity, SyOps) remain **controller-agnostic** (already good), or do they need physics-engine context (e.g., "spikes per mm of gimbal travel")?

### **D. Data Generation & Training**

11. **Expert Trajectory Generation**:
    - Currently you generate PI trajectories for SNN training. Should the new system require:
      - `BaselineController.generate_dataset(scenarios: List[Scenario], output_dir: Path)`?
      - Or keep this as manual scripts?

12. **Cross-Domain Transfer**:
    - Do you want to test if a PMSM-trained SNN can **transfer** to a gimbal (unlikely but scientifically interesting)? Or strictly separate training per domain?

---

## **II. DESIGN REQUIREMENTS CHECKLIST**

### **A. Core Abstraction Layers**

#### **1. PhysicsEngine Protocol** ✅ REQUIRED
```python
class PhysicsEngine(Protocol):
    """Abstract interface for physical systems."""
    
    def reset(self, seed: int | None = None) -> ObservationDict:
        """Reset to initial state."""
        ...
    
    def step(self, action: ActionDict) -> tuple[ObservationDict, float, bool, bool, InfoDict]:
        """Execute one control step."""
        ...
    
    def get_configuration(self) -> SystemConfig:
        """Return system parameters (for controller tuning)."""
        ...
    
    @property
    def observation_space(self) -> spaces.Dict:
        """Standardized observation space."""
        ...
    
    @property
    def action_space(self) -> spaces.Dict:
        """Standardized action space."""
        ...
```

**Files to Create**:
- `embark/benchmark/physics_engine.py` - Protocol definition
- `embark/benchmark/pmsm_physics_adapter.py` - GEM wrapper (refactor from `pmsm_env.py`)
- `embark/benchmark/system_config.py` - Dataclasses for system parameters

**Files to Modify**:
- `pmsm_env.py` → Extract physics logic to adapter, keep only reference generation

---

#### **2. BaselineController Protocol** ✅ REQUIRED
```python
class BaselineController(Protocol):
    """Abstract interface for classical controllers."""
    
    def __call__(self, observation: ObservationDict) -> ActionDict:
        """Compute control action."""
        ...
    
    def reset(self) -> None:
        """Reset internal state."""
        ...
    
    @classmethod
    def from_system_config(cls, config: SystemConfig, tuning: str = "optimal") -> "BaselineController":
        """Factory method for auto-tuning."""
        ...
```

**Files to Create**:
- `embark/benchmark/baseline_controller.py` - Protocol + abstract base class
- `embark/benchmark/baselines/pi_controller.py` - Refactor from `agents.py`
- `embark/benchmark/baselines/pid_controller.py` - Optional extension

**Files to Modify**:
- `agents.py` → Move `PIControllerAgent` to new structure, keep only `SNNControllerAgent`

---

#### **3. Metrics Framework Redesign** ⚠️ MODERATE REFACTOR

**Requirements**:
- Metrics must accept **generic observation/action keys** (not hardcoded `i_d`, `i_q`)
- Add `MetricRegistry` to dynamically select relevant metrics per physics engine
- Keep existing metric computation logic, just abstract the data access

**Files to Modify**:
- `embark/metrics/benchmark_metrics.py`:
  - Extract `compute_accuracy_metrics()` → `compute_tracking_metrics(time, actual, reference)`
  - Extract `compute_dynamics_metrics()` → Make axis-agnostic
  - Add `PhysicsEngineMetrics` protocol with method `get_metric_mappings() -> dict`

**Files to Create**:
- `embark/metrics/metric_registry.py` - Dynamic metric selection
- `embark/metrics/pmsm_metrics.py` - PMSM-specific mappings (e.g., "primary axis" = i_q)

---

### **B. Backward Compatibility Requirements**

13. **Existing Checkpoints**:
    - Your trained SNN models expect specific input shapes `[i_d, i_q, e_d, e_q]`. Should the new system:
      - Keep a `LegacyPMSMAdapter` for old models?
      - Require retraining with new observation dict format?

14. **Existing Benchmark Results**:
    - You have CSV files with PMSM-specific column names. Should migration scripts convert these to new generic format?

---

### **C. Implementation Scope Boundaries**

15. **What's IN Scope** (you must implement):
    - [ ] `PhysicsEngine` protocol
    - [ ] `PMSMPhysicsAdapter` (wraps existing GEM)
    - [ ] `BaselineController` protocol
    - [ ] Refactored PI controller as `PMSMPIController`
    - [ ] Generic `BenchmarkRunner(physics_engine, controller, metrics)`
    - [ ] Metrics registry system
    - [ ] Updated documentation with extension guide

16. **What's OUT of Scope** (user provides):
    - [ ] Gimbal physics implementation
    - [ ] Gimbal baseline controller
    - [ ] Gimbal-specific metrics (though you provide template)
    - [ ] Any actual hardware adapters

---

## **III. ARCHITECTURAL PROPOSAL**

### **Proposed New Structure**
```
embark/
├── benchmark/
│   ├── interfaces/              # NEW: Protocol definitions
│   │   ├── physics_engine.py
│   │   ├── baseline_controller.py
│   │   └── observation.py       # Standardized data structures
│   │
│   ├── physics/                 # NEW: Physics adapters
│   │   ├── __init__.py
│   │   ├── pmsm_adapter.py     # Refactored from pmsm_env.py
│   │   └── example_gimbal.py   # Template/documentation
│   │
│   ├── baselines/               # NEW: Classical controllers
│   │   ├── __init__.py
│   │   ├── pi_controller.py
│   │   └── pid_controller.py
│   │
│   ├── agents.py               # KEEP: Only SNN agents now
│   ├── controller_interface.py # MODIFY: Update to use new protocols
│   ├── processors.py           # KEEP: Unchanged (SNN-specific)
│   └── run_benchmark.py        # MODIFY: Use new PhysicsEngine interface
│
├── metrics/
│   ├── benchmark_metrics.py    # MODIFY: Genericize data access
│   ├── metric_registry.py      # NEW: Dynamic metric selection
│   └── pmsm_metrics.py         # NEW: PMSM-specific mappings
│
└── utils/
    ├── config.py               # MODIFY: Add generic SystemConfig
    └── validation.py           # NEW: Validate adapter implementations
```

---

## **IV. MIGRATION CHECKLIST**

### **Phase 1: Interface Definition** (1-2 days)
- [ ] Define `PhysicsEngine` protocol
- [ ] Define `BaselineController` protocol  
- [ ] Define `ObservationDict`, `ActionDict`, `SystemConfig` types
- [ ] Write validation tests for protocol compliance

### **Phase 2: PMSM Refactoring** (2-3 days)
- [ ] Extract GEM logic from `PMSMEnv` → `PMSMPhysicsAdapter`
- [ ] Move PI controller from `agents.py` → `baselines/pi_controller.py`
- [ ] Update `run_benchmark.py` to use new abstractions
- [ ] Verify all existing tests pass

### **Phase 3: Metrics Generalization** (1-2 days)
- [ ] Create `MetricRegistry` system
- [ ] Refactor metrics to accept generic observation keys
- [ ] Create `PMSMMetricMapping` class
- [ ] Add metric validation tests

### **Phase 4: Documentation** (1 day)
- [ ] Write "How to Add a New Physics Engine" guide
- [ ] Write "How to Implement a Baseline Controller" guide
- [ ] Update `ARCHITECTURE.md` with new diagrams
- [ ] Add example gimbal stub implementation

### **Phase 5: Validation** (1 day)
- [ ] Run full benchmark suite with refactored code
- [ ] Compare results with pre-refactor baseline (should be identical)
- [ ] Generate comparison report
- [ ] Document any numerical differences

---

## **V. CRITICAL DECISIONS NEEDED FROM YOU**

Before I can proceed with implementation, please answer:

### **DECISION 1: Observation Format**
Choose one:
- **Option A**: Standardized dict keys (`{"primary_state": float, "secondary_state": float, "primary_error": float, ...}`)
- **Option B**: Domain-specific dict keys (`{"i_d": float, "i_q": float, ...}` for PMSM, `{"theta": float, "omega": float, ...}` for gimbal)
- **Option C**: Hybrid (dict with both generic + domain-specific keys)

**Recommendation**: **Option B** (domain-specific keys) with metric registry mapping them to generic concepts.

### **DECISION 2: Baseline Auto-Tuning**
- Should `BaselineController.from_system_config()` be **required** or **optional**?
- Should you provide a default implementation using Ziegler-Nichols or similar?

**Recommendation**: Make it **required** to force proper baseline implementation.

### **DECISION 3: Backward Compatibility**
- Should we keep a `LegacyPMSMEnv` wrapper for old code, or do full migration?

**Recommendation**: **Full migration** with deprecation warnings. Old code still works via legacy imports for 1 release.

### **DECISION 4: Metrics Time Normalization**
- Should ITAE/TV metrics be reported as:
  - Total over episode (current approach)
  - Per-second average (control_frequency agnostic)
  - Both?

**Recommendation**: **Both** - store raw total + computed per-second in results dict.

---

## **VI. ESTIMATED FILE CHANGES**

| Category | New Files | Modified Files | Deleted Files |
|----------|-----------|----------------|---------------|
| Interfaces | 3 | 0 | 0 |
| Physics | 2 | 1 (pmsm_env.py) | 0 |
| Baselines | 2 | 1 (agents.py) | 0 |
| Metrics | 2 | 1 (benchmark_metrics.py) | 0 |
| Utils | 2 | 1 (config.py) | 0 |
| Tests | 5 | 3 | 0 |
| Docs | 3 | 1 | 0 |
| **TOTAL** | **19** | **8** | **0** |

---

## **Next Steps**

Once you answer the **4 critical decisions**, I will:

1. Create a detailed implementation plan with code stubs
2. Generate the protocol definitions
3. Show you example implementations for PMSM adapter
4. Create validation tests to ensure nothing breaks