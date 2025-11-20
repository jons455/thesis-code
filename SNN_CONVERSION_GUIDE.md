# SNN Konvertierungs-Guide: Von ANN zu Spiking Neural Network

## Überblick

Dieser Guide beschreibt den zweistufigen Ansatz zur Erstellung eines Spiking Neural Network (SNN) Controllers:

1. **Phase 1:** ANN mit ReLU-Aktivierung trainieren (Edge Impulse)
2. **Phase 2:** Konvertierung zu SNN mit LIF-Neuronen

---

## 🎯 Warum dieser Ansatz?

### Vorteile der ANN-to-SNN Konvertierung

✅ **Training ist einfacher:**
- ANNs: Backpropagation ist gut etabliert und schnell
- SNNs: Direktes Training mit Spikes ist komplizierter (STDP, Surrogate Gradients)

✅ **Nutzt bewährte Tools:**
- Edge Impulse für ANN-Training
- Standard SNN-Toolboxen für Konvertierung

✅ **Deployment-Vorteile:**
- Ultra-niedrige Energie auf neuromorpher Hardware (Intel Loihi, BrainScaleS)
- Event-driven processing (nur aktiv bei Änderungen)
- Asynchrone Verarbeitung

---

## 📊 Datenlage: Ausreichend!

**Ihre aktuelle Datenmenge:**
- ✅ 2,001,000 Samples
- ✅ 1000 verschiedene Betriebspunkte
- ✅ Vollständige Abdeckung des Arbeitsbereichs

**Für ein 3→64→32→16→2 Netzwerk (~3,500 Parameter):**
- **Verhältnis:** ~570 Samples pro Parameter
- **Bewertung:** **MEHR als ausreichend!** 🎉
- **Typische Regel:** 10-100 Samples/Parameter für Regression

**Mehr Daten nur nötig wenn:**
- ❌ Overfitting trotz Regularisierung
- ❌ Schlechte Performance auf Test-Set
- ❌ Andere Motoren/Bedingungen abdecken

---

## 🔧 Phase 1: ANN Training (Edge Impulse)

### Schritt 1.1: Daten vorbereiten
```bash
cd data-preperation
python prepare_edge_impulse.py
```

### Schritt 1.2: Edge Impulse Training
Siehe **EDGE_IMPULSE_TRAINING_GUIDE.md** für Details.

**Architektur:**
```
Input [3]: i_d, i_q, n
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Output [2]: u_d, u_q (Linear)
```

**Ziel:** MAE < 1.0V

### Schritt 1.3: Model Export
Nach erfolgreichem Training:
1. **Deployment** Tab → **TensorFlow Lite** oder **Keras**
2. Download: `model.h5` oder `model.tflite`
3. Oder: Export als **ONNX** für maximale Kompatibilität

---

## 🧠 Phase 2: SNN Konvertierung

### Option A: SNN Toolbox (Empfohlen für Start)

#### Installation
```bash
pip install snntoolbox
```

#### Konvertierung
```python
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# Konfiguration erstellen
config = import_configparser()

# Pfade setzen
config.set('paths', 'path_wd', './snn_output')
config.set('paths', 'dataset_path', './edge_impulse_data')
config.set('paths', 'filename_ann', 'model.h5')  # Ihr trainiertes ANN

# Input/Output Shape
config.set('input', 'model_lib', 'keras')
config.set('input', 'dataset_format', 'npz')

# SNN Parameter
config.set('conversion', 'simulator', 'INI')  # oder 'brian2', 'nest', etc.
config.set('conversion', 'neuron_type', 'IF_curr_exp')  # Leaky IF
config.set('conversion', 'conversion_method', 'max-weight')

# Simulation Parameter
config.set('simulation', 'duration', 100)  # Anzahl Zeitschritte
config.set('simulation', 'dt', 0.1)  # Zeitschritt in ms
config.set('simulation', 'batch_size', 1)

# Konvertierung durchführen
main(config)
```

#### LIF-Neuron Parameter
```python
# Typische Werte für LIF-Neuronen
tau_mem = 20.0      # Membrane time constant (ms)
tau_syn = 5.0       # Synaptic time constant (ms)
v_threshold = 1.0   # Firing threshold
v_reset = 0.0       # Reset potential
t_refrac = 2.0      # Refractory period (ms)
```

### Option B: Norse (für PyTorch)

#### Installation
```bash
pip install norse
```

#### Konvertierung & Training
```python
import torch
import torch.nn as nn
import norse.torch as norse

# ANN-Gewichte laden
ann_weights = torch.load('model.pth')

# SNN erstellen
class SNNController(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LIF-Parameter
        lif_params = norse.LIFParameters(
            tau_mem_inv=1/20e-3,  # 20ms membrane time constant
            tau_syn_inv=1/5e-3,   # 5ms synaptic time constant
            v_th=1.0,             # threshold
            v_reset=0.0           # reset potential
        )
        
        # Netzwerk mit LIF-Neuronen
        self.fc1 = nn.Linear(3, 64)
        self.lif1 = norse.LIFCell(p=lif_params)
        
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = norse.LIFCell(p=lif_params)
        
        self.fc3 = nn.Linear(32, 16)
        self.lif3 = norse.LIFCell(p=lif_params)
        
        self.fc_out = nn.Linear(16, 2)
        
    def forward(self, x, state1=None, state2=None, state3=None):
        # LIF-Neuron States
        if state1 is None:
            state1 = self.lif1.initial_state(x.shape[0], self.fc1.out_features)
        if state2 is None:
            state2 = self.lif2.initial_state(x.shape[0], self.fc2.out_features)
        if state3 is None:
            state3 = self.lif3.initial_state(x.shape[0], self.fc3.out_features)
        
        # Forward pass durch SNN
        x = self.fc1(x)
        spikes1, state1 = self.lif1(x, state1)
        
        x = self.fc2(spikes1)
        spikes2, state2 = self.lif2(x, state2)
        
        x = self.fc3(spikes2)
        spikes3, state3 = self.lif3(x, state3)
        
        # Output layer (rate-coded)
        output = self.fc_out(spikes3)
        
        return output, (state1, state2, state3)

# Model erstellen und Gewichte übertragen
snn = SNNController()

# ANN-Gewichte in SNN laden
# (manuell Gewichte kopieren: fc1, fc2, fc3, fc_out)
snn.fc1.weight.data = ann_weights['layer1.weight']
snn.fc1.bias.data = ann_weights['layer1.bias']
# ... etc für alle Layers
```

### Option C: snnTorch

#### Installation
```bash
pip install snntorch
```

#### Rate-Coded SNN
```python
import torch
import torch.nn as nn
import snntorch as snn

class SNNController(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Leaky Integrate-and-Fire Parameter
        beta = 0.95  # Decay rate (höher = längeres "Gedächtnis")
        
        # Netzwerk-Layers
        self.fc1 = nn.Linear(3, 64)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(32, 16)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.fc_out = nn.Linear(16, 2)
        
    def forward(self, x, num_steps=25):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Output accumulator
        output_sum = torch.zeros(x.shape[0], 2)
        
        # Simuliere über mehrere Zeitschritte
        for t in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Output (rate-coded: summiere Spikes)
            output_sum += self.fc_out(spk3)
        
        # Durchschnitt über Zeit = Rate-Coding
        return output_sum / num_steps

# Gewichte von trainiertem ANN laden
snn = SNNController()
# ... Gewichte kopieren von Edge Impulse Model
```

---

## 🔬 Phase 3: SNN Evaluation & Tuning

### Wichtige Metriken

#### 1. Genauigkeit
```python
import numpy as np
from sklearn.metrics import mean_absolute_error

# Test auf Validierungsdaten
predictions_ann = ann_model.predict(X_test)
predictions_snn = snn_model(X_test, num_steps=50)

mae_ann = mean_absolute_error(y_test, predictions_ann)
mae_snn = mean_absolute_error(y_test, predictions_snn.numpy())

print(f"ANN MAE: {mae_ann:.3f} V")
print(f"SNN MAE: {mae_snn:.3f} V")
print(f"Genauigkeitsverlust: {(mae_snn - mae_ann):.3f} V ({(mae_snn/mae_ann - 1)*100:.1f}%)")
```

**Typisch:** 1-5% Genauigkeitsverlust bei Konvertierung

#### 2. Latenz
```python
import time

# ANN Inference Zeit
start = time.perf_counter()
for _ in range(1000):
    _ = ann_model.predict(sample_input)
ann_time = (time.perf_counter() - start) / 1000

# SNN Inference Zeit
start = time.perf_counter()
for _ in range(1000):
    _ = snn_model(sample_input, num_steps=50)
snn_time = (time.perf_counter() - start) / 1000

print(f"ANN Latenz: {ann_time*1000:.2f} ms")
print(f"SNN Latenz: {snn_time*1000:.2f} ms")
print(f"Für 10 kHz Control: {'✅ OK' if snn_time < 0.1e-3 else '⚠️ Zu langsam'}")
```

**Herausforderung:** 10 kHz = 100 μs pro Cycle
- SNNs brauchen mehrere Zeitschritte
- 50 Schritte @ 0.1ms = 5ms → **zu langsam für 10 kHz!**
- Lösung: Weniger Schritte (10-20) oder schnellere Hardware

#### 3. Energie-Verbrauch (auf Neuromorpher Hardware)

```python
# Nur möglich mit echter neuromorpher Hardware
# Beispiel: Intel Loihi

# Spikes zählen
total_spikes = count_spikes_during_inference(snn_model, X_test)
avg_spikes_per_sample = total_spikes / len(X_test)

# Energie ≈ Anzahl Spikes (auf neuromorpher HW)
print(f"Durchschnitt Spikes pro Inference: {avg_spikes_per_sample:.1f}")

# Vergleich zu ANN (grobe Schätzung)
# ANN: Alle Neuronen aktiv bei jedem Sample
# SNN: Nur spikende Neuronen verbrauchen Energie
```

### Parameter-Tuning

#### 1. Anzahl Simulationsschritte
```python
# Trade-off: Genauigkeit vs Latenz
for num_steps in [10, 25, 50, 100]:
    pred = snn_model(X_test, num_steps=num_steps)
    mae = mean_absolute_error(y_test, pred)
    print(f"Steps={num_steps:3d}: MAE={mae:.3f}V")

# Optimum finden
# Oft: 25-50 Schritte ausreichend
```

#### 2. LIF Time Constants
```python
# Membrane time constant (tau_mem)
# Höher = längeres "Gedächtnis", glattere Outputs
# Niedriger = schnellere Reaktion, mehr Spikes

for tau_mem in [10, 20, 30, 50]:
    lif_params = norse.LIFParameters(tau_mem_inv=1/(tau_mem*1e-3))
    # ... Model mit neuen Parametern testen
```

#### 3. Firing Threshold
```python
# Threshold bestimmt Spike-Rate
# Zu niedrig: Zu viele Spikes, ineffizient
# Zu hoch: Zu wenige Spikes, Information verloren

for threshold in [0.5, 1.0, 1.5, 2.0]:
    # ... Model mit neuem Threshold testen
```

---

## 📊 Erwartete Ergebnisse

### Baseline (ANN mit ReLU)
| Metrik | Wert |
|--------|------|
| MAE u_d | 0.8 V |
| MAE u_q | 0.8 V |
| Inference Zeit | 1-2 ms |
| Energie | Baseline |

### Nach SNN-Konvertierung
| Metrik | Optimistisch | Realistisch | Pessimistisch |
|--------|--------------|-------------|---------------|
| MAE u_d | 0.9 V | 1.0 V | 1.5 V |
| MAE u_q | 0.9 V | 1.0 V | 1.5 V |
| Genauigkeitsverlust | 10% | 20% | 50% |
| Inference Zeit (50 steps) | 5 ms | 10 ms | 20 ms |
| Energie (neuromorph) | 10× weniger | 50× weniger | 100× weniger |

---

## 🚀 Deployment-Optionen

### Option 1: Software-SNN auf Standard-MCU
**Pro:**
- Kein spezielle Hardware nötig
- Flexibel für Debugging

**Contra:**
- Keine echten Energie-Vorteile
- Latenz könnte zu hoch sein für 10 kHz

**Geeignet für:**
- Proof-of-Concept
- Vergleichsstudien
- Wenn neuromorphe Hardware nicht verfügbar

### Option 2: Neuromorphe Hardware

#### Intel Loihi 2
- 1 Million LIF-Neuronen
- Event-driven, asynchrone Verarbeitung
- ~1000× energie-effizienter als CPU
- Zugang: Intel Neuromorphic Research Community

#### BrainScaleS-2
- Analog neuromorphe Hardware
- 10,000× schneller als biologische Zeit
- Universität Heidelberg

#### SpiNNaker
- Digital neuromorphe Hardware
- Bis zu 1 Million Kerne
- Universität Manchester

### Option 3: Hybrid (ANN + SNN)
```cpp
// Fallback-Strategie
if (snn_inference_time < deadline) {
    use_snn_output();  // Niedrige Energie
} else {
    use_ann_output();  // Schneller Fallback
}
```

---

## 📝 Thesis-Gliederung (Vorschlag)

### Kapitel 4: ANN Training
- 4.1 Datensammlung (MATLAB Simulation)
- 4.2 Datenaufbereitung
- 4.3 Edge Impulse Training
- 4.4 ANN Performance Evaluation

### Kapitel 5: SNN Konvertierung
- 5.1 Motivation (Energie-Effizienz)
- 5.2 ANN-to-SNN Konvertierungsmethoden
  - 5.2.1 Rate Coding
  - 5.2.2 LIF-Neuron Modell
- 5.3 Implementierung (SNN Toolbox/Norse/snnTorch)
- 5.4 Parameter-Tuning

### Kapitel 6: Vergleich & Evaluation
- 6.1 Genauigkeit: ANN vs SNN
- 6.2 Latenz-Analyse
- 6.3 Energie-Verbrauch (falls neuromorphe HW verfügbar)
- 6.4 Trade-offs und Diskussion

### Kapitel 7: Deployment
- 7.1 Hardware-Optionen
- 7.2 Real-Time Anforderungen (10 kHz)
- 7.3 Praktische Herausforderungen
- 7.4 Ausblick

---

## ⚠️ KRITISCH: 10 kHz Real-Time Anforderung!

### **Das echte Problem: Timing**

Ihre Daten haben eine **10 kHz Abtastrate** (100 μs zwischen Samples):
```matlab
Ts = 1/10000;  // 100 μs = 0.1 ms
```

**Das bedeutet:**
- ✅ 10 Datenpunkte pro Millisekunde (nicht 1!)
- ⚠️ Control-Loop muss in < 100 μs abgeschlossen sein
- ⚠️ Typische SNN-Latenz (50 steps @ 0.1ms) = 5 ms = **50× zu langsam!**

### Problem 1: SNN Latenz >> Control-Loop Periode
**Symptom:** SNN braucht 5-10 ms, aber nur 100 μs verfügbar

**Lösungen:**

#### **Lösung 1A: Neuromorphe Hardware (Empfohlen)**
```
Intel Loihi 2: Asynchrone Verarbeitung
  → Spikes propagieren mit ~1 μs Latenz
  → Event-driven: Kein festes Zeitraster
  → Kann 10 kHz schaffen! ✅
```

#### **Lösung 1B: Drastisch reduzierte Zeitschritte**
```python
# Minimal SNN (nur 5-10 Schritte)
num_steps = 5  # Statt 50
dt = 10e-6     # 10 μs statt 100 μs
total_time = 50 μs  # Gerade noch machbar

# Trade-off: Weniger Genauigkeit, aber erfüllt Timing
```

#### **Lösung 1C: Kleineres Netzwerk**
```
Statt: 3 → 64 → 32 → 16 → 2
Nutze: 3 → 16 → 8 → 2
  
Vorteil: ~10× schneller
Nachteil: Evtl. weniger genau
```

#### **Lösung 1D: Downsampling (PRAGMATISCH)**
```
Original: 10 kHz (100 μs)
SNN läuft: 1 kHz (1 ms)
  
→ Nutze jeden 10. Datenpunkt
→ SNN hat 1 ms Zeit statt 100 μs
→ Linear interpoliere zwischen SNN-Outputs

Akzeptabel wenn: Motor-Dynamik langsamer als 1 kHz
```

#### **Lösung 1E: Hybrid ANN-SNN**
```cpp
// SNN läuft asynchron bei niedrigerer Rate
void control_loop_10kHz() {
    read_sensors();  // i_d, i_q, n
    
    if (snn_output_ready()) {
        u_d = snn_output.u_d;  // Energie-effizient
        u_q = snn_output.u_q;
        snn_output_ready = false;
        trigger_next_snn_inference();
    } else {
        // Fallback: Extrapoliere oder nutze ANN
        u_d = extrapolate_u_d();
        u_q = extrapolate_u_q();
    }
    
    apply_voltages(u_d, u_q);
}
```

### Problem 2: Genauigkeitsverlust > 20%
**Symptom:** SNN MAE >> ANN MAE

**Lösungen:**
- Mehr Simulationsschritte
- Parameter-Tuning (tau_mem, threshold)
- Bessere Konvertierungsmethode (z.B. Burst Coding)
- SNN direkt trainieren (statt konvertieren)

### Problem 3: Zu viele Spikes
**Symptom:** Hohe Energie trotz SNN

**Lösungen:**
- Threshold erhöhen
- Sparse Input Encoding
- Temporal Coding statt Rate Coding

---

## 📚 Weiterführende Literatur

### ANN-to-SNN Conversion
- Rueckauer et al. (2017): "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification"
- Diehl et al. (2015): "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing"

### LIF-Neuronen & SNNs
- Gerstner & Kistler (2002): "Spiking Neuron Models"
- Maass (1997): "Networks of Spiking Neurons: The Third Generation of Neural Network Models"

### Neuromorphe Hardware
- Davies et al. (2018): "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning"
- Furber et al. (2014): "The SpiNNaker Project"

---

## ✅ Zusammenfassung

**Ihr Ansatz ist sehr gut!**

✅ **Phase 1 (ANN):** Nutzt bewährte Training-Methoden
✅ **Phase 2 (SNN):** Erhält neuromorphe Vorteile
✅ **Datenmenge:** 2M Samples sind mehr als ausreichend
✅ **Forschungs-Relevanz:** Vergleich ANN vs SNN für Motor-Control

**Nächste Schritte:**
1. ✅ ANN in Edge Impulse trainieren (MAE < 1V)
2. ⏭️ Gewichte exportieren
3. ⏭️ SNN-Konvertierung (SNN Toolbox empfohlen für Start)
4. ⏭️ Performance-Vergleich dokumentieren
5. ⏭️ Thesis schreiben 🎓

**Erfolgsaussichten:** Sehr gut! 🚀

