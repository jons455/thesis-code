# Zusammenfassung: Arbeit mit gym-electric-motor (GEM)

## 1. Verwendete Environments

| Environment-ID | Typ | Referenz-Größen | Status |
|----------------|-----|-----------------|--------|
| `Cont-CC-PMSM-v0` | Current Control | `i_sd`, `i_sq` | ✅ Hauptsächlich verwendet |
| `Cont-SC-PMSM-v0` | Speed Control | `omega` | ⚠️ Existiert, nicht getestet |
| `Cont-TC-PMSM-v0` | Torque Control | Drehmoment | ⚠️ Existiert, nicht getestet |

---

## 2. Kernerkenntnisse zu GEM

### 2.1 Drehzahl ist im CC-Env nicht direkt steuerbar
- Das `Cont-CC-PMSM-v0` Environment berechnet die Drehzahl aus der **Motorphysik**
- **Lösung**: `ConstantSpeedLoad` als `load`-Parameter übergeben

```python
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad

omega_fixed = n_rpm * 2 * np.pi / 60  # rad/s
load = ConstantSpeedLoad(omega_fixed=omega_fixed)
```

### 2.2 GEM ignoriert `motor_parameter` und `limit_values` als kwargs!
**KRITISCHER BUG**: Wenn man `motor_parameter=...` und `limit_values=...` an `gem.make()` übergibt, werden diese **IGNORIERT**! GEM verwendet stattdessen die Default-Werte.

**Lösung**: Motor-Objekt **direkt** erstellen und übergeben:

```python
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor

# Motor mit eigenen Parametern erstellen
motor = PermanentMagnetSynchronousMotor(
    motor_parameter=dict(p=3, r_s=0.543, l_d=0.00113, l_q=0.00142, psi_p=0.0169),
    limit_values=dict(i=10.8, u=48.0, omega=3000 * 2 * np.pi / 60),
)

# An gem.make() übergeben
env = gem.make('Cont-CC-PMSM-v0', motor=motor, load=load, tau=1e-4, ...)
```

### 2.3 Constraints können done=True verursachen
Das `Cont-CC-PMSM-v0` Environment hat standardmäßig `SquaredConstraint(("i_sq", "i_sd"))`. Wenn die Strom-Limits überschritten werden, wird `done=True` gesetzt.

**Lösung**: Constraints deaktivieren:
```python
env = gem.make('Cont-CC-PMSM-v0', ..., constraints=())
```

### 2.4 Zustände sind normalisiert
GEM liefert alle States normalisiert auf [-1, +1]:
- `i_sd_norm = i_sd / i_limit`
- `omega_norm = omega / omega_limit`
- `epsilon_norm = epsilon / pi`

Die Limits können vom Physical System gelesen werden:
```python
ps = env_unwrapped.physical_system
limits = {name: ps.limits[i] for i, name in enumerate(ps.state_names)}
```

### 2.5 State-Indizes (für `Cont-CC-PMSM-v0`)
```python
state_names = ['omega', 'torque', 'i_a', 'i_b', 'i_c', 'i_sd', 'i_sq',
               'u_a', 'u_b', 'u_c', 'u_sd', 'u_sq', 'epsilon', 'u_sup']
# Typische Indizes: omega=0, i_sd=5, i_sq=6, epsilon=12
```

### 2.6 step() gibt Tuple zurück
Nach `env.step(action)` ist das Ergebnis ein 5-Tuple, aber das erste Element (`observation`) ist selbst ein Tuple!

```python
step_result = env.step(action)  # 5-Tuple
state, reward, done, truncated, info = step_result
# ACHTUNG: state ist ein Tuple, nicht numpy array!

# Richtig auspacken:
if isinstance(state, tuple):
    state = state[0]
```

---

## 3. GEM Standard-Controller (`gem_controllers`)

### 3.1 Erstellung
```python
import gem_controllers as gc

controller = gc.GemController.make(
    env_unwrapped,
    'Cont-CC-PMSM-v0',
    decoupling=True,
    current_safety_margin=0.2,
    base_current_controller="PI",
    a=4,  # Bandbreiten-Parameter für Polplatzierung
    block_diagram=False,
)
```

### 3.2 Verwendung
```python
# Controller.reset() nach env.reset()!
controller.reset()

# Referenz als normalisierte Werte
ref_normalized = np.array([i_d_ref / i_limit, i_q_ref / i_limit])

# Control
action = controller.control(state_normalized, ref_normalized)

# Action ist normalisiert, direkt an env.step() übergeben
```

### 3.3 InputStage denormalisiert die Referenz
Der GEM-Controller hat eine `InputStage`, die die normalisierte Referenz denormalisiert:
- Referenz muss normalisiert übergeben werden (z.B. `2.0 A / 10.8 A = 0.185`)
- Die InputStage multipliziert mit `env.limits[reference_indices]`

---

## 4. Eigener MATLAB-kompatibler Controller

### 4.1 PI-Regler mit Entkopplung (wie MATLAB)
Implementiert in `simulate_pmsm_matlab_match.py`:

```python
class PIController:
    def __init__(self, Kp_d, Ki_d, Kp_q, Ki_q, L_d, L_q, Psi_PM, tau):
        self.Kp_d, self.Ki_d = Kp_d, Ki_d
        self.Kp_q, self.Ki_q = Kp_q, Ki_q
        self.L_d, self.L_q, self.Psi_PM = L_d, L_q, Psi_PM
        self.tau = tau
        self.integral_d = 0.0
        self.integral_q = 0.0

    def control(self, i_d, i_q, i_d_ref, i_q_ref, omega_elec):
        # Regelfehler
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # Integratoren (Euler vorwärts)
        self.integral_d += self.Ki_d * e_d * self.tau
        self.integral_q += self.Ki_q * e_q * self.tau

        # PI-Ausgang
        u_d_pi = self.Kp_d * e_d + self.integral_d
        u_q_pi = self.Kp_q * e_q + self.integral_q

        # Entkopplung
        u_d = u_d_pi - omega_elec * self.L_q * i_q
        u_q = u_q_pi + omega_elec * (self.L_d * i_d + self.Psi_PM)

        return u_d, u_q
```

### 4.2 Gain-Berechnung (Technische Optimaleinstellung)
Aus dem MATLAB Simulink-Modell:

```python
Ts = 2 * tau  # Abtastzeit (2x Controller-Periode für Totzeit)

K_Pd = L_d / (2 * Ts)  # = 0.00113 / 0.0002 = 5.65
K_Pq = L_q / (2 * Ts)  # = 0.00142 / 0.0002 = 7.10
K_Id = K_Iq = R_s / (2 * Ts)  # = 0.543 / 0.0002 = 2715
```

---

## 5. Bekannte Bugs und Probleme

### 5.1 ❌ Drehzahl-Offset (BEHOBEN)
**Problem**: GEM zeigte 25% weniger rpm als angefordert.
**Ursache**: `limit_values` wurden nicht ans Physical System weitergereicht, falsche Denormalisierung.
**Lösung**: Motor direkt erstellen und `actual_limits` vom Physical System lesen.

### 5.2 ⚠️ Oszillationen im Hauptskript
**Problem**: Das Hauptskript `simulate_pmsm.py` produziert oszillierende Ströme, während ein isoliertes Debug-Skript perfekt funktioniert.
**Status**: Noch nicht vollständig gelöst. Der isolierte Code funktioniert, aber im Hauptskript driftet der Controller ab.
**Workaround**: Debug-Skript verwenden oder Hauptskript entsprechend anpassen.

### 5.3 ⚠️ done=True bei Limit-Überschreitung
**Problem**: Wenn Constraints aktiv sind, kann die Simulation terminieren.
**Lösung**: `constraints=()` beim Environment setzen.

---

## 6. MATLAB-Controller Struktur (aus Simulink-Bild)

### 6.1 Eingänge
- Port 1: `i_d*` (d-Achsen Sollstrom)
- Port 2: `i_d` (d-Achsen Iststrom)
- Port 3: `i_q*` (q-Achsen Sollstrom)
- Port 4: `i_q` (q-Achsen Iststrom)
- Port 5: `n` → umgerechnet zu `omega_el = n * 2*pi/60 * p`

### 6.2 Regelstruktur
```
e_d = i_d* - i_d
e_q = i_q* - i_q

u_d_pi = K_Pd * e_d + K_Id * ∫e_d dt
u_q_pi = K_Pq * e_q + K_Iq * ∫e_q dt

u_d = u_d_pi - omega_el * L_q * i_q          (Entkopplung)
u_q = u_q_pi + omega_el * (L_d * i_d + Psi_PM)  (Entkopplung)
```

### 6.3 Gain-Berechnung (rechte Seite des Diagramms)
```
K_Pq = L_q / (2 * Ts)
K_Pd = L_d / (2 * Ts)
K_Id = K_Iq = R_s / (2 * Ts)
```

---

## 7. Empfehlungen

1. **Motor immer direkt erstellen** und an `gem.make()` übergeben
2. **Constraints deaktivieren** wenn volle Kontrolle gewünscht
3. **Limits vom Physical System lesen** für korrekte Denormalisierung
4. **controller.reset() nach env.reset()** aufrufen
5. **Zustand korrekt auspacken** (Tuple-Handling)

---

## 8. Dateiübersicht

| Datei | Beschreibung |
|-------|--------------|
| `simulate_pmsm.py` | Simulation mit GEM Standard-Controller |
| `simulate_pmsm_matlab_match.py` | Simulation mit eigenem PI-Controller (MATLAB-Parameter) |
| `compare_simulations.py` | Vergleich von MATLAB, GEM Standard und GEM Custom |
| `debug_script.py` | Isoliertes Debug-Skript (funktioniert!) |

---

*Letzte Aktualisierung: 18. Dezember 2025*
