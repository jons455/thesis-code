# PMSM Simulation mit GEM

Ziel: MATLAB/Simulink FOC-Simulation durch Python + gym-electric-motor (GEM) ersetzen, um Trainingsdaten für neuronale Netze zu generieren.


## Warum GEM?

Die bestehende MATLAB/Simulink-Toolchain funktioniert, hat aber Nachteile:
- MATLAB-Lizenz nötig
- Export/Import zwischen MATLAB und Python umständlich
- Für Neurobench-Integration wäre eine reine Python-Pipeline sauberer

**GEM kann das:** Das Package simuliert die komplette Motorstrecke (Plant) in Python. Eine PMSM mit FOC lässt sich damit nachbilden, ohne Simulink zu benötigen.

### Warum es klappen sollte

- Motorparameter (R, L, Trägheit, Polpaare) sind 1:1 übertragbar
- GEM ist OpenAI-Gym-kompatibel → passt direkt zu Neurobench
- Aktiv gepflegt, von Uni Paderborn entwickelt
- Physikalisch korrekte Modelle, nicht nur Approximationen

### Das Environment heißt nicht "FOC"

Wichtig: GEM benennt Environments nach der **Regelgröße**, nicht nach der Regelstrategie. FOC = Stromregelung im dq-System = **Current Control (CC)**.

```python
import gym_electric_motor as gem
from gem_controllers import GemController

# Environment = Ersatz für Simulink-Strecke (Plant)
env = gem.make(
    'Cont-CC-PMSM-v0',  # Continuous Current Control PMSM
    motor_parameter=dict(r_s=0.543, l_d=1.13e-3, l_q=1.42e-3, psi_p=16.9e-3, p=3),
    visualization=None
)

# PI-Controller als Benchmark (oder eigener, siehe unten)
controller = GemController.make(env, env_id='Cont-CC-PMSM-v0')

# Simulation Loop
state, reference = env.reset()
done = False
while not done:
    action = controller.control(state, reference)
    # Hier könnte auch ein SNN stehen: action = snn.predict(state, reference)
    (state, reference), reward, done, info = env.step(action)
```

### Links

| Resource | URL |
|----------|-----|
| GEM GitHub | https://github.com/upb-lea/gym-electric-motor |
| GEM Docs | https://upb-lea.github.io/gym-electric-motor/ |
| gem_controllers | https://github.com/upb-lea/gem-controllers |


## Eigener Controller vs. gem_controllers

Der mitgelieferte `gem_controllers` ist praktisch für schnelle Prototypen, aber eine Black-Box. Die PI-Gains werden intern berechnet und passen nicht unbedingt zu einem bestehenden MATLAB-Modell.

**Gute Nachricht:** Ein eigener Controller ist trivial. GEM liefert im State-Vektor alles was man braucht (i_d, i_q, omega, epsilon), und erwartet als Action nur die Stellspannungen. Dazwischen kann beliebige Logik stehen.

```python
class MatlabFOCController:
    def __init__(self, L_d, L_q, R_s, psi_pm, p, Ts):
        # Technische Optimaleinstellung
        self.K_Pd = L_d / (2 * Ts)
        self.K_Pq = L_q / (2 * Ts)
        self.K_Id = R_s / (2 * Ts)
        self.K_Iq = R_s / (2 * Ts)
        # ... Rest der Parameter
        self.integral_d = 0.0
        self.integral_q = 0.0

    def control(self, i_d, i_q, i_d_ref, i_q_ref, omega_el):
        # PI + Entkopplung, ~20 Zeilen Code
        e_d, e_q = i_d_ref - i_d, i_q_ref - i_q
        # ... PI-Berechnung + Back-EMF Kompensation
        return u_d, u_q
```

Das sind ca. 50 Zeilen Python. Siehe `simulate_pmsm_matlab_match.py` für die vollständige Implementierung.

**Warum GEM trotzdem?**

Auch mit eigenem Controller bleibt GEM wertvoll:

| Was | Selbst bauen? | GEM |
|-----|---------------|-----|
| PMSM Elektrodynamik | Komplex (DGL-System) | ✅ fertig |
| Park/Clarke Transformation | Machbar | ✅ fertig |
| Spannungsbegrenzung, Modulation | Fummelig | ✅ fertig |
| Verschiedene Motortypen (SCIM, BLDC, ...) | Jedes Mal neu | ✅ fertig |
| Gym-Interface für RL/Neurobench | Wrapper nötig | ✅ nativ |

Der Controller ist der einfache Teil. Die **Strecke** (Motor + Physik + Constraints) korrekt zu simulieren ist aufwändig. GEM nimmt einem genau das ab.


## Status

Aktuell: Validierung ob GEM dieselben Ergebnisse wie MATLAB liefert.

## TL;DR

| Aspekt | Status |
|--------|--------|
| Motorparameter | ✅ Abgeglichen |
| Simulationsparameter (Ts, Dauer) | ✅ Identisch |
| PI-Regler | ⚠️ Eigener Controller nötig |
| Drehzahl | ❌ Nicht direkt steuerbar in GEM CC-Env |
| Vorzeichen i_q | ❌ Invertiert gegenüber MATLAB |

**Kernproblem:** GEM's Current-Control Environment berechnet die Drehzahl aus der Motorphysik. In MATLAB wird sie als externer Parameter gesetzt. Direkter 1:1 Vergleich daher schwierig.


## Erkenntnisse

### GEM Controller ist eine Black-Box

Der Standard `gem_controllers` berechnet PI-Gains intern via Polplatzierung (Tuning-Parameter `a=4`). Das führt zu anderen Werten als im MATLAB-Modell. Für exakte Reproduktion muss ein eigener Controller her.

### Technische Optimaleinstellung

MATLAB verwendet diese Formeln für die PI-Gains:

```
K_Pd = L_d / (2*Ts) = 0.00113 / (2*0.0001) = 5.65
K_Pq = L_q / (2*Ts) = 0.00142 / (2*0.0001) = 7.10
K_Id = K_Iq = R_s / (2*Ts) = 0.543 / (2*0.0001) = 2715
```

Plus Entkopplungsterme:
- `u_d_dec = -ω_el * L_q * i_q`
- `u_q_dec = +ω_el * (L_d * i_d + Ψ_PM)`

### Drehzahl im CC-Environment

Das `Cont-CC-PMSM-v0` Environment ist für **reine Stromregelung** ausgelegt. Die Drehzahl ergibt sich aus:
- Elektrischem Drehmoment (abhängig von i_q)
- Lastmoment
- Trägheit

Startdrehzahl liegt bei ca. 716 RPM, nicht kontrollierbar über Reference-Generator.

### Step-Timing

MATLAB aktiviert Sollwerte erst bei t=0.1s via Step-Block. Das wurde in den Python-Skripten nachgebaut (`step_time_k = 1000` bei Ts=100µs).


## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `simulate_pmsm.py` | GEM mit Standard-Controller (gem_controllers) |
| `simulate_pmsm_matlab_match.py` | Eigener PI-Controller mit MATLAB-Parametern |
| `compare_simulations.py` | Vergleich MATLAB vs. Python Exports |


## Motorparameter

```python
motor_parameter = dict(
    p=3,              # Polpaare
    r_s=0.543,        # Statorwiderstand [Ω]
    l_d=0.00113,      # d-Induktivität [H]
    l_q=0.00142,      # q-Induktivität [H]
    psi_p=0.0169,     # Permanentmagnetfluss [Wb]
)

limit_values = dict(
    i=10.8,           # Maximalstrom [A]
    u=48.0,           # DC-Spannung [V]
    omega=3000 * 2 * np.pi / 60  # Max. Winkelgeschw. [rad/s]
)
```


## Vorgehen

### 1. Baseline herstellen

Bevor Vergleiche sinnvoll sind, muss geklärt werden ob die MATLAB Step-Blöcke feste Werte oder Workspace-Variablen nutzen. Sonst vergleicht man Äpfel mit Birnen.

**Offener Punkt:** Rückmeldung von Dennis zur Simulink-Verdrahtung.

### 2. Drehzahl angleichen

Für einen fairen Vergleich:
- Option A: MATLAB auf ~716 RPM setzen (GEM Startwert)
- Option B: Eigenes Motormodell in Python mit externer Drehzahl
- Option C: GEM Anfangsbedingungen anpassen (falls möglich)

### 3. Vorzeichen prüfen

i_q hat unterschiedliche Vorzeichen. Mögliche Ursachen:
- Koordinatenkonvention (Rechtssystem vs. Linkssystem)
- Stromrichtungsdefinition
- Transformationsmatrix (Park/Clarke)

Braucht Blick in die GEM-Doku oder den Sourcecode.

### 4. Validierung

Nach Angleichung:
1. Beide Simulationen mit identischen Parametern laufen lassen
2. `compare_simulations.py` ausführen
3. Plots und Metriken auswerten
4. Bei Abweichungen: Ursache isolieren

### 5. Trainingsdaten generieren

Wenn Baseline steht:
- Drehzahl über weiten Bereich variieren (wichtig wegen Back-EMF!)
- Verschiedene Strom-Sollwerte
- Evtl. Lastsprünge


## Offene Fragen

- [ ] Simulink: Feste Werte oder Workspace-Variablen?
- [ ] GEM: Kann man Anfangsdrehzahl beim Reset setzen?
- [ ] GEM: Woher kommt die Vorzeichen-Invertierung bei i_q?
- [ ] Brauchen wir Speed-Control Environment statt Current-Control?

