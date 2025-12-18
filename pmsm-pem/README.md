# PMSM Simulation mit GEM

Ziel: MATLAB/Simulink FOC-Simulation durch Python + gym-electric-motor (GEM) ersetzen, um Trainingsdaten für neuronale Netze zu generieren.

## Ordnerstruktur

```
pmsm-pem/
├── simulation/                    # Simulationsskripte
│   ├── simulate_pmsm.py           # GEM mit Standard-Controller
│   ├── simulate_pmsm_matlab_match.py  # Eigener MATLAB-kompatibler PI-Controller
│   └── run_operating_point_tests.py   # Batch-Simulation verschiedener Arbeitspunkte
├── validation/                    # Validierungs- und Vergleichsskripte
│   ├── compare_simulations.py     # Vergleich MATLAB vs. GEM (Drehzahl-Sweep)
│   ├── compare_operating_points.py    # Vergleich verschiedener id/iq-Arbeitspunkte
│   └── debug_script.py            # Debug-Hilfsskript
├── export/                        # Simulationsergebnisse (CSV + Plots)
│   ├── gem_standard/              # Ergebnisse mit GEM Standard-Controller
│   ├── matlab_match/              # Ergebnisse mit eigenem Controller
│   └── comparison/                # Vergleichsplots und Metriken
├── docs/                          # Technische Dokumentation
│   └── GEM_Zusammenfassung.md
└── venv/                          # Python Virtual Environment
```


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


## Schnellstart

```powershell
cd pmsm-pem
.\venv\Scripts\activate

# Einzelne Simulation
python simulation/simulate_pmsm.py --n-rpm 1000 --iq-ref 5

# Batch-Simulation verschiedener Arbeitspunkte
python simulation/run_operating_point_tests.py

# Validierung gegen MATLAB
python validation/compare_simulations.py
python validation/compare_operating_points.py
```


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

Das sind ca. 50 Zeilen Python. Siehe `simulation/simulate_pmsm_matlab_match.py` für die vollständige Implementierung.

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

| Aspekt | Status |
|--------|--------|
| Motorparameter | ✅ Abgeglichen |
| Simulationsparameter (Ts, Dauer) | ✅ Identisch |
| PI-Regler | ✅ GEM Standard erreicht exakt MATLAB Steady-State |
| Drehzahl | ✅ Über ConstantSpeedLoad steuerbar |
| Vorzeichen i_q | ✅ Gelöst |


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


## Arbeitspunkt-Testmatrix

Für den Vergleich bei 1000 rpm mit verschiedenen id/iq-Kombinationen:

| Testfall | id [A] | iq [A] | |I| [A] | Beschreibung |
|----------|--------|--------|--------|--------------|
| 1 | 0 | 2 | 2.0 | Baseline (niedrige Last) |
| 2 | 0 | 5 | 5.0 | Mittlere Last |
| 3 | 0 | 8 | 8.0 | Hohe Last |
| 4 | -3 | 2 | 3.6 | Moderate Feldschwächung |
| 5 | -3 | 5 | 5.8 | Feldschwächung + mittlere Last |
| 6 | -5 | 5 | 7.1 | Stärkere Feldschwächung + Last |


## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `simulation/simulate_pmsm.py` | GEM mit Standard-Controller (gem_controllers) |
| `simulation/simulate_pmsm_matlab_match.py` | Eigener PI-Controller mit MATLAB-Parametern |
| `simulation/run_operating_point_tests.py` | Batch-Simulation aller Arbeitspunkte |
| `validation/compare_simulations.py` | Vergleich MATLAB vs. Python (Drehzahl-Sweep) |
| `validation/compare_operating_points.py` | Vergleich verschiedener Arbeitspunkte |


## Technische Dokumentation

Siehe `docs/` für detaillierte technische Dokumentation:
- `GEM_Zusammenfassung.md` - Alle GEM-spezifischen Erkenntnisse
- `../docs/GEM_KONFIGURATION.md` - Vollständige Setup-Dokumentation
- `../docs/GEM_LEARNINGS.md` - Learnings aus der Validierungsarbeit
