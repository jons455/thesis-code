# PMSM FOC Neural Network Training

Training neuronaler Netze zur Nachbildung eines FOC-Reglers für PMSM.

## Struktur

```
thesis-code/
├── pmsm-matlab/              # MATLAB/Simulink Simulation
│   ├── foc_pmsm.slx          # Simulink FOC-Modell
│   ├── pmsm_init.m           # Motorparameter
│   ├── pmsm_validation_compare.m   # Drehzahl-Sweep
│   └── pmsm_operating_points.m     # Arbeitspunkt-Tests
│
├── pmsm-pem/                 # Python Simulation (GEM-basiert)
│   ├── simulation/           # Simulationsskripte
│   │   ├── simulate_pmsm.py              # GEM Standard-Controller
│   │   ├── simulate_pmsm_matlab_match.py # Eigener PI-Controller
│   │   └── run_operating_point_tests.py  # Batch-Simulation
│   ├── validation/           # Vergleichsskripte
│   │   ├── compare_simulations.py        # Drehzahl-Vergleich
│   │   └── compare_operating_points.py   # Arbeitspunkt-Vergleich
│   └── export/               # Ergebnisse (CSV, Plots)
│
├── data-preperation/         # Datenaufbereitung
│   ├── merge_simulation_data.py
│   ├── prepare_edge_impulse.py
│   └── data_exploration.ipynb
│
└── docs/                     # Dokumentation
    ├── GEM_KONFIGURATION.md  # GEM Setup & Konfiguration
    ├── GEM_LEARNINGS.md      # Learnings & Problemlösungen
    ├── TRAINING_GUIDE.md     # Trainingsanleitung
    └── EDGE_IMPULSE_GUIDE.md # Edge Impulse Integration
```

## Motor

| Parameter | Wert |
|-----------|------|
| Polpaare | 3 |
| R_s | 0.543 Ω |
| L_d | 1.13 mH |
| L_q | 1.42 mH |
| Ψ_PM | 16.9 mWb |
| I_max | 10.8 A |
| V_DC | 48 V |
| n_max | 3000 RPM |

## Validierungsergebnisse

Der **GEM Standard Controller** erreicht exaktes Tracking über alle getesteten Arbeitspunkte:

| Test | id [A] | iq [A] | Tracking-Fehler |
|------|--------|--------|-----------------|
| Baseline | 0 | 2 | 0.0000 A |
| Mittlere Last | 0 | 5 | 0.0000 A |
| Hohe Last | 0 | 8 | 0.0000 A |
| Feldschwächung | -3 | 2 | 0.0000 A |
| Feldschw. + Last | -3 | 5 | 0.0000 A |
| Starke Feldschw. | -5 | 5 | 0.0000 A |

## Quick Start

### MATLAB Simulation
```matlab
cd pmsm-matlab
pmsm_init
pmsm_validation_compare    % Drehzahl-Sweep
pmsm_operating_points      % Arbeitspunkt-Tests
```

### Python Simulation
```powershell
cd pmsm-pem
.\venv\Scripts\activate
python simulation/run_operating_point_tests.py   # Simulation
python validation/compare_operating_points.py    # Vergleich
```

### Datenaufbereitung
```bash
cd data-preperation
pip install -r requirements.txt
python merge_simulation_data.py
python prepare_edge_impulse.py
```

## Simulation

- Abtastrate: 10 kHz (Ts = 100 µs)
- Dauer: 0.2 s pro Run
- Schritte: 2000 pro Run

## Datenformat

**Features:** `i_d`, `i_q` [A], `n` [RPM]  
**Targets:** `u_d`, `u_q` [V]

Siehe `docs/` für Details.
