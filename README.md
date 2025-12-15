# PMSM FOC Neural Network Training

Training neuronaler Netze zur Nachbildung eines FOC-Reglers für PMSM.

## Struktur

```
thesis-code/
├── pmsm-matlab/          # MATLAB/Simulink Simulation
│   ├── foc_pmsm.slx
│   └── pmsm_init.m
│
├── pmsm-pem/             # Python Simulation (GEM-basiert)
│   ├── simulate_pmsm.py              # Mit GEM Controller
│   └── simulate_pmsm_matlab_match.py # Mit MATLAB-Parametern
│
├── data-preperation/     # Datenaufbereitung
│   ├── merge_simulation_data.py
│   ├── prepare_edge_impulse.py
│   └── data_exploration.ipynb
│
└── docs/                 # Dokumentation
```

## Motor

| Parameter | Wert |
|-----------|------|
| Polpaare | 3 |
| R_s | 0.543 Ω |
| L_d | 1.13 mH |
| L_q | 1.42 mH |
| Ψ_PM | 16.9 mWb |
| I_nenn | 4.2 A |
| I_max | 10.8 A |
| V_DC | 48 V |
| n_nenn | 3000 RPM |

## Datenformat

**Features:** `i_d`, `i_q` [A], `n` [RPM]  
**Targets:** `u_d`, `u_q` [V]

## Quick Start

### Option A: MATLAB Simulation
```matlab
cd pmsm-matlab
pmsm_init
```

### Option B: Python Simulation
```powershell
cd pmsm-pem
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python simulate_pmsm.py
```

### Datenaufbereitung
```bash
cd data-preperation
pip install -r requirements.txt
python merge_simulation_data.py
python prepare_edge_impulse.py
```

## Simulation

- Abtastrate: 10 kHz
- Dauer: 0.2 s pro Run
- Schritte: 2000 pro Run

Siehe `docs/` für Details zum Training.
