# PMSM Python Simulation

Python-basierte PMSM FOC Simulation mit `gym-electric-motor` als Alternative zur MATLAB/Simulink Simulation.

## Setup

```powershell
cd pmsm-pem
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Nutzung

### Mit GEM-Controller (empfohlen)
```powershell
python simulate_pmsm.py
```

### Mit MATLAB-kompatiblem PI-Controller
```powershell
python simulate_pmsm_matlab_match.py
```

Beide Skripte erzeugen CSV-Dateien in `export/train/`.

## Motorparameter

| Parameter | Wert |
|-----------|------|
| Polpaare | 3 |
| R_s | 0.543 Ω |
| L_d | 1.13 mH |
| L_q | 1.42 mH |
| Ψ_PM | 16.9 mWb |
| I_max | 10.8 A |
| V_DC | 48 V |
| n_nenn | 3000 RPM |

## Output

CSV-Dateien mit Spalten:
- `time`: Zeit [s]
- `i_d`, `i_q`: d/q-Ströme [A]
- `n`: Drehzahl [RPM]
- `u_d`, `u_q`: d/q-Spannungen [V]

Siehe `VERGLEICH_MATLAB.md` für Details zum Unterschied zwischen den Skripten.
