# Vergleich Python vs MATLAB Simulation

## Parametervergleich

### Motor
| Parameter | MATLAB | Python |
|-----------|--------|--------|
| Polpaare (p) | 3 | 3 |
| R_s | 0.543 Ω | 0.543 Ω |
| L_d | 1.13 mH | 1.13 mH |
| L_q | 1.42 mH | 1.42 mH |
| Ψ_PM | 16.9 mWb | 16.9 mWb |
| I_max | 10.8 A | 10.8 A |
| V_DC | 48 V | 48 V |
| n_nenn | 3000 RPM | 3000 RPM |

### Simulation
| Parameter | MATLAB | Python |
|-----------|--------|--------|
| Abtastrate | 10 kHz | 10 kHz |
| Ts | 100 µs | 100 µs |
| Dauer | 0.2 s | 0.2 s |
| Schritte | 2000 | 2000 |

### Output-Format
Beide Simulationen erzeugen CSV-Dateien mit:
- `time`: Zeit [s]
- `i_d`, `i_q`: d/q-Ströme [A]
- `n`: Drehzahl [RPM]
- `u_d`, `u_q`: d/q-Spannungen [V]

## Skript-Varianten

### simulate_pmsm.py
Verwendet den GEM-Controller (`gem_controllers`), der automatisch parametriert wird.
Vorteil: Einfache Nutzung, robust.

### simulate_pmsm_matlab_match.py
Verwendet eigenen PI-Controller mit exakten MATLAB-Parametern:
- K_Pd = L_d / Ts = 11.3
- K_Id = R_s / Ts = 5430
- K_Pq = L_q / Ts = 14.2
- K_Iq = R_s / Ts = 5430
- Back-EMF Entkopplung aktiviert

Vorteil: Exakte Übereinstimmung mit MATLAB Simulink-Modell.

## Fazit

Beide Skripte produzieren vergleichbare Ergebnisse. Für maximale Übereinstimmung mit MATLAB sollte `simulate_pmsm_matlab_match.py` verwendet werden.
