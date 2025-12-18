# PMSM MATLAB/Simulink Simulation

MATLAB/Simulink-basierte FOC-Simulation eines Permanentmagnet-Synchronmotors (PMSM).

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `foc_pmsm.slx` | Simulink-Modell mit PI-Stromregler und Back-EMF-Entkopplung |
| `pmsm_init.m` | Initialisierungsskript mit Motorparametern |
| `pmsm_validation_compare.m` | Drehzahl-Sweep (500/1500/2500 rpm) für Python-Validierung |
| `pmsm_operating_points.m` | Arbeitspunkt-Variation (verschiedene id/iq bei 1000 rpm) |

## Schnellstart

```matlab
% 1. Initialisierung
pmsm_init

% 2. Validierungssimulation (Drehzahl-Sweep)
pmsm_validation_compare

% 3. Arbeitspunkt-Tests (id/iq-Variation)
pmsm_operating_points
```

## Export

Ergebnisse werden nach `export/` exportiert:
- `export/validation/` - CSV-Dateien für Python-Vergleich
- `export/train/` - Trainingsdaten

## Motorparameter

```matlab
R_s = 0.543;      % Statorwiderstand [Ω]
L_d = 0.00113;    % d-Induktivität [H]
L_q = 0.00142;    % q-Induktivität [H]
Psi_PM = 0.0169;  % PM-Fluss [Wb]
p = 3;            % Polpaare
Ts = 1/10000;     % Abtastzeit [s]
```

