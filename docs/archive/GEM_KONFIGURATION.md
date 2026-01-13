# GEM PMSM Simulation - Konfiguration und Setup

Diese Dokumentation beschreibt die vollständige Konfiguration der gym-electric-motor (GEM) Simulation für die Validierung gegen MATLAB/Simulink.

---

## Übersicht

Die GEM-Simulation wurde so konfiguriert, dass sie das MATLAB/Simulink PMSM-Modell möglichst genau nachbildet. Nach umfangreichen Debugging- und Anpassungsarbeiten erreicht der GEM Standard Controller exakt die gleichen Steady-State Stromwerte wie MATLAB.

---

## Verwendetes Environment

**Environment-ID:** `Cont-CC-PMSM-v0`

Dieses Environment ist für kontinuierliche Stromregelung (Current Control) eines Permanentmagnet-Synchronmotors ausgelegt. Die Referenzgrößen sind die dq-Ströme i_sd und i_sq. Das Environment stammt vollständig aus der GEM-Bibliothek und simuliert die komplette Motorelektrik inklusive dq-Transformation.

---

## Was von GEM bereitgestellt wird

**Physikalisches Motormodell:** GEM stellt ein vollständiges PMSM-Modell mit differentialgleichungsbasierter Simulation bereit. Das Modell berechnet die elektrischen Zustandsgrößen basierend auf den Motorparametern und dem aktuellen Rotorwinkel. Die Implementierung erfolgt über einen ODE-Solver.

**dq-Transformation:** Die Transformation zwischen abc- und dq-Koordinaten wird intern von GEM durchgeführt. Der Controller arbeitet im dq-System, die Ausgabe erfolgt in abc-Phasenspannungen. Die Transformation verwendet den aktuellen elektrischen Rotorwinkel epsilon.

**Normalisierung:** GEM normalisiert alle Zustandsgrößen auf den Bereich [-1, +1] basierend auf definierten Limits. Diese Normalisierung ist fundamental für die Verwendung mit Reinforcement Learning und muss bei der Denormalisierung berücksichtigt werden.

**Standard PI-Stromregler:** Die gem_controllers-Bibliothek stellt einen vorkonfigurierten PI-Stromregler bereit. Die Reglerparameter werden automatisch über Polplatzierung berechnet, wobei der Tuning-Parameter a=4 die Bandbreite bestimmt. Zusätzlich ist eine Strom-Sicherheitsmarge von 20% aktiv.

**Entkopplung:** Der GEM-Controller bietet eine optionale Entkopplung der d- und q-Achse zur Kompensation der Kreuzverkopplung. Diese ist aktiviert und verbessert das dynamische Verhalten.

**ConstantSpeedLoad:** GEM stellt verschiedene Last-Modelle bereit. Die ConstantSpeedLoad erzwingt eine konstante mechanische Drehzahl, was ideal für die Validierung der Stromregelung bei definierter Drehzahl ist.

---

## Was wir selbst konfiguriert haben

**Motorparameter aus MATLAB:** Die konkreten Motorparameter wurden dem MATLAB/Simulink-Modell entnommen:
- Polpaarzahl p = 3
- Statorwiderstand R_s = 0.543 Ω
- d-Induktivität L_d = 1.13 mH
- q-Induktivität L_q = 1.42 mH
- Permanentmagnet-Flussverkettung Ψ_PM = 16.9 mWb

**Limit-Werte entsprechend MATLAB:** Die Grenzen für Strom, Spannung und Drehzahl wurden auf das MATLAB-Setup abgestimmt:
- Maximaler Strom: 10.8 A
- Maximale Spannung: 48 V
- Maximale Drehzahl: 3000 rpm (314.16 rad/s)

**Direkte Motor-Objekt-Erstellung:** Da GEM übergebene Parameter als kwargs ignoriert, erstellen wir Motor- und Last-Objekte explizit mit der Klasse PermanentMagnetSynchronousMotor und übergeben diese an gem.make(). Dies ist eine notwendige Umgehung eines GEM-Designproblems.

**Korrekte Limit-Auslese:** Die tatsächlichen Limits werden vom Physical System ausgelesen und für die Denormalisierung verwendet. GEM dokumentiert dieses Vorgehen nicht explizit, es ist aber essentiell für korrekte Ergebnisse.

**Constraint-Deaktivierung:** Die standardmäßig aktiven Stromgrenzen-Constraints (SquaredConstraint) wurden deaktiviert, um unerwartete Simulationsabbrüche zu verhindern. Der Parameter constraints=() wird beim Environment-Erstellen übergeben.

**Robustes State-Auspacken:** Eine robuste Logik zur Verarbeitung des teils verschachtelten Tuple-Formats der GEM-Rückgabewerte wurde implementiert. Der step()-Aufruf gibt ein 5-Tuple zurück, dessen erstes Element selbst ein Tuple sein kann.

**Spannungs-Rücktransformation:** Für das Logging der dq-Spannungen wird eine inverse Park-Transformation durchgeführt, da GEM nur abc-Spannungen als Action ausgibt. Dies erfolgt unter Verwendung des aktuellen Rotorwinkels.

---

## Eigener MATLAB-kompatibler Controller

Zusätzlich zum GEM-Standard-Controller wurde ein eigener PI-Controller implementiert, der exakt die MATLAB-Struktur nachbildet.

**Reglerstruktur:** Klassischer PI-Regler mit P-Anteil und Euler-vorwärts-Integration für den I-Anteil, getrennt für d- und q-Achse.

**Gain-Berechnung nach Technischer Optimaleinstellung:** Die Gains werden berechnet wie in MATLAB, wobei T_s = 2 × tau die effektive Abtastzeit mit Totzeit-Kompensation ist:
- K_P = L / (2 × T_s)
- K_I = R_s / (2 × T_s)

**Entkopplungsterme entsprechend MATLAB-Blockdiagramm:**
- u_d erhält einen negativen Entkopplungsterm proportional zu ω_el × L_q × i_q
- u_q erhält einen positiven Entkopplungsterm proportional zu ω_el × (L_d × i_d + Ψ_PM)

---

## Testparameter für Validierung

### Drehzahl-Sweep (Baseline)

**Strom-Sollwerte:** i_d,ref = 0 A und i_q,ref = 2 A, sofort aktiv (step_time = 0)

**Getestete Drehzahlen:** 500 rpm, 1500 rpm, 2500 rpm

**Simulationsdauer:** 0.2 Sekunden (2000 Schritte bei tau = 100 µs)

**Abtastzeit:** tau = 0.0001 s (entspricht 10 kHz Regelfrequenz)

### Arbeitspunkt-Variation (Erweitert)

**Feste Drehzahl:** 1000 rpm

**Arbeitspunkt-Testmatrix:**

| Testfall | id [A] | iq [A] | |I| [A] | Beschreibung |
|----------|--------|--------|--------|--------------|
| 1 | 0 | 2 | 2.0 | Baseline (niedrige Last) |
| 2 | 0 | 5 | 5.0 | Mittlere Last |
| 3 | 0 | 8 | 8.0 | Hohe Last |
| 4 | -3 | 2 | 3.6 | Moderate Feldschwächung |
| 5 | -3 | 5 | 5.8 | Feldschwächung + mittlere Last |
| 6 | -5 | 5 | 7.1 | Stärkere Feldschwächung + Last |

**Zweck:** Validierung der Controller-Performance über den gesamten Arbeitspunktbereich, nicht nur bei id=0.

---

## Vergleich der beiden Python-Controller

### GEM Standard Controller

**Stärken:**
- Erreicht exakt die gleichen Steady-State Ströme wie MATLAB (MAE = 0.0000 A bei allen Drehzahlen)
- Automatisches Tuning basierend auf Motorparametern
- Robuste Implementierung mit integrierten Sicherheitsmechanismen
- Schnelles und stabiles Einschwingverhalten

**Schwächen:**
- Spannungen weichen von MATLAB ab (2.5 bis 9.8 V Offset je nach Drehzahl)
- Black-Box-Charakter, da die genaue Gain-Berechnung nicht direkt einsehbar ist
- Der Spannungsoffset skaliert mit der Drehzahl, was auf unterschiedliche Back-EMF-Kompensation hindeutet

### Eigener MATLAB-kompatibler Controller

**Stärken:**
- Vollständig transparente Implementierung
- Exakt nachvollziehbare Gains entsprechend der Technischen Optimaleinstellung
- Entkopplungsterme entsprechen dem MATLAB-Blockdiagramm

**Schwächen:**
- Erreicht die Sollströme nicht korrekt (MAE ≈ 2 A)
- Spannungen weichen stark ab (über 20 V Offset)
- Das Problem liegt vermutlich in der Action-Normalisierung oder der Interaktion mit GEM's internem Spannungs-Handling

---

## Empfehlung

Für die praktische Arbeit wie Trainingsdatengenerierung und RL-Training sollte der **GEM Standard Controller** verwendet werden, da er die Strom-Sollwerte perfekt erreicht. Der eigene Controller dient primär dem Verständnis der Regelungsstruktur und könnte mit weiterer Analyse an GEM angepasst werden.

---

## Quantitative Ergebnisse

| Drehzahl | i_q MAE (GEM Std) | i_q MAE (Eigener) | u_q MAE (GEM Std) | u_q MAE (Eigener) |
|----------|-------------------|-------------------|-------------------|-------------------|
| 500 rpm  | 0.0000 A          | 2.05 A            | 2.54 V            | 24.18 V           |
| 1500 rpm | 0.0000 A          | 1.95 A            | 6.15 V            | 22.79 V           |
| 2500 rpm | 0.0000 A          | 1.86 A            | 9.76 V            | 21.26 V           |

---

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `pmsm-pem/simulate_pmsm.py` | Simulation mit GEM Standard-Controller |
| `pmsm-pem/simulate_pmsm_matlab_match.py` | Simulation mit eigenem MATLAB-kompatiblem PI-Controller |
| `pmsm-pem/compare_simulations.py` | Vergleichsskript für MATLAB vs. GEM |
| `pmsm-pem/docs/GEM_Zusammenfassung.md` | Technische Zusammenfassung aller GEM-Erkenntnisse |

---

*Letzte Aktualisierung: Dezember 2025*

