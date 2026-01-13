"""
PMSM Arbeitspunkt-Variation - Batch-Simulation
================================================
Führt Simulationen bei 1000 rpm mit verschiedenen id/iq-Kombinationen durch,
um die Modell-Performance über den gesamten Arbeitspunktbereich zu testen.

Physikalischer Hintergrund:
- iq: Bestimmt das Drehmoment (T ∝ ψ_PM × iq)
- id: Feldschwächung (negativ) für höhere Drehzahlen

Testmatrix:
| Testfall | id [A] | iq [A] | |I| [A] | Beschreibung                    |
|----------|--------|--------|--------|---------------------------------|
| 1        | 0      | 2      | 2.0    | Baseline (niedrige Last)        |
| 2        | 0      | 5      | 5.0    | Mittlere Last                   |
| 3        | 0      | 8      | 8.0    | Hohe Last                       |
| 4        | -3     | 2      | 3.6    | Moderate Feldschwächung         |
| 5        | -3     | 5      | 5.8    | Feldschwächung + mittlere Last  |
| 6        | -5     | 5      | 7.1    | Stärkere Feldschwächung + Last  |

Alle Tests bei 1000 rpm für konsistente Vergleichbarkeit.
"""

import subprocess
import sys
import os
from pathlib import Path
import numpy as np

# Konfiguration
SCRIPT_DIR = Path(__file__).parent
# Run 003: Andere Drehzahl
N_RPM = 1500  # Feste Drehzahl für alle Tests (vorher 1000)

# Arbeitspunkt-Testmatrix
# Run 003: Andere Kombinationen für breitere Validierung
OPERATING_POINTS = [
    (0.0, 1.0, "very_low_load"),           # Sehr geringe Last
    (0.0, 3.5, "mid_low_load"),            # Niedrige-mittlere Last
    (0.0, 6.0, "mid_high_load"),           # Mittlere-hohe Last
    (-2.0, 3.0, "fw_light"),               # Leichte Feldschwächung
    (-4.0, 4.0, "fw_balanced"),            # Balancierte Feldschwächung
    (-6.0, 3.0, "fw_strong_low_torque"),   # Starke FS, niedriges Drehmoment
]

# Maximaler Strom (zur Validierung)
I_MAX = 10.8  # A


def validate_operating_point(id_val: float, iq_val: float) -> bool:
    """Prüft ob der Arbeitspunkt innerhalb der Stromgrenzen liegt."""
    i_total = np.sqrt(id_val**2 + iq_val**2)
    if i_total > I_MAX:
        print(f"  [WARNUNG] |I| = {i_total:.2f} A > {I_MAX} A (Limit überschritten!)")
        return False
    return True


def run_simulation(script_name: str, id_ref: float, iq_ref: float, 
                   output_name: str, controller_type: str) -> bool:
    """Führt eine einzelne Simulation aus."""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"  [FEHLER] Skript nicht gefunden: {script_path}")
        return False
    
    cmd = [
        sys.executable, str(script_path),
        "--n-rpm", str(N_RPM),
        "--id-ref", str(id_ref),
        "--iq-ref", str(iq_ref),
        "--step-time", "0.0",  # Sofort aktiv
        "--output", output_name
    ]
    
    print(f"  [{controller_type}] id={id_ref:+.1f} A, iq={iq_ref:+.1f} A -> {output_name}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR)
        )
        if result.returncode != 0:
            print(f"    [FEHLER] {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"    [FEHLER] {e}")
        return False


def main():
    print("=" * 70)
    print("PMSM Arbeitspunkt-Variation - Batch-Simulation")
    print("=" * 70)
    print(f"Drehzahl: {N_RPM} rpm")
    print(f"Anzahl Arbeitspunkte: {len(OPERATING_POINTS)}")
    print()
    
    # Validiere alle Arbeitspunkte
    print("[1] Validiere Arbeitspunkte...")
    for id_val, iq_val, name in OPERATING_POINTS:
        i_total = np.sqrt(id_val**2 + iq_val**2)
        valid = "[OK]" if validate_operating_point(id_val, iq_val) else "[X]"
        print(f"  {valid} {name}: id={id_val:+.1f} A, iq={iq_val:+.1f} A, |I|={i_total:.2f} A")
    
    print()
    print("[2] Starte Simulationen...")
    
    success_count = 0
    total_count = 0
    
    # Simulationen mit GEM Standard Controller (verifiziert als MATLAB-äquivalent)
    print("\n--- GEM Standard Controller ---")
    for id_val, iq_val, name in OPERATING_POINTS:
        output_name = f"op_n{N_RPM:04d}_id{int(id_val):+03d}_iq{int(iq_val):+03d}.csv"
        total_count += 1
        if run_simulation("simulate_pmsm.py", id_val, iq_val, output_name, "GEM Std"):
            success_count += 1
    
    # Eigener Controller deaktiviert - hat in Run 001/002 nicht funktioniert
    # print("\n--- Eigener MATLAB Controller ---")
    # for id_val, iq_val, name in OPERATING_POINTS:
    #     output_name = f"op_n{N_RPM:04d}_id{int(id_val):+03d}_iq{int(iq_val):+03d}.csv"
    #     total_count += 1
    #     if run_simulation("simulate_pmsm_matlab_match.py", id_val, iq_val, output_name, "MATLAB Ctrl"):
    #         success_count += 1
    
    print()
    print("=" * 70)
    print(f"FERTIG! {success_count}/{total_count} Simulationen erfolgreich")
    print("=" * 70)
    PMSM_PEM_DIR = SCRIPT_DIR.parent  # pmsm-pem/
    print()
    print("Ergebnisse unter:")
    print(f"  GEM Standard: {PMSM_PEM_DIR / 'export' / 'gem_standard'}")
    print(f"  MATLAB Match: {PMSM_PEM_DIR / 'export' / 'matlab_match'}")
    print()
    print("Nächster Schritt: Vergleich mit validation/compare_operating_points.py")
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

