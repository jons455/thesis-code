"""
Vergleich MATLAB vs. Python (GEM) PMSM-Simulation
==================================================
Dieses Skript vergleicht die Simulationsergebnisse aus MATLAB und Python,
um die Übereinstimmung der Implementierungen zu validieren.

WICHTIG: simulate_pmsm.py wird NICHT verändert!
         Dieses Skript liest nur die exportierten CSV-Dateien.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# ============================================================================
# KONFIGURATION
# ============================================================================

# Pfade zu den CSV-Dateien (relativ zum Skript-Verzeichnis)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# MATLAB-CSV (aus pmsm_validation_compare.m)
MATLAB_CSV = PROJECT_ROOT / "pmsm-matlab" / "export" / "validation" / "validation_sim.csv"

# Python-CSV (letzter Export aus simulate_pmsm.py)
PYTHON_CSV = SCRIPT_DIR / "export" / "train" / "sim_0001.csv"

# Signale zum Vergleich
SIGNALS_TO_COMPARE = ['i_d', 'i_q', 'n', 'u_d', 'u_q']

# Toleranzen für den Vergleich
TOLERANCE_ABSOLUTE = 0.1    # Absolute Toleranz
TOLERANCE_RELATIVE = 0.05   # Relative Toleranz (5%)


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def load_csv(filepath: Path, name: str) -> pd.DataFrame | None:
    """Lädt eine CSV-Datei und gibt sie als DataFrame zurück."""
    if not filepath.exists():
        print(f"[FEHLER] {name}-Datei nicht gefunden: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"[OK] {name}-Daten geladen: {len(df)} Zeilen, Spalten: {list(df.columns)}")
    return df


def resample_to_common_time(df1: pd.DataFrame, df2: pd.DataFrame, 
                            time_col: str = 'time') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resampled beide DataFrames auf eine gemeinsame Zeitbasis."""
    # Gemeinsamer Zeitbereich
    t_start = max(df1[time_col].min(), df2[time_col].min())
    t_end = min(df1[time_col].max(), df2[time_col].max())
    
    # Kleinere Schrittweite verwenden
    dt1 = np.median(np.diff(df1[time_col]))
    dt2 = np.median(np.diff(df2[time_col]))
    dt = min(dt1, dt2)
    
    t_common = np.arange(t_start, t_end, dt)
    
    # Interpolieren
    df1_resampled = pd.DataFrame({'time': t_common})
    df2_resampled = pd.DataFrame({'time': t_common})
    
    for col in df1.columns:
        if col != time_col and col in df2.columns:
            df1_resampled[col] = np.interp(t_common, df1[time_col], df1[col])
            df2_resampled[col] = np.interp(t_common, df2[time_col], df2[col])
    
    return df1_resampled, df2_resampled


def compute_metrics(matlab_data: np.ndarray, python_data: np.ndarray) -> dict:
    """Berechnet Vergleichsmetriken zwischen zwei Signalen."""
    error = matlab_data - python_data
    
    metrics = {
        'MAE': np.mean(np.abs(error)),                          # Mean Absolute Error
        'RMSE': np.sqrt(np.mean(error**2)),                     # Root Mean Square Error
        'MaxError': np.max(np.abs(error)),                      # Maximaler Fehler
        'MeanMATLAB': np.mean(matlab_data),                     # Mittelwert MATLAB
        'MeanPython': np.mean(python_data),                     # Mittelwert Python
        'StdMATLAB': np.std(matlab_data),                       # Standardabw. MATLAB
        'StdPython': np.std(python_data),                       # Standardabw. Python
    }
    
    # Relative Fehler (wenn Mittelwert nicht ~0)
    if np.abs(metrics['MeanMATLAB']) > 1e-6:
        metrics['RelativeError'] = metrics['MAE'] / np.abs(metrics['MeanMATLAB'])
    else:
        metrics['RelativeError'] = np.nan
    
    return metrics


def check_match(metrics: dict, signal_name: str) -> bool:
    """Prüft ob die Metriken innerhalb der Toleranzen liegen."""
    mae_ok = metrics['MAE'] < TOLERANCE_ABSOLUTE
    
    if not np.isnan(metrics['RelativeError']):
        rel_ok = metrics['RelativeError'] < TOLERANCE_RELATIVE
    else:
        rel_ok = True
    
    passed = mae_ok and rel_ok
    status = "[PASS]" if passed else "[FAIL]"
    
    print(f"  {signal_name:8s}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
          f"MaxErr={metrics['MaxError']:.4f}, RelErr={metrics['RelativeError']:.2%} -> {status}")
    
    return passed


# ============================================================================
# VISUALISIERUNG
# ============================================================================

def plot_comparison(df_matlab: pd.DataFrame, df_python: pd.DataFrame, 
                   signals: list, output_path: Path | None = None):
    """Erstellt Vergleichsplots für alle Signale."""
    
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(14, 3*n_signals))
    fig.suptitle('MATLAB vs. Python (GEM) Simulation - Vergleich', fontsize=14, fontweight='bold')
    
    colors = {'matlab': '#1f77b4', 'python': '#ff7f0e', 'error': '#d62728'}
    
    for idx, signal in enumerate(signals):
        if signal not in df_matlab.columns or signal not in df_python.columns:
            print(f"[WARNUNG] Signal '{signal}' nicht in beiden Datensätzen vorhanden.")
            continue
        
        t = df_matlab['time']
        y_matlab = df_matlab[signal]
        y_python = df_python[signal]
        error = y_matlab - y_python
        
        # Linkes Plot: Signalvergleich
        ax_left = axes[idx, 0] if n_signals > 1 else axes[0]
        ax_left.plot(t * 1000, y_matlab, color=colors['matlab'], label='MATLAB', linewidth=1.5)
        ax_left.plot(t * 1000, y_python, color=colors['python'], label='Python (GEM)', 
                     linewidth=1.5, linestyle='--')
        ax_left.set_ylabel(signal)
        ax_left.legend(loc='upper right')
        ax_left.grid(True, alpha=0.3)
        ax_left.set_title(f'{signal} - Signalvergleich')
        
        # Rechtes Plot: Fehler
        ax_right = axes[idx, 1] if n_signals > 1 else axes[1]
        ax_right.plot(t * 1000, error, color=colors['error'], linewidth=1)
        ax_right.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_right.axhline(y=TOLERANCE_ABSOLUTE, color='green', linestyle='--', 
                        linewidth=1, label=f'±{TOLERANCE_ABSOLUTE} Toleranz')
        ax_right.axhline(y=-TOLERANCE_ABSOLUTE, color='green', linestyle='--', linewidth=1)
        ax_right.set_ylabel(f'Δ{signal}')
        ax_right.legend(loc='upper right')
        ax_right.grid(True, alpha=0.3)
        ax_right.set_title(f'{signal} - Differenz (MATLAB - Python)')
    
    # X-Achsen-Label nur unten
    if n_signals > 1:
        axes[-1, 0].set_xlabel('Zeit [ms]')
        axes[-1, 1].set_xlabel('Zeit [ms]')
    else:
        axes[0].set_xlabel('Zeit [ms]')
        axes[1].set_xlabel('Zeit [ms]')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Plot gespeichert: {output_path}")
    
    plt.show()


def plot_phase_portrait(df_matlab: pd.DataFrame, df_python: pd.DataFrame,
                        output_path: Path | None = None):
    """Erstellt id-iq Phasenportrait für beide Simulationen."""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(df_matlab['i_d'], df_matlab['i_q'], 
            color='#1f77b4', label='MATLAB', linewidth=1.5, alpha=0.8)
    ax.plot(df_python['i_d'], df_python['i_q'], 
            color='#ff7f0e', label='Python (GEM)', linewidth=1.5, linestyle='--', alpha=0.8)
    
    # Start- und Endpunkte markieren
    ax.plot(df_matlab['i_d'].iloc[0], df_matlab['i_q'].iloc[0], 
            'o', color='#1f77b4', markersize=10, label='Start MATLAB')
    ax.plot(df_python['i_d'].iloc[0], df_python['i_q'].iloc[0], 
            's', color='#ff7f0e', markersize=10, label='Start Python')
    
    ax.set_xlabel('i_d [A]')
    ax.set_ylabel('i_q [A]')
    ax.set_title('Phasenportrait i_d vs. i_q')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Phasenportrait gespeichert: {output_path}")
    
    plt.show()


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    print("=" * 70)
    print("PMSM Simulation Vergleich: MATLAB vs. Python (GEM)")
    print("=" * 70)
    print()
    
    # Output-Verzeichnis erstellen
    output_dir = SCRIPT_DIR / "export" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Daten laden
    print("[1] Lade Simulationsdaten...")
    df_matlab = load_csv(MATLAB_CSV, "MATLAB")
    df_python = load_csv(PYTHON_CSV, "Python")
    
    if df_matlab is None or df_python is None:
        print("\n[FEHLER] Konnte nicht alle Daten laden. Abbruch.")
        print("\nHinweise:")
        print(f"  - MATLAB-Datei erwartet: {MATLAB_CSV}")
        print(f"  - Python-Datei erwartet: {PYTHON_CSV}")
        print("  - Führe zuerst pmsm_validation_compare.m in MATLAB aus!")
        print("  - Führe dann simulate_pmsm.py aus!")
        sys.exit(1)
    
    # Auf gemeinsame Zeitbasis bringen
    print("\n[2] Synchronisiere Zeitachsen...")
    df_matlab_sync, df_python_sync = resample_to_common_time(df_matlab, df_python)
    print(f"    Gemeinsamer Zeitbereich: {df_matlab_sync['time'].min():.4f}s - "
          f"{df_matlab_sync['time'].max():.4f}s ({len(df_matlab_sync)} Punkte)")
    
    # Metriken berechnen
    print("\n[3] Berechne Vergleichsmetriken...")
    print(f"    Toleranzen: absolut={TOLERANCE_ABSOLUTE}, relativ={TOLERANCE_RELATIVE:.0%}")
    print()
    
    all_passed = True
    results = {}
    
    for signal in SIGNALS_TO_COMPARE:
        if signal in df_matlab_sync.columns and signal in df_python_sync.columns:
            metrics = compute_metrics(
                df_matlab_sync[signal].values,
                df_python_sync[signal].values
            )
            results[signal] = metrics
            passed = check_match(metrics, signal)
            all_passed = all_passed and passed
        else:
            print(f"  {signal:8s}: [SKIP] Signal nicht in beiden Datensätzen")
    
    # Gesamtergebnis
    print()
    print("-" * 70)
    if all_passed:
        print("ERGEBNIS: [OK] Alle Signale innerhalb der Toleranzen!")
        print("          Die Simulationen stimmen ueberein.")
    else:
        print("ERGEBNIS: [DIFF] Einige Signale ausserhalb der Toleranzen!")
        print("          Die Simulationen unterscheiden sich signifikant.")
    print("-" * 70)
    
    # Plots erstellen
    print("\n[4] Erstelle Vergleichsplots...")
    plot_comparison(
        df_matlab_sync, df_python_sync, 
        SIGNALS_TO_COMPARE,
        output_path=output_dir / "comparison_plot.png"
    )
    
    if 'i_d' in df_matlab_sync.columns and 'i_q' in df_matlab_sync.columns:
        plot_phase_portrait(
            df_matlab_sync, df_python_sync,
            output_path=output_dir / "phase_portrait.png"
        )
    
    # Ergebnisse als CSV speichern
    results_df = pd.DataFrame(results).T
    results_csv = output_dir / "comparison_metrics.csv"
    results_df.to_csv(results_csv)
    print(f"\n[OK] Metriken gespeichert: {results_csv}")
    
    print("\n[5] Fertig!")
    print(f"    Alle Ergebnisse unter: {output_dir}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

