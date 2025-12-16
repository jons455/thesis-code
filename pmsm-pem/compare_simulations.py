"""
Dreier-Vergleich: MATLAB vs. GEM Standard vs. Eigener Controller
=================================================================
Vergleicht die Simulationsergebnisse aus:
1. MATLAB/Simulink (Referenz)
2. Python + GEM mit Standard-Controller (gem_controllers)
3. Python + GEM mit eigenem MATLAB-kompatiblen Controller
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# ============================================================================
# KONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Pfade zu den CSV-Dateien
SOURCES = {
    'MATLAB': PROJECT_ROOT / "pmsm-matlab" / "export" / "validation" / "validation_sim.csv",
    'GEM Standard': SCRIPT_DIR / "export" / "gem_standard" / "sim_0001.csv",
    'GEM Eigener Ctrl': SCRIPT_DIR / "export" / "matlab_match" / "sim_0001.csv",
}

SIGNALS = ['i_d', 'i_q', 'n', 'u_d', 'u_q']
COLORS = {
    'MATLAB': '#1f77b4',           # Blau
    'GEM Standard': '#ff7f0e',      # Orange
    'GEM Eigener Ctrl': '#2ca02c',  # Grün
}


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def load_csv(filepath: Path, name: str) -> pd.DataFrame | None:
    if not filepath.exists():
        print(f"[SKIP] {name}: Datei nicht gefunden ({filepath})")
        return None
    df = pd.read_csv(filepath)
    print(f"[OK] {name}: {len(df)} Zeilen geladen")
    return df


def resample_to_common_time(dataframes: dict, time_col='time'):
    """Bringt alle DataFrames auf gemeinsame Zeitbasis."""
    valid_dfs = {k: v for k, v in dataframes.items() if v is not None}
    if len(valid_dfs) < 2:
        return valid_dfs
    
    # Gemeinsamer Zeitbereich
    t_start = max(df[time_col].min() for df in valid_dfs.values())
    t_end = min(df[time_col].max() for df in valid_dfs.values())
    dt = min(np.median(np.diff(df[time_col])) for df in valid_dfs.values())
    
    t_common = np.arange(t_start, t_end, dt)
    
    resampled = {}
    for name, df in valid_dfs.items():
        new_df = pd.DataFrame({'time': t_common})
        for col in df.columns:
            if col != time_col:
                new_df[col] = np.interp(t_common, df[time_col], df[col])
        resampled[name] = new_df
    
    return resampled


def compute_metrics(ref_data: np.ndarray, test_data: np.ndarray) -> dict:
    error = ref_data - test_data
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_err = np.max(np.abs(error))
    
    ref_mean = np.mean(np.abs(ref_data))
    rel_err = mae / ref_mean if ref_mean > 1e-6 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'MaxErr': max_err, 'RelErr': rel_err}


# ============================================================================
# VISUALISIERUNG
# ============================================================================

def plot_all_signals(dataframes: dict, output_path: Path):
    """Plottet alle Signale im Vergleich."""
    n_signals = len(SIGNALS)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals), sharex=True)
    
    for idx, signal in enumerate(SIGNALS):
        ax = axes[idx]
        for name, df in dataframes.items():
            if df is not None and signal in df.columns:
                ax.plot(df['time']*1000, df[signal], 
                       color=COLORS.get(name, 'gray'), 
                       label=name, linewidth=1.5,
                       linestyle='-' if name == 'MATLAB' else '--')
        ax.set_ylabel(signal)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Zeit [ms]')
    fig.suptitle('PMSM Simulation - Dreier-Vergleich', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Plot gespeichert: {output_path}")


def plot_errors_vs_matlab(dataframes: dict, output_path: Path):
    """Plottet Fehler relativ zu MATLAB."""
    if 'MATLAB' not in dataframes or dataframes['MATLAB'] is None:
        print("[SKIP] Fehlerplot: MATLAB-Daten fehlen")
        return
    
    df_matlab = dataframes['MATLAB']
    others = {k: v for k, v in dataframes.items() if k != 'MATLAB' and v is not None}
    
    if not others:
        print("[SKIP] Fehlerplot: Keine Vergleichsdaten")
        return
    
    n_signals = len(SIGNALS)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 2.5*n_signals), sharex=True)
    
    for idx, signal in enumerate(SIGNALS):
        ax = axes[idx]
        if signal not in df_matlab.columns:
            continue
        for name, df in others.items():
            if signal in df.columns:
                error = df_matlab[signal].values - df[signal].values
                ax.plot(df['time']*1000, error, 
                       color=COLORS.get(name, 'gray'), 
                       label=f'MATLAB - {name}', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel(f'Δ{signal}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Zeit [ms]')
    fig.suptitle('Fehler relativ zu MATLAB', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Fehlerplot gespeichert: {output_path}")


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    print("=" * 70)
    print("PMSM Simulation - Dreier-Vergleich")
    print("=" * 70)
    print()
    
    # Output-Verzeichnis
    output_dir = SCRIPT_DIR / "export" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Daten laden
    print("[1] Lade Simulationsdaten...\n")
    dataframes = {}
    for name, path in SOURCES.items():
        dataframes[name] = load_csv(path, name)
    
    loaded_count = sum(1 for df in dataframes.values() if df is not None)
    print(f"\n    {loaded_count}/{len(SOURCES)} Quellen geladen")
    
    if loaded_count < 2:
        print("\n[FEHLER] Mindestens 2 Datenquellen nötig für Vergleich!")
        print("\nFühre zuerst aus:")
        print("  1. MATLAB: pmsm_validation_compare.m")
        print("  2. Python: python simulate_pmsm.py")
        print("  3. Python: python simulate_pmsm_matlab_match.py")
        sys.exit(1)
    
    # Synchronisieren
    print("\n[2] Synchronisiere Zeitachsen...")
    dataframes = resample_to_common_time(dataframes)
    
    sample_df = next(df for df in dataframes.values() if df is not None)
    print(f"    Zeitbereich: {sample_df['time'].min():.4f}s - {sample_df['time'].max():.4f}s")
    print(f"    Datenpunkte: {len(sample_df)}")
    
    # Metriken berechnen (relativ zu MATLAB wenn vorhanden)
    print("\n[3] Berechne Metriken...")
    
    if 'MATLAB' in dataframes and dataframes['MATLAB'] is not None:
        ref_name = 'MATLAB'
    else:
        ref_name = next(k for k, v in dataframes.items() if v is not None)
    
    print(f"    Referenz: {ref_name}\n")
    
    results = {}
    for name, df in dataframes.items():
        if name == ref_name or df is None:
            continue
        results[name] = {}
        print(f"  {name} vs. {ref_name}:")
        for signal in SIGNALS:
            if signal in df.columns and signal in dataframes[ref_name].columns:
                m = compute_metrics(dataframes[ref_name][signal].values, df[signal].values)
                results[name][signal] = m
                status = "✓" if m['MAE'] < 0.5 else "✗"
                print(f"    {signal:6s}: MAE={m['MAE']:7.4f}, RMSE={m['RMSE']:7.4f}, "
                      f"MaxErr={m['MaxErr']:7.4f} {status}")
        print()
    
    # Plots erstellen
    print("[4] Erstelle Plots...")
    plot_all_signals(dataframes, output_dir / "comparison_all.png")
    plot_errors_vs_matlab(dataframes, output_dir / "errors_vs_matlab.png")
    
    # Metriken speichern
    if results:
        rows = []
        for source, signals in results.items():
            for signal, metrics in signals.items():
                rows.append({'Source': source, 'Signal': signal, **metrics})
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        print(f"[OK] Metriken gespeichert: {output_dir / 'metrics.csv'}")
    
    print("\n[5] Fertig!")
    print(f"    Ergebnisse unter: {output_dir}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
