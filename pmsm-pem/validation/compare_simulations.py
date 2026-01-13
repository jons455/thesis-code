"""
Dreier-Vergleich: MATLAB vs. GEM Standard vs. Eigener Controller
=================================================================
Vergleicht die Simulationsergebnisse aus:
1. MATLAB/Simulink (Referenz)
2. Python + GEM mit Standard-Controller (gem_controllers)
3. Python + GEM mit eigenem MATLAB-kompatiblen Controller

Jetzt mit Unterstützung für mehrere Drehzahlen (500, 1500, 2500 rpm).
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

SCRIPT_DIR = Path(__file__).parent      # validation/
PMSM_PEM_DIR = SCRIPT_DIR.parent        # pmsm-pem/
PROJECT_ROOT = PMSM_PEM_DIR.parent      # thesis-code/

# Drehzahlen für den Vergleich
# Run 003: Andere Drehzahlen
SPEEDS_RPM = [750, 1250, 2000]

# Pfade zu den CSV-Dateien (pro Drehzahl)
# Run 003: Nur GEM Standard (verifiziert als MATLAB-äquivalent)
def get_sources(n_rpm: int) -> dict:
    return {
        'MATLAB': PROJECT_ROOT / "pmsm-matlab" / "export" / "validation" / f"validation_sim_n{n_rpm:04d}.csv",
        'GEM Standard': PMSM_PEM_DIR / "export" / "gem_standard" / f"sim_n{n_rpm:04d}.csv",
        # Eigener Controller deaktiviert - hat nicht funktioniert
        # 'GEM Eigener Ctrl': PMSM_PEM_DIR / "export" / "matlab_match" / f"sim_n{n_rpm:04d}.csv",
    }

SIGNALS = ['i_d', 'i_q', 'n', 'u_d', 'u_q']
COLORS = {
    'MATLAB': '#1f77b4',           # Blau
    'GEM Standard': '#2ca02c',      # Grün (Run 003: als Hauptvergleich)
    # 'GEM Eigener Ctrl': '#ff7f0e',  # Deaktiviert
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


def slice_window(df: pd.DataFrame, t0: float | None, t1: float | None) -> pd.DataFrame:
    if df is None:
        return df
    m = np.ones(len(df), dtype=bool)
    if t0 is not None:
        m &= df["time"].values >= t0
    if t1 is not None:
        m &= df["time"].values < t1
    return df.loc[m].reset_index(drop=True)


# ============================================================================
# VISUALISIERUNG
# ============================================================================

def plot_all_signals(dataframes: dict, output_path: Path, n_rpm: int):
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
    fig.suptitle(f'PMSM Simulation - Dreier-Vergleich @ {n_rpm} rpm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Plot gespeichert: {output_path}")


def plot_errors_vs_matlab(dataframes: dict, output_path: Path, n_rpm: int):
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
    fig.suptitle(f'Fehler relativ zu MATLAB @ {n_rpm} rpm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Fehlerplot gespeichert: {output_path}")


def plot_summary(all_results: list, output_path: Path):
    """Plottet Zusammenfassung über alle Drehzahlen."""
    # Sammle Daten für Bar-Plot
    sources = ['GEM Standard', 'GEM Eigener Ctrl']
    signals_to_plot = ['i_d', 'i_q', 'u_d', 'u_q']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, signal in enumerate(signals_to_plot):
        ax = axes[ax_idx]
        x = np.arange(len(SPEEDS_RPM))
        width = 0.35
        
        for s_idx, source in enumerate(sources):
            maes = []
            for r in all_results:
                if r['source'] == source and r['signal'] == signal and r['window'] == 'steady_state':
                    maes.append(r['MAE'])
            if len(maes) == len(SPEEDS_RPM):
                offset = (s_idx - 0.5) * width
                bars = ax.bar(x + offset, maes, width, label=source, color=COLORS.get(source, 'gray'))
        
        ax.set_xlabel('Drehzahl [rpm]')
        ax.set_ylabel(f'MAE {signal}')
        ax.set_title(f'{signal} - MAE vs MATLAB')
        ax.set_xticks(x)
        ax.set_xticklabels(SPEEDS_RPM)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vergleich über alle Drehzahlen (Steady-State)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Zusammenfassungs-Plot gespeichert: {output_path}")


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def compare_single_speed(n_rpm: int, output_dir: Path) -> list:
    """Vergleicht eine einzelne Drehzahl und gibt Ergebnisse zurück."""
    print(f"\n{'='*70}")
    print(f"Vergleich @ {n_rpm} rpm")
    print('='*70)
    
    sources = get_sources(n_rpm)
    
    # Daten laden
    print("\n[1] Lade Simulationsdaten...")
    dataframes = {}
    for name, path in sources.items():
        dataframes[name] = load_csv(path, name)
    
    loaded_count = sum(1 for df in dataframes.values() if df is not None)
    print(f"    {loaded_count}/{len(sources)} Quellen geladen")
    
    if loaded_count < 2:
        print(f"[WARNUNG] Nicht genug Daten für {n_rpm} rpm, überspringe...")
        return []
    
    # Synchronisieren
    print("\n[2] Synchronisiere Zeitachsen...")
    dataframes = resample_to_common_time(dataframes)
    
    sample_df = next(df for df in dataframes.values() if df is not None)
    t_max = sample_df['time'].max()
    print(f"    Zeitbereich: 0s - {t_max:.4f}s ({len(sample_df)} Punkte)")
    
    # Metriken berechnen
    print("\n[3] Berechne Metriken...")
    ref_name = 'MATLAB'
    
    # Fenster: transient (erste 20ms) vs steady_state (ab 50ms)
    windows = [
        ("all", None, None),
        ("transient", 0.0, 0.02),
        ("steady_state", 0.05, None),
    ]
    
    results = []
    for name, df in dataframes.items():
        if name == ref_name or df is None:
            continue
        print(f"\n  {name} vs. {ref_name}:")
        for window_name, t0, t1 in windows:
            df_ref_w = slice_window(dataframes[ref_name], t0, t1)
            df_test_w = slice_window(df, t0, t1)
            if df_ref_w is None or df_test_w is None or len(df_ref_w) == 0 or len(df_test_w) == 0:
                continue
            for signal in SIGNALS:
                if signal in df_test_w.columns and signal in df_ref_w.columns:
                    m = compute_metrics(df_ref_w[signal].values, df_test_w[signal].values)
                    results.append({
                        'n_rpm': n_rpm,
                        'source': name,
                        'window': window_name,
                        'signal': signal,
                        **m
                    })
            # Kurzprint steady_state i_q
            if window_name == "steady_state":
                for r in results:
                    if r['source'] == name and r['window'] == 'steady_state' and r['signal'] == 'i_q' and r['n_rpm'] == n_rpm:
                        print(f"    {window_name} i_q: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}")
                        break
    
    # Plots erstellen
    print("\n[4] Erstelle Plots...")
    plot_all_signals(dataframes, output_dir / f"comparison_n{n_rpm:04d}.png", n_rpm)
    plot_errors_vs_matlab(dataframes, output_dir / f"errors_n{n_rpm:04d}.png", n_rpm)
    
    return results


def main():
    print("=" * 70)
    print("PMSM Simulation - Multi-Speed Dreier-Vergleich")
    print("=" * 70)
    print(f"Drehzahlen: {SPEEDS_RPM} rpm")
    
    # Output-Verzeichnis (im pmsm-pem/export/)
    output_dir = PMSM_PEM_DIR / "export" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alle Drehzahlen vergleichen
    all_results = []
    for n_rpm in SPEEDS_RPM:
        results = compare_single_speed(n_rpm, output_dir)
        all_results.extend(results)
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_dir / "metrics_all_speeds.csv", index=False)
        print(f"\n[OK] Alle Metriken gespeichert: {output_dir / 'metrics_all_speeds.csv'}")
        
        # Zusammenfassungs-Tabelle für Steady-State
        print("\n" + "-" * 70)
        print("Steady-State Metriken (t >= 50ms)")
        print("-" * 70)
        
        steady = df_results[df_results['window'] == 'steady_state']
        for signal in ['i_q', 'u_q']:
            print(f"\n{signal}:")
            for source in ['GEM Standard', 'GEM Eigener Ctrl']:
                print(f"  {source}:")
                for n_rpm in SPEEDS_RPM:
                    row = steady[(steady['source'] == source) & 
                                (steady['signal'] == signal) & 
                                (steady['n_rpm'] == n_rpm)]
                    if not row.empty:
                        mae = row['MAE'].values[0]
                        rmse = row['RMSE'].values[0]
                        print(f"    {n_rpm:4d} rpm: MAE={mae:7.4f}, RMSE={rmse:7.4f}")
        
        # Zusammenfassungs-Plot
        plot_summary(all_results, output_dir / "summary_all_speeds.png")
    
    print("\n" + "=" * 70)
    print("FERTIG!")
    print(f"Ergebnisse unter: {output_dir}")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
