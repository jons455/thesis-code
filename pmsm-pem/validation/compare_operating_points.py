"""
PMSM Arbeitspunkt-Vergleich
============================
Vergleicht die Simulationsergebnisse verschiedener Arbeitspunkte (id/iq)
zwischen MATLAB, GEM Standard und eigenem Controller.

Fokus auf 1000 rpm mit variabler Last und Feldschwächung.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

# ============================================================================
# KONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent      # validation/
PMSM_PEM_DIR = SCRIPT_DIR.parent        # pmsm-pem/
PROJECT_ROOT = PMSM_PEM_DIR.parent      # thesis-code/

N_RPM = 1500  # Run 003: Andere Drehzahl (vorher 1000)

# Arbeitspunkte (müssen mit run_operating_point_tests.py übereinstimmen)
# Run 003: Andere Kombinationen
OPERATING_POINTS = [
    (0.0, 1.0, "very_low_load"),
    (0.0, 3.5, "mid_low_load"),
    (0.0, 6.0, "mid_high_load"),
    (-2.0, 3.0, "fw_light"),
    (-4.0, 4.0, "fw_balanced"),
    (-6.0, 3.0, "fw_strong_low_torque"),
]

COLORS = {
    'MATLAB': '#1f77b4',            # Blau
    'GEM Standard': '#2ca02c',      # Grün (Run 003: Hauptvergleich)
    # 'GEM Eigener Ctrl': '#ff7f0e',  # Deaktiviert
}

SIGNALS = ['i_d', 'i_q', 'u_d', 'u_q']


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def get_op_filename(id_val: float, iq_val: float) -> str:
    """Generiert Dateinamen für einen Arbeitspunkt."""
    return f"op_n{N_RPM:04d}_id{int(id_val):+03d}_iq{int(iq_val):+03d}.csv"


def get_sources_for_op(id_val: float, iq_val: float) -> dict:
    """Pfade zu den CSV-Dateien für einen Arbeitspunkt."""
    filename = get_op_filename(id_val, iq_val)
    return {
        'MATLAB': PROJECT_ROOT / "pmsm-matlab" / "export" / "validation" / 
                  f"validation_op_n{N_RPM:04d}_id{int(id_val):+03d}_iq{int(iq_val):+03d}.csv",
        'GEM Standard': PMSM_PEM_DIR / "export" / "gem_standard" / filename,
        # Eigener Controller deaktiviert
        # 'GEM Eigener Ctrl': PMSM_PEM_DIR / "export" / "matlab_match" / filename,
    }


def load_csv(filepath: Path, name: str) -> pd.DataFrame | None:
    """Lädt eine CSV-Datei."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"  [FEHLER] {name}: {e}")
        return None


def compute_metrics(ref_data: np.ndarray, test_data: np.ndarray) -> dict:
    """Berechnet Vergleichsmetriken."""
    # Längen anpassen falls nötig
    min_len = min(len(ref_data), len(test_data))
    ref_data = ref_data[:min_len]
    test_data = test_data[:min_len]
    
    error = ref_data - test_data
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_err = np.max(np.abs(error))
    
    ref_mean = np.mean(np.abs(ref_data))
    rel_err = mae / ref_mean if ref_mean > 1e-6 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'MaxErr': max_err, 'RelErr': rel_err}


def get_steady_state_value(df: pd.DataFrame, signal: str, 
                           t_start: float = 0.1) -> float:
    """Extrahiert den Steady-State Wert (Mittelwert ab t_start)."""
    if df is None or signal not in df.columns:
        return np.nan
    mask = df['time'] >= t_start
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, signal].mean()


# ============================================================================
# VERGLEICH ZWISCHEN CONTROLLERN (GEM Standard vs. Eigener)
# ============================================================================

def compare_controllers_for_all_ops(output_dir: Path) -> pd.DataFrame:
    """Vergleicht GEM Standard vs. Eigenen Controller für alle Arbeitspunkte."""
    print("\n[1] Vergleiche Controller für alle Arbeitspunkte...")
    
    results = []
    
    for id_val, iq_val, name in OPERATING_POINTS:
        sources = get_sources_for_op(id_val, iq_val)
        
        df_gem = load_csv(sources['GEM Standard'], 'GEM Standard')
        df_own = load_csv(sources['GEM Eigener Ctrl'], 'Eigener Ctrl')
        
        if df_gem is None:
            print(f"  [SKIP] {name}: Keine GEM Standard Daten")
            continue
            
        print(f"  Arbeitspunkt: id={id_val:+.1f} A, iq={iq_val:+.1f} A ({name})")
        
        row = {
            'id_ref': id_val,
            'iq_ref': iq_val,
            'name': name,
            'i_total': np.sqrt(id_val**2 + iq_val**2),
        }
        
        # Steady-State Werte für GEM Standard
        for signal in ['i_d', 'i_q', 'u_d', 'u_q']:
            row[f'{signal}_gem'] = get_steady_state_value(df_gem, signal)
            if df_own is not None:
                row[f'{signal}_own'] = get_steady_state_value(df_own, signal)
        
        # Fehler relativ zum Sollwert
        row['id_error_gem'] = row.get('i_d_gem', np.nan) - id_val
        row['iq_error_gem'] = row.get('i_q_gem', np.nan) - iq_val
        
        if df_own is not None:
            row['id_error_own'] = row.get('i_d_own', np.nan) - id_val
            row['iq_error_own'] = row.get('i_q_own', np.nan) - iq_val
        
        results.append(row)
    
    df_results = pd.DataFrame(results)
    return df_results


def plot_operating_point_comparison(df_results: pd.DataFrame, output_dir: Path):
    """Erstellt Visualisierung der Arbeitspunkt-Ergebnisse."""
    if df_results.empty:
        print("  [SKIP] Keine Daten für Plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Strom-Tracking Genauigkeit (id)
    ax = axes[0, 0]
    x = np.arange(len(df_results))
    width = 0.35
    
    if 'id_error_gem' in df_results.columns:
        ax.bar(x - width/2, df_results['id_error_gem'], width, 
               label='GEM Standard', color=COLORS['GEM Standard'])
    if 'id_error_own' in df_results.columns:
        ax.bar(x + width/2, df_results['id_error_own'], width, 
               label='Eigener Ctrl', color=COLORS['GEM Eigener Ctrl'])
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('id Fehler [A]')
    ax.set_title('d-Achsen Strom: Tracking-Fehler')
    ax.set_xticks(x)
    ax.set_xticklabels([f"id={r.id_ref:+.0f}\niq={r.iq_ref:.0f}" 
                        for r in df_results.itertuples()], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Strom-Tracking Genauigkeit (iq)
    ax = axes[0, 1]
    
    if 'iq_error_gem' in df_results.columns:
        ax.bar(x - width/2, df_results['iq_error_gem'], width, 
               label='GEM Standard', color=COLORS['GEM Standard'])
    if 'iq_error_own' in df_results.columns:
        ax.bar(x + width/2, df_results['iq_error_own'], width, 
               label='Eigener Ctrl', color=COLORS['GEM Eigener Ctrl'])
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('iq Fehler [A]')
    ax.set_title('q-Achsen Strom: Tracking-Fehler')
    ax.set_xticks(x)
    ax.set_xticklabels([f"id={r.id_ref:+.0f}\niq={r.iq_ref:.0f}" 
                        for r in df_results.itertuples()], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Spannungen (ud)
    ax = axes[1, 0]
    
    if 'u_d_gem' in df_results.columns:
        ax.bar(x - width/2, df_results['u_d_gem'], width, 
               label='GEM Standard', color=COLORS['GEM Standard'])
    if 'u_d_own' in df_results.columns:
        ax.bar(x + width/2, df_results['u_d_own'], width, 
               label='Eigener Ctrl', color=COLORS['GEM Eigener Ctrl'])
    
    ax.set_ylabel('ud [V]')
    ax.set_title('d-Achsen Spannung')
    ax.set_xticks(x)
    ax.set_xticklabels([f"id={r.id_ref:+.0f}\niq={r.iq_ref:.0f}" 
                        for r in df_results.itertuples()], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Spannungen (uq)
    ax = axes[1, 1]
    
    if 'u_q_gem' in df_results.columns:
        ax.bar(x - width/2, df_results['u_q_gem'], width, 
               label='GEM Standard', color=COLORS['GEM Standard'])
    if 'u_q_own' in df_results.columns:
        ax.bar(x + width/2, df_results['u_q_own'], width, 
               label='Eigener Ctrl', color=COLORS['GEM Eigener Ctrl'])
    
    ax.set_ylabel('uq [V]')
    ax.set_title('q-Achsen Spannung')
    ax.set_xticks(x)
    ax.set_xticklabels([f"id={r.id_ref:+.0f}\niq={r.iq_ref:.0f}" 
                        for r in df_results.itertuples()], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Arbeitspunkt-Vergleich @ {N_RPM} rpm\nController-Vergleich', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "operating_points_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Plot gespeichert: {output_path}")


def plot_time_series_all_ops(output_dir: Path):
    """Plottet Zeitverläufe für alle Arbeitspunkte."""
    print("\n[2] Erstelle Zeitverlauf-Plots...")
    
    n_ops = len(OPERATING_POINTS)
    fig, axes = plt.subplots(n_ops, 2, figsize=(14, 3 * n_ops), sharex=True)
    
    for idx, (id_val, iq_val, name) in enumerate(OPERATING_POINTS):
        sources = get_sources_for_op(id_val, iq_val)
        df_gem = load_csv(sources['GEM Standard'], 'GEM Standard')
        df_own = load_csv(sources['GEM Eigener Ctrl'], 'Eigener Ctrl')
        
        # i_d und i_q Plots
        ax_id = axes[idx, 0]
        ax_iq = axes[idx, 1]
        
        if df_gem is not None:
            ax_id.plot(df_gem['time'] * 1000, df_gem['i_d'], 
                      color=COLORS['GEM Standard'], label='GEM Std', linewidth=1)
            ax_iq.plot(df_gem['time'] * 1000, df_gem['i_q'], 
                      color=COLORS['GEM Standard'], label='GEM Std', linewidth=1)
        
        if df_own is not None:
            ax_id.plot(df_own['time'] * 1000, df_own['i_d'], 
                      color=COLORS['GEM Eigener Ctrl'], label='Eigener', linewidth=1, linestyle='--')
            ax_iq.plot(df_own['time'] * 1000, df_own['i_q'], 
                      color=COLORS['GEM Eigener Ctrl'], label='Eigener', linewidth=1, linestyle='--')
        
        # Sollwert-Linien
        ax_id.axhline(id_val, color='red', linestyle=':', linewidth=1, label=f'Soll: {id_val} A')
        ax_iq.axhline(iq_val, color='red', linestyle=':', linewidth=1, label=f'Soll: {iq_val} A')
        
        ax_id.set_ylabel(f'id [A]\n({name[:15]})')
        ax_iq.set_ylabel('iq [A]')
        ax_id.legend(loc='upper right', fontsize=7)
        ax_iq.legend(loc='upper right', fontsize=7)
        ax_id.grid(True, alpha=0.3)
        ax_iq.grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Zeit [ms]')
    axes[-1, 1].set_xlabel('Zeit [ms]')
    
    plt.suptitle(f'Zeitverläufe aller Arbeitspunkte @ {N_RPM} rpm', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "operating_points_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Plot gespeichert: {output_path}")


def plot_operating_point_map(df_results: pd.DataFrame, output_dir: Path):
    """Erstellt eine 2D-Karte der Arbeitspunkte mit Fehler-Heatmap."""
    if df_results.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # GEM Standard: Arbeitspunktkarte mit iq-Fehler als Größe
    ax = axes[0]
    
    if 'iq_error_gem' in df_results.columns:
        errors = np.abs(df_results['iq_error_gem'].values)
        sizes = 100 + errors * 500  # Größe proportional zum Fehler
        
        scatter = ax.scatter(df_results['id_ref'], df_results['iq_ref'],
                            c=errors, s=sizes, cmap='RdYlGn_r', 
                            edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax, label='|iq Fehler| [A]')
    
    # Annotationen
    for _, row in df_results.iterrows():
        ax.annotate(f'{row.get("iq_error_gem", 0):.3f}A', 
                   (row['id_ref'], row['iq_ref']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('id [A]')
    ax.set_ylabel('iq [A]')
    ax.set_title('GEM Standard: iq Tracking-Fehler')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    # Eigener Controller: Arbeitspunktkarte
    ax = axes[1]
    
    if 'iq_error_own' in df_results.columns:
        errors = np.abs(df_results['iq_error_own'].values)
        sizes = 100 + errors * 500
        
        scatter = ax.scatter(df_results['id_ref'], df_results['iq_ref'],
                            c=errors, s=sizes, cmap='RdYlGn_r',
                            edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax, label='|iq Fehler| [A]')
    
    # Annotationen
    for _, row in df_results.iterrows():
        if 'iq_error_own' in row:
            ax.annotate(f'{row.get("iq_error_own", 0):.3f}A', 
                       (row['id_ref'], row['iq_ref']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('id [A]')
    ax.set_ylabel('iq [A]')
    ax.set_title('Eigener Controller: iq Tracking-Fehler')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    plt.suptitle(f'Arbeitspunkt-Karte @ {N_RPM} rpm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "operating_points_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Plot gespeichert: {output_path}")


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    print("=" * 70)
    print("PMSM Arbeitspunkt-Vergleich")
    print("=" * 70)
    print(f"Drehzahl: {N_RPM} rpm")
    print(f"Anzahl Arbeitspunkte: {len(OPERATING_POINTS)}")
    
    # Output-Verzeichnis (im pmsm-pem/export/)
    output_dir = PMSM_PEM_DIR / "export" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Controller-Vergleich
    df_results = compare_controllers_for_all_ops(output_dir)
    
    if not df_results.empty:
        # CSV Export
        csv_path = output_dir / "operating_points_metrics.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n  [OK] Metriken gespeichert: {csv_path}")
        
        # Plots erstellen
        plot_operating_point_comparison(df_results, output_dir)
        plot_operating_point_map(df_results, output_dir)
    
    # Zeitverläufe
    plot_time_series_all_ops(output_dir)
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    
    if not df_results.empty:
        print("\nSteady-State Tracking-Fehler (GEM Standard):")
        print("-" * 50)
        for _, row in df_results.iterrows():
            id_err = row.get('id_error_gem', np.nan)
            iq_err = row.get('iq_error_gem', np.nan)
            print(f"  id={row['id_ref']:+.0f} A, iq={row['iq_ref']:.0f} A: "
                  f"d_id={id_err:+.4f} A, d_iq={iq_err:+.4f} A")
        
        if 'iq_error_own' in df_results.columns:
            print("\nSteady-State Tracking-Fehler (Eigener Controller):")
            print("-" * 50)
            for _, row in df_results.iterrows():
                id_err = row.get('id_error_own', np.nan)
                iq_err = row.get('iq_error_own', np.nan)
                print(f"  id={row['id_ref']:+.0f} A, iq={row['iq_ref']:.0f} A: "
                      f"d_id={id_err:+.4f} A, d_iq={iq_err:+.4f} A")
    
    print("\n" + "=" * 70)
    print("FERTIG!")
    print(f"Ergebnisse unter: {output_dir}")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

