import os
import re
import glob
import pandas as pd
import numpy as np


def parse_run_id(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else 0


def infer_dt(series: pd.Series) -> float:
    if series.size < 2:
        return 0.0
    diffs = np.diff(series.values)
    return float(np.median(diffs))


def load_and_sort(input_dir: str, pattern: str):
    files = glob.glob(os.path.join(input_dir, pattern))
    files.sort(key=parse_run_id)
    return files


def merge_panel(files, time_col: str = "time") -> pd.DataFrame:
    parts = []
    for f in files:
        df = pd.read_csv(f)
        rid = parse_run_id(f)
        df.insert(0, "run_id", rid)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["run_id", time_col], kind="mergesort").reset_index(drop=True)
    return panel


def merge_stack(files, time_col: str = "time", continuous: bool = True) -> pd.DataFrame:
    stack_parts = []
    t_offset = 0.0
    last_dt = None

    for idx, f in enumerate(files):
        df = pd.read_csv(f)
        rid = parse_run_id(f)
        df["run_id"] = rid

        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in {f}")

        # Normalize time to start at 0 for each run
        t0 = float(df[time_col].iloc[0])
        df[time_col] = df[time_col] - t0

        if idx == 0:
            last_dt = infer_dt(df[time_col])
        else:
            if last_dt is None or last_dt <= 0:
                last_dt = infer_dt(df[time_col])

        if continuous:
            df[time_col] = df[time_col] + t_offset

        stack_parts.append(df)

        if continuous:
            # Calculate offset for next run: last time + dt (avoid duplicate timestamps)
            t_offset = float(df[time_col].iloc[-1])
            if last_dt and last_dt > 0:
                t_offset += last_dt

    if not stack_parts:
        return pd.DataFrame()

    stacked = pd.concat(stack_parts, ignore_index=True)
    stacked = stacked.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    return stacked


def save_outputs(panel: pd.DataFrame, stacked: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    panel_path_csv = os.path.join(out_dir, "merged_panel.csv")
    stacked_path_csv = os.path.join(out_dir, "merged_stacked.csv")
    panel.to_csv(panel_path_csv, index=False)
    stacked.to_csv(stacked_path_csv, index=False)
    # Save Parquet format if available
    try:
        panel.to_parquet(os.path.join(out_dir, "merged_panel.parquet"), index=False)
        stacked.to_parquet(os.path.join(out_dir, "merged_stacked.parquet"), index=False)
    except Exception:
        pass
    return panel_path_csv, stacked_path_csv


def main(input_dir: str,
         pattern: str = "sim_*.csv",
         time_col: str = "time",
         out_dir: str = None,
         mode: str = "both"):
    """
    Merge simulation CSV files into unified formats.
    
    Modes:
      - "panel": Preserves original time + run_id per run
      - "stack": Concatenates runs into continuous time series
      - "both": Generates both output formats
    """
    if out_dir is None:
        out_dir = os.path.join(input_dir, "merged")
    files = load_and_sort(input_dir, pattern)
    if not files:
        raise SystemExit(f"No files found in {input_dir!r} matching pattern {pattern!r}")

    panel = pd.DataFrame()
    stacked = pd.DataFrame()
    if mode in ("panel", "both"):
        panel = merge_panel(files, time_col=time_col)
    if mode in ("stack", "both"):
        stacked = merge_stack(files, time_col=time_col, continuous=True)

    paths = save_outputs(panel, stacked, out_dir)
    print("Wrote files:", paths)
    print("Input files:", len(files))
    if not panel.empty:
        print("Panel shape:", panel.shape)
    if not stacked.empty:
        print("Stacked shape:", stacked.shape)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merge simulation CSVs into panel and/or single stacked time series.")
    p.add_argument("input_dir", type=str, help="Directory containing sim_*.csv")
    p.add_argument("--pattern", type=str, default="sim_*.csv", help="Glob pattern (default: sim_*.csv)")
    p.add_argument("--time-col", type=str, default="time", help="Time column name")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <input_dir>/merged)")
    p.add_argument("--mode", type=str, choices=["panel", "stack", "both"], default="both", help="Output mode")
    args = p.parse_args()
    main(args.input_dir, args.pattern, args.time_col, args.out_dir, args.mode)
