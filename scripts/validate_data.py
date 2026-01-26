"""Validate training data quality."""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.utils.paths import DATA_RAW_DIR


def validate_data(data_dir: str):
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    print("=== Training Data Validation ===")
    print(f"Directory: {data_dir}")
    print(f"Total files: {len(files)}")

    if len(files) == 0:
        print("No files found!")
        return

    # Collect statistics
    errors_iq = []
    errors_id = []
    bad_files = []

    for f in files:
        df = pd.read_csv(f)
        err_iq = abs(df.i_q.iloc[-1] - df.i_q_ref.iloc[-1])
        err_id = abs(df.i_d.iloc[-1] - df.i_d_ref.iloc[-1])
        errors_iq.append(err_iq)
        errors_id.append(err_id)

        # Check if tracking is correct (sign should match)
        final_iq = df.i_q.iloc[-1]
        ref_iq = df.i_q_ref.iloc[-1]
        if ref_iq != 0 and np.sign(final_iq) != np.sign(ref_iq):
            bad_files.append((f, final_iq, ref_iq))

    print()
    print("i_q tracking error (final step):")
    print(f"  Mean: {np.mean(errors_iq):.6f} A")
    print(f"  Max:  {np.max(errors_iq):.6f} A")
    print(f"  All < 0.1A: {sum([e < 0.1 for e in errors_iq])} / {len(errors_iq)}")

    print()
    print("i_d tracking error (final step):")
    print(f"  Mean: {np.mean(errors_id):.6f} A")
    print(f"  Max:  {np.max(errors_id):.6f} A")
    print(f"  All < 0.1A: {sum([e < 0.1 for e in errors_id])} / {len(errors_id)}")

    if bad_files:
        print()
        print(f"WARNING: {len(bad_files)} files with wrong sign!")
        for f, val, ref in bad_files[:5]:
            print(f"  {Path(f).name}: i_q={val:.3f}, ref={ref:.3f}")

    # Check sample file
    df = pd.read_csv(files[len(files) // 2])
    print()
    print(f"Sample file ({Path(files[len(files) // 2]).name}):")
    print(f"  i_q_ref range: [{df.i_q_ref.min():.3f}, {df.i_q_ref.max():.3f}]")
    print(f"  u_q range: [{df.u_q.min():.3f}, {df.u_q.max():.3f}]")
    print(f"  Steps: {len(df)}")

    print()
    if np.max(errors_iq) < 0.1 and np.max(errors_id) < 0.1 and len(bad_files) == 0:
        print("=== DATA IS CLEAN ===")
        return True
    else:
        print("=== DATA HAS ISSUES ===")
        return False


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_RAW_DIR / "train")
    validate_data(data_dir)
