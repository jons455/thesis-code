"""
Edge Impulse Data Preparation Script
====================================

This script prepares the PMSM FOC controller data for Edge Impulse training.

Usage:
    python prepare_edge_impulse.py

Output:
    - edge_impulse_train.csv (70% of data)
    - edge_impulse_validation.csv (15% of data) 
    - edge_impulse_test.csv (15% of data)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_data(use_parquet=True):
    """Load the merged panel data."""
    data_dir = Path(__file__).parent / 'data' / 'merged'
    
    if use_parquet:
        file_path = data_dir / 'merged_panel.parquet'
        print(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)
    else:
        file_path = data_dir / 'merged_panel.csv'
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df):,} samples")
    return df


def prepare_basic_data(df):
    """
    Prepare basic instantaneous mapping data.
    
    Features: i_d, i_q, n
    Targets: u_d, u_q
    """
    print("\n=== Preparing Basic (Instantaneous) Mapping ===")
    
    # Select only needed columns
    df_model = df[['i_d', 'i_q', 'n', 'u_d', 'u_q']].copy()
    
    # Check for missing values
    missing = df_model.isnull().sum()
    if missing.any():
        print(f"Warning: Found missing values:\n{missing}")
        df_model = df_model.dropna()
        print(f"Dropped to {len(df_model):,} samples")
    
    # Data statistics
    print("\n--- Feature Statistics ---")
    print(df_model.describe())
    
    return df_model


def prepare_windowed_data(df, window_size=5):
    """
    Prepare time-windowed data for temporal pattern learning.
    
    Creates sliding windows of [i_d, i_q, n] over time.
    """
    print(f"\n=== Preparing Windowed Data (window={window_size}) ===")
    
    features = ['i_d', 'i_q', 'n']
    targets = ['u_d', 'u_q']
    
    windows = []
    runs_processed = 0
    
    for run_id in df['run_id'].unique():
        run_data = df[df['run_id'] == run_id].reset_index(drop=True)
        
        # Create windows for this run
        for i in range(window_size, len(run_data)):
            # Get window of features (flattened)
            window_features = []
            for t in range(window_size):
                for feat in features:
                    window_features.append(run_data.loc[i - window_size + t, feat])
            
            # Get current target
            current_targets = run_data.loc[i, targets].values
            
            # Combine
            row = window_features + list(current_targets)
            windows.append(row)
        
        runs_processed += 1
        if runs_processed % 100 == 0:
            print(f"  Processed {runs_processed}/{len(df['run_id'].unique())} runs...")
    
    # Create column names
    cols = []
    for t in range(window_size):
        for feat in features:
            cols.append(f'{feat}_t{t}')
    cols.extend(targets)
    
    df_windowed = pd.DataFrame(windows, columns=cols)
    print(f"Created {len(df_windowed):,} windowed samples")
    
    return df_windowed


def split_by_runs(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data by run_id to prevent data leakage.
    
    Args:
        df: DataFrame with 'run_id' column
        train_ratio: Fraction of runs for training
        val_ratio: Fraction of runs for validation
        test_ratio: Fraction of runs for testing
    """
    print("\n=== Splitting Data by Runs ===")
    
    # Get unique run IDs
    unique_runs = sorted(df['run_id'].unique())
    n_runs = len(unique_runs)
    
    # Calculate split points
    n_train = int(n_runs * train_ratio)
    n_val = int(n_runs * val_ratio)
    # n_test = n_runs - n_train - n_val  # Remaining
    
    # Split run IDs
    train_runs = unique_runs[:n_train]
    val_runs = unique_runs[n_train:n_train + n_val]
    test_runs = unique_runs[n_train + n_val:]
    
    print(f"Total runs: {n_runs}")
    print(f"  Train runs: {len(train_runs)} (runs {min(train_runs)}-{max(train_runs)})")
    print(f"  Val runs:   {len(val_runs)} (runs {min(val_runs)}-{max(val_runs)})")
    print(f"  Test runs:  {len(test_runs)} (runs {min(test_runs)}-{max(test_runs)})")
    
    # Split data
    df_train = df[df['run_id'].isin(train_runs)].copy()
    df_val = df[df['run_id'].isin(val_runs)].copy()
    df_test = df[df['run_id'].isin(test_runs)].copy()
    
    # Drop run_id column (not needed for training)
    if 'run_id' in df_train.columns:
        df_train = df_train.drop('run_id', axis=1)
        df_val = df_val.drop('run_id', axis=1)
        df_test = df_test.drop('run_id', axis=1)
    
    print(f"\n--- Sample Counts ---")
    print(f"  Train: {len(df_train):,} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(df_val):,} samples ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(df_test):,} samples ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_val, df_test


def save_datasets(df_train, df_val, df_test, output_dir='edge_impulse_data', prefix=''):
    """Save train/val/test datasets as CSV files."""
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    train_file = output_path / f'{prefix}edge_impulse_train.csv'
    val_file = output_path / f'{prefix}edge_impulse_validation.csv'
    test_file = output_path / f'{prefix}edge_impulse_test.csv'
    
    print(f"\n=== Saving Datasets to {output_path} ===")
    
    df_train.to_csv(train_file, index=False)
    print(f"  ✓ Train: {train_file.name} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    df_val.to_csv(val_file, index=False)
    print(f"  ✓ Val:   {val_file.name} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    df_test.to_csv(test_file, index=False)
    print(f"  ✓ Test:  {test_file.name} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    return train_file, val_file, test_file


def print_data_info(df_train, df_val, df_test):
    """Print summary statistics of the datasets."""
    print("\n" + "="*60)
    print("=== Dataset Summary ===")
    print("="*60)
    
    print("\n--- Training Data Statistics ---")
    print(df_train.describe())
    
    print("\n--- Validation Data Statistics ---")
    print(df_val.describe())
    
    print("\n--- Test Data Statistics ---")
    print(df_test.describe())
    
    # Check value ranges
    print("\n--- Value Ranges ---")
    for col in df_train.columns:
        train_range = (df_train[col].min(), df_train[col].max())
        val_range = (df_val[col].min(), df_val[col].max())
        test_range = (df_test[col].min(), df_test[col].max())
        print(f"{col:8s}: Train [{train_range[0]:7.3f}, {train_range[1]:7.3f}] | "
              f"Val [{val_range[0]:7.3f}, {val_range[1]:7.3f}] | "
              f"Test [{test_range[0]:7.3f}, {test_range[1]:7.3f}]")


def main():
    """Main execution function."""
    print("="*60)
    print("Edge Impulse Data Preparation for PMSM FOC Controller")
    print("="*60)
    
    # Load data
    df = load_data(use_parquet=True)
    
    # Option 1: Basic instantaneous mapping (RECOMMENDED)
    print("\n" + "="*60)
    print("OPTION 1: Basic Instantaneous Mapping (RECOMMENDED)")
    print("="*60)
    
    df_basic = prepare_basic_data(df)
    df_train, df_val, df_test = split_by_runs(df_basic)
    save_datasets(df_train, df_val, df_test, prefix='basic_')
    print_data_info(df_train, df_val, df_test)
    
    # Option 2: Windowed data (ADVANCED)
    print("\n\n" + "="*60)
    print("OPTION 2: Windowed Data (ADVANCED - Optional)")
    print("="*60)
    
    user_input = input("\nPrepare windowed data? (y/n, default=n): ").strip().lower()
    
    if user_input == 'y':
        window_size = input("Window size (default=5): ").strip()
        window_size = int(window_size) if window_size else 5
        
        df_windowed = prepare_windowed_data(df, window_size=window_size)
        df_train_w, df_val_w, df_test_w = split_by_runs(df_windowed)
        save_datasets(df_train_w, df_val_w, df_test_w, prefix=f'windowed{window_size}_')
        print_data_info(df_train_w, df_val_w, df_test_w)
    else:
        print("Skipping windowed data preparation.")
    
    # Final summary
    print("\n" + "="*60)
    print("✓ Data Preparation Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Go to Edge Impulse Studio: https://studio.edgeimpulse.com/")
    print("2. Create a new 'Regression' project")
    print("3. Upload 'basic_edge_impulse_train.csv' as Training data")
    print("4. Upload 'basic_edge_impulse_validation.csv' as Testing data")
    print("5. Follow the EDGE_IMPULSE_TRAINING_GUIDE.md for detailed instructions")
    print("\nRecommended model architecture:")
    print("  Input: [i_d, i_q, n] (3 features)")
    print("  Hidden: 64 → 32 → 16 neurons (ReLU)")
    print("  Output: [u_d, u_q] (2 outputs, Linear)")
    print("\nHappy training! 🚀")


if __name__ == "__main__":
    main()

