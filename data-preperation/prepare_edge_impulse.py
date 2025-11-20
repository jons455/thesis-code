"""
Edge Impulse Data Preparation Script
====================================

This script prepares the PMSM FOC controller data for Edge Impulse training.

Original data: 10 kHz sampling rate (100 μs between samples)
This script supports downsampling for SNN training (e.g., 10 kHz → 1 kHz).

Usage:
    python prepare_edge_impulse.py

Output:
    - edge_impulse_train.csv (70% of data)
    - edge_impulse_validation.csv (15% of data) 
    - edge_impulse_test.csv (15% of data)
    
Downsampling options:
    - 10 kHz (original): factor=1
    - 1 kHz: factor=10 (recommended for SNN)
    - 100 Hz: factor=100
    - Custom: any factor
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


def downsample_data(df, factor=10, preserve_time=True):
    """
    Downsample data by taking every Nth sample.
    
    Original data: 10 kHz (100 μs between samples)
    After downsampling: 10 kHz / factor
    
    Args:
        df: DataFrame with 'run_id' and 'time' columns
        factor: Downsampling factor (1=no downsampling, 10=10kHz→1kHz)
        preserve_time: If True, keep original time column; if False, recalculate
    
    Returns:
        Downsampled DataFrame
    """
    if factor == 1:
        print("\n=== No Downsampling (using original 10 kHz data) ===")
        return df.copy()
    
    print(f"\n=== Downsampling Data (factor={factor}) ===")
    original_rate_khz = 10.0
    new_rate_khz = original_rate_khz / factor
    original_samples = len(df)
    
    print(f"Original sampling rate: {original_rate_khz:.1f} kHz")
    print(f"New sampling rate: {new_rate_khz:.1f} kHz")
    print(f"Original samples: {original_samples:,}")
    
    downsampled_parts = []
    runs_processed = 0
    
    for run_id in sorted(df['run_id'].unique()):
        run_data = df[df['run_id'] == run_id].reset_index(drop=True)
        
        # Take every Nth sample
        run_downsampled = run_data.iloc[::factor, :].copy()
        
        # Recalculate time if requested
        if not preserve_time:
            # New time starts at 0 and increments by factor * original_dt
            original_dt = run_data['time'].iloc[1] - run_data['time'].iloc[0] if len(run_data) > 1 else 0.0001
            new_dt = original_dt * factor
            run_downsampled['time'] = np.arange(len(run_downsampled)) * new_dt
        
        downsampled_parts.append(run_downsampled)
        runs_processed += 1
        
        if runs_processed % 100 == 0:
            print(f"  Processed {runs_processed}/{len(df['run_id'].unique())} runs...")
    
    df_downsampled = pd.concat(downsampled_parts, ignore_index=True)
    new_samples = len(df_downsampled)
    
    print(f"New samples: {new_samples:,}")
    print(f"Reduction: {original_samples/new_samples:.1f}× ({original_samples-new_samples:,} samples removed)")
    print(f"Sampling period: {1000/new_rate_khz:.1f} μs (was 100 μs)")
    
    return df_downsampled


def prepare_basic_data(df, keep_run_id=False):
    """
    Prepare basic instantaneous mapping data.
    
    Features: i_d, i_q, n
    Targets: u_d, u_q
    """
    print("\n=== Preparing Basic (Instantaneous) Mapping ===")
    
    # Select columns (keep run_id if needed for splitting)
    if keep_run_id:
        df_model = df[['run_id', 'i_d', 'i_q', 'n', 'u_d', 'u_q']].copy()
    else:
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


def save_datasets(df_train, df_val, df_test, output_dir='edge_impulse_data', prefix='', add_timestamp=True):
    """
    Save train/val/test datasets as CSV files.
    
    Edge Impulse requires timestamp column for CSV uploads.
    """
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    train_file = output_path / f'{prefix}edge_impulse_train.csv'
    val_file = output_path / f'{prefix}edge_impulse_validation.csv'
    test_file = output_path / f'{prefix}edge_impulse_test.csv'
    
    print(f"\n=== Saving Datasets to {output_path} ===")
    
    # Add timestamp column for Edge Impulse compatibility
    if add_timestamp:
        # Create timestamp: milliseconds since start (0, 1, 2, ...)
        # Edge Impulse expects timestamp in milliseconds
        df_train_copy = df_train.copy()
        df_val_copy = df_val.copy()
        df_test_copy = df_test.copy()
        
        # Add timestamp column (in milliseconds, starting from 0)
        df_train_copy.insert(0, 'timestamp', np.arange(len(df_train_copy)) * 1.0)  # 1 ms steps
        df_val_copy.insert(0, 'timestamp', np.arange(len(df_val_copy)) * 1.0)
        df_test_copy.insert(0, 'timestamp', np.arange(len(df_test_copy)) * 1.0)
        
        df_train_copy.to_csv(train_file, index=False)
        df_val_copy.to_csv(val_file, index=False)
        df_test_copy.to_csv(test_file, index=False)
    else:
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)
    
    print(f"  ✓ Train: {train_file.name} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  ✓ Val:   {val_file.name} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
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
    
    # Ask about downsampling
    print("\n" + "="*60)
    print("DOWNSAMPLING OPTIONS")
    print("="*60)
    print("\nOriginal data: 10 kHz sampling rate (100 μs between samples)")
    print("\nDownsampling options:")
    print("  1 = 10 kHz (original, recommended for ANN)")
    print("  10 = 1 kHz (recommended for SNN)")
    print("  100 = 100 Hz (for very slow SNN or analysis)")
    print("  custom = specify your own factor")
    
    downsample_input = input("\nDownsampling factor (1/10/100/custom, default=1): ").strip().lower()
    
    if downsample_input == '' or downsample_input == '1':
        downsample_factor = 1
        rate_label = '10khz'
    elif downsample_input == '10':
        downsample_factor = 10
        rate_label = '1khz'
    elif downsample_input == '100':
        downsample_factor = 100
        rate_label = '100hz'
    elif downsample_input == 'custom':
        try:
            downsample_factor = int(input("Enter downsampling factor: ").strip())
            rate_label = f'{10/downsample_factor:.0f}hz' if downsample_factor > 1 else '10khz'
        except ValueError:
            print("Invalid input, using factor=1 (no downsampling)")
            downsample_factor = 1
            rate_label = '10khz'
    else:
        try:
            downsample_factor = int(downsample_input)
            rate_label = f'{10/downsample_factor:.0f}hz' if downsample_factor > 1 else '10khz'
        except ValueError:
            print("Invalid input, using factor=1 (no downsampling)")
            downsample_factor = 1
            rate_label = '10khz'
    
    # Apply downsampling
    if downsample_factor > 1:
        df = downsample_data(df, factor=downsample_factor, preserve_time=True)
    
    # Option 1: Basic instantaneous mapping (RECOMMENDED)
    print("\n" + "="*60)
    print(f"OPTION 1: Basic Instantaneous Mapping @ {rate_label.upper()}")
    print("="*60)
    
    df_basic = prepare_basic_data(df, keep_run_id=True)
    df_train, df_val, df_test = split_by_runs(df_basic)
    save_datasets(df_train, df_val, df_test, prefix=f'basic_{rate_label}_')
    print_data_info(df_train, df_val, df_test)
    
    # Option 2: Windowed data (ADVANCED)
    print("\n\n" + "="*60)
    print(f"OPTION 2: Windowed Data @ {rate_label.upper()} (ADVANCED - Optional)")
    print("="*60)
    
    user_input = input("\nPrepare windowed data? (y/n, default=n): ").strip().lower()
    
    if user_input == 'y':
        window_size = input("Window size (default=5): ").strip()
        window_size = int(window_size) if window_size else 5
        
        df_windowed = prepare_windowed_data(df, window_size=window_size)
        df_train_w, df_val_w, df_test_w = split_by_runs(df_windowed)
        save_datasets(df_train_w, df_val_w, df_test_w, prefix=f'windowed{window_size}_{rate_label}_')
        print_data_info(df_train_w, df_val_w, df_test_w)
    else:
        print("Skipping windowed data preparation.")
    
    # Final summary
    print("\n" + "="*60)
    print("✓ Data Preparation Complete!")
    print("="*60)
    
    if downsample_factor > 1:
        print(f"\n📊 Sampling Rate: {rate_label.upper()}")
        print(f"   Original: 10 kHz → Downsampled: {10/downsample_factor:.1f} kHz")
        print(f"   Factor: {downsample_factor}× (every {downsample_factor}th sample)")
        if downsample_factor == 10:
            print("\n   ⚠️  NOTE: 1 kHz is recommended for SNN training")
            print("      (SNNs typically need 1-5 ms per inference)")
        elif downsample_factor == 100:
            print("\n   ⚠️  NOTE: 100 Hz is very slow - only for analysis")
            print("      (May lose important dynamics)")
    else:
        print(f"\n📊 Sampling Rate: 10 kHz (original)")
        print("   ⚠️  NOTE: For SNN, consider downsampling to 1 kHz")
        print("      (Run script again with factor=10)")
    
    print("\nNext Steps:")
    print("1. Upload generated CSV files to your training platform")
    print(f"2. Training file: basic_{rate_label}_edge_impulse_train.csv")
    print(f"3. Validation file: basic_{rate_label}_edge_impulse_validation.csv")
    print("4. See docs/TRAINING_GUIDE.md for model architecture recommendations")
    print("\nRecommended model architecture:")
    print("  Input: [i_d, i_q, n] (3 features)")
    print("  Hidden: 64 → 32 → 16 neurons (ReLU)")
    print("  Output: [u_d, u_q] (2 outputs, Linear)")
    
    if downsample_factor == 10:
        print("\nNote: 1 kHz sampling rate recommended for SNN training")
        print("      (provides sufficient time for spiking neural network inference)")


if __name__ == "__main__":
    import sys
    
    # Allow command-line argument for downsampling factor
    if len(sys.argv) > 1:
        try:
            factor = int(sys.argv[1])
            # Override main() to use this factor
            print("="*60)
            print("Edge Impulse Data Preparation for PMSM FOC Controller")
            print("="*60)
            df = load_data(use_parquet=True)
            
            if factor == 1:
                downsample_factor = 1
                rate_label = '10khz'
            elif factor == 10:
                downsample_factor = 10
                rate_label = '1khz'
            elif factor == 100:
                downsample_factor = 100
                rate_label = '100hz'
            else:
                downsample_factor = factor
                rate_label = f'{10/factor:.0f}hz' if factor > 1 else '10khz'
            
            print(f"\nUsing downsampling factor: {downsample_factor} ({rate_label})")
            
            if downsample_factor > 1:
                df = downsample_data(df, factor=downsample_factor, preserve_time=True)
            
            print("\n" + "="*60)
            print(f"OPTION 1: Basic Instantaneous Mapping @ {rate_label.upper()}")
            print("="*60)
            
            df_basic = prepare_basic_data(df, keep_run_id=True)
            df_train, df_val, df_test = split_by_runs(df_basic)
            save_datasets(df_train, df_val, df_test, prefix=f'basic_{rate_label}_')
            print_data_info(df_train, df_val, df_test)
            
            print("\n" + "="*60)
            print("Data Preparation Complete!")
            print("="*60)
            print(f"\nSampling Rate: {rate_label.upper()}")
            print(f"Training file: basic_{rate_label}_edge_impulse_train.csv")
            print(f"Validation file: basic_{rate_label}_edge_impulse_validation.csv")
        except ValueError:
            print(f"Invalid factor: {sys.argv[1]}. Must be an integer.")
            sys.exit(1)
    else:
        main()

