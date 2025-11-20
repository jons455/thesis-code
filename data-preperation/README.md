# Data Pipeline - Simulation Data Merger

This pipeline merges multiple simulation runs into unified panel and stacked time series formats for analysis.

## Input Data

### Data Structure
The pipeline processes **1000 simulation CSV files** located in the `data/` directory:
- **Files:** `sim_0001.csv` to `sim_1000.csv`
- **Format:** CSV with comma delimiter
- **Rows per file:** 2,001 time steps
- **Time range:** 0.0 to 0.2 seconds
- **Time step:** 0.0001 seconds (100 μs)

### Input Columns
| Column | Type | Description |
|--------|------|-------------|
| `time` | float64 | Time in seconds (0.0 to 0.2) |
| `i_d` | float64 | Direct-axis current in Amperes |
| `i_q` | float64 | Quadrature-axis current in Amperes |
| `n` | int64 | Rotational speed in RPM |
| `u_d` | float64 | Direct-axis voltage in Volts |
| `u_q` | float64 | Quadrature-axis voltage in Volts |

**Total input data points:** 2,001,000 (1000 files × 2001 rows)

### Example Input Data
```csv
time,i_d,i_q,n,u_d,u_q
0,0,0,1000,0,5.30929158456675
0.0001,-0.00717090122731184,-0.366775669875778,1000,0.204136476433917,7.91085317096524
0.0002,-0.0206971206157331,-0.352669851151716,1000,0.276213823082066,7.90547964492052
```

## Output Data

The pipeline generates **two output formats** in the `data/merged/` directory:

### 1. Panel Format (`merged_panel.csv` / `.parquet`)

**Purpose:** Preserves the original structure of each simulation run with run identifiers.

**Structure:**
- **Rows:** 2,001,000
- **Columns:** 7 (`run_id` + original 6 columns)
- **Sorting:** By `run_id`, then by `time`

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `run_id` | int | Simulation run identifier (1 to 1000) |
| `time` | float64 | Original time from each simulation (0.0 to 0.2) |
| `i_d` | float64 | Direct-axis current |
| `i_q` | float64 | Quadrature-axis current |
| `n` | int64 | Rotational speed |
| `u_d` | float64 | Direct-axis voltage |
| `u_q` | float64 | Quadrature-axis voltage |

**Use case:** Compare multiple runs side-by-side, analyze differences between simulations, run-specific statistics.

**Example:**
```csv
run_id,time,i_d,i_q,n,u_d,u_q
1,0.0000,0.000000,0.000000,1000,0.000000,5.309292
1,0.0001,-0.007171,-0.366776,1000,0.204136,7.910853
...
2,0.0000,0.000000,0.000000,1000,0.000000,5.309292
2,0.0001,-0.007171,-0.366776,1000,0.204136,7.910853
```

### 2. Stacked Format (`merged_stacked.csv` / `.parquet`)

**Purpose:** Creates a single continuous time series by concatenating all simulation runs sequentially.

**Structure:**
- **Rows:** 2,001,000
- **Columns:** 7 (original 6 columns + `run_id`)
- **Sorting:** By continuous `time`

**Key Features:**
- **Continuous time:** Time values adjusted to be sequential across all runs
- **No gaps:** Each run starts exactly where the previous run ended (plus one time step)
- **Time range:** 0.0 to ~200.1 seconds (1000 runs × 0.2 seconds + spacing)
- **Run tracking:** `run_id` preserved to identify which run each data point belongs to

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `time` | float64 | Continuous time across all runs (0.0 to ~200.1) |
| `i_d` | float64 | Direct-axis current |
| `i_q` | float64 | Quadrature-axis current |
| `n` | int64 | Rotational speed |
| `u_d` | float64 | Direct-axis voltage |
| `u_q` | float64 | Quadrature-axis voltage |
| `run_id` | int | Original run identifier |

**Use case:** Time series analysis, continuous signal processing, trend analysis across multiple runs.

**Example:**
```csv
time,i_d,i_q,n,u_d,u_q,run_id
0.0000,0.000000,0.000000,1000,0.000000,5.309292,1
0.0001,-0.007171,-0.366776,1000,0.204136,7.910853,1
...
0.2001,0.000000,0.000000,1000,0.000000,5.309292,2
0.2002,-0.007171,-0.366776,1000,0.204136,7.910853,2
```

### File Formats

Both outputs are available in two formats:
- **CSV:** Human-readable, larger file size (~100-150 MB each)
- **Parquet:** Compressed binary format, smaller size (3-15 MB), faster to load

## Process Description

### Pipeline Workflow

```
Input: 1000 CSV files
    ↓
1. File Discovery & Sorting
   - Find all files matching pattern (sim_*.csv)
   - Sort by run number extracted from filename
    ↓
2. Panel Format Generation
   - Read each CSV file
   - Add run_id column (extracted from filename)
   - Concatenate all dataframes
   - Sort by run_id and time
   - Output: merged_panel.csv/.parquet
    ↓
3. Stacked Format Generation
   - Read each CSV file
   - Normalize time to start at 0 for each run
   - Calculate time offset to make continuous
   - Shift time values by cumulative offset
   - Add run_id column
   - Concatenate all dataframes
   - Sort by continuous time
   - Output: merged_stacked.csv/.parquet
```

### Key Processing Steps

#### 1. **Run ID Parsing**
- Extracts numeric ID from filenames using regex `(\d+)`
- Example: `sim_0042.csv` → run_id = 42

#### 2. **Time Normalization (Stacked Mode)**
- Each run's time values are shifted to start at 0
- Original: `[t₀, t₁, ..., tₙ]`
- Normalized: `[0, t₁-t₀, ..., tₙ-t₀]`

#### 3. **Time Continuity (Stacked Mode)**
- Calculates median time step (dt) for each run
- Adds cumulative offset to maintain continuity
- Offset = previous run's last time + dt
- Prevents overlapping time values between runs

#### 4. **Data Concatenation**
- Combines all runs using pandas `concat()`
- Uses `ignore_index=True` to reset row indices
- Employs stable `mergesort` for consistent ordering

#### 5. **Output Generation**
- Saves CSV format for compatibility
- Attempts Parquet format for efficiency (requires pyarrow)
- Creates output directory if it doesn't exist

## Usage

### Quick Run Commands

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
python merge_simulation_data.py data
```

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
python merge_simulation_data.py data
```

### Quick Start (Windows)

**For detailed Windows setup instructions, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md)**

1. **One-time setup:**
   ```cmd
   setup.bat
   ```
   or
   ```powershell
   .\setup.ps1
   ```

2. **Run the pipeline:**
   ```cmd
   run.bat
   ```
   or
   ```powershell
   .\run.ps1
   ```

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   - pandas >= 2.0.0
   - numpy >= 1.24.0
   - pyarrow >= 12.0.0 (optional, for Parquet support)

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows Command Prompt:
venv\Scripts\activate.bat
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Basic usage (both formats):**
```bash
python merge_simulation_data.py data
```

**Generate only panel format:**
```bash
python merge_simulation_data.py data --mode panel
```

**Generate only stacked format:**
```bash
python merge_simulation_data.py data --mode stack
```

**Custom output directory:**
```bash
python merge_simulation_data.py data --out-dir output/
```

**Custom file pattern:**
```bash
python merge_simulation_data.py data --pattern "simulation_*.csv"
```

**Custom time column name:**
```bash
python merge_simulation_data.py data --time-col timestamp
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_dir` | Directory containing CSV files | Required |
| `--pattern` | Glob pattern for file matching | `sim_*.csv` |
| `--time-col` | Name of time column | `time` |
| `--out-dir` | Output directory path | `<input_dir>/merged` |
| `--mode` | Output mode: panel/stack/both | `both` |

### Example Output

```
Wrote files: ('data/merged/merged_panel.csv', 'data/merged/merged_stacked.csv')
Input files: 1000
Panel shape: (2001000, 7)
Stacked shape: (2001000, 7)
```

## Data Exploration

### Using the Jupyter Notebook

A comprehensive Jupyter notebook is provided for data exploration:

```bash
# Activate virtual environment
venv\Scripts\activate.bat  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install jupyter if not already installed
pip install jupyter matplotlib seaborn

# Launch notebook
jupyter notebook data_exploration.ipynb
```

The notebook includes:
- Data loading and structure overview
- Statistical summaries
- Single run detailed analysis
- Multi-run comparisons
- Time series visualization
- Correlation analysis
- Distribution plots
- Custom analysis examples (power calculation)

### Data Loading Examples

```python
import pandas as pd

# Load panel format
df_panel = pd.read_csv('data/merged/merged_panel.csv')

# Load stacked format
df_stacked = pd.read_csv('data/merged/merged_stacked.csv')

# Load Parquet (faster)
df_panel = pd.read_parquet('data/merged/merged_panel.parquet')
df_stacked = pd.read_parquet('data/merged/merged_stacked.parquet')
```

### Filtering Specific Runs

```python
# Get data for run 42
run_42 = df_panel[df_panel['run_id'] == 42]

# Get first 10 runs
first_10 = df_panel[df_panel['run_id'] <= 10]
```

### Time Series Analysis

```python
# Plot continuous time series
import matplotlib.pyplot as plt

plt.plot(df_stacked['time'], df_stacked['i_q'])
plt.xlabel('Time (s)')
plt.ylabel('Quadrature Current (A)')
plt.show()
```

## Project Structure

```
Data-Pipeline/
├── data/
│   ├── sim_0001.csv
│   ├── sim_0002.csv
│   ├── ...
│   ├── sim_1000.csv
│   └── merged/
│       ├── merged_panel.csv
│       ├── merged_panel.parquet
│       ├── merged_stacked.csv
│       └── merged_stacked.parquet
├── venv/                    # Virtual environment
├── merge_simulation_data.py  # Data merging script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Performance

- **Processing time:** ~10-30 seconds for 1000 files
- **Memory usage:** ~500-800 MB peak
- **Output sizes:**
  - CSV: ~120 MB per file
  - Parquet: 3-15 MB per file (compressed)

## Notes

- All input files must have the same column structure
- The `time` column must exist in all input files
- Files are processed in numerical order based on filename
- Missing or malformed files will cause the pipeline to exit with error
- Parquet output is optional and skipped if pyarrow is not installed
