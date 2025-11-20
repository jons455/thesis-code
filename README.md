# PMSM FOC Controller Neural Network Training

This repository contains the code and data preparation pipeline for training neural network models to replicate a Field-Oriented Control (FOC) controller for Permanent Magnet Synchronous Motors (PMSM).

## Repository Structure

```
thesis-code/
├── pmsm-matlab/              # MATLAB/Simulink simulation model
│   ├── foc_pmsm.slx         # FOC controller Simulink model
│   ├── pmsm_init.m          # Motor parameters and simulation setup
│   └── export/              # Generated simulation data (excluded from git)
│
├── data-preparation/         # Data processing pipeline
│   ├── main.py              # Data merging and preprocessing
│   ├── prepare_edge_impulse.py  # Edge Impulse data preparation
│   ├── data_exploration.ipynb   # Data analysis notebook
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Data preparation documentation
│
└── docs/                     # Documentation
    ├── DATA_GENERATION.md   # MATLAB simulation documentation
    └── TRAINING_GUIDE.md     # Neural network training guide
```

## Overview

### Objective
Train neural networks (ANN and SNN) to replicate the behavior of a MATLAB-implemented FOC controller for PMSM motor control.

### Data Generation
- **Simulation Tool:** MATLAB/Simulink
- **Model:** FOC PMSM controller (`foc_pmsm.slx`)
- **Runs:** 1000 simulations with randomized operating points
- **Sampling Rate:** 10 kHz (100 μs timestep)
- **Duration per Run:** 200 ms
- **Total Samples:** 2,001,000 data points

### Motor Specifications
- Nominal Current: 4.2 A
- Maximum Current: 10.8 A
- DC Bus Voltage: 48 V
- Nominal Speed: 3000 RPM
- Direct-axis Inductance (L_d): 1.13 mH
- Quadrature-axis Inductance (L_q): 1.42 mH
- Stator Resistance (R_s): 0.543 Ω
- Pole Pairs: 3
- Permanent Magnet Flux (Ψ_PM): 16.9 mWb

## Quick Start

### 1. MATLAB Simulation

```matlab
% Navigate to pmsm-matlab directory
cd pmsm-matlab

% Run simulation script
pmsm_init.m

% Output: 1000 CSV files in export/train/
```

### 2. Data Preparation

```bash
# Navigate to data-preparation directory
cd data-preparation

# Install dependencies
pip install -r requirements.txt

# Merge simulation data
python main.py data

# Prepare data for Edge Impulse
python prepare_edge_impulse.py
```

### 3. Neural Network Training

See `docs/TRAINING_GUIDE.md` for detailed instructions on training ANN and SNN models.

## Data Format

### Input Features
- `i_d`: Direct-axis current (A)
- `i_q`: Quadrature-axis current (A)
- `n`: Rotational speed (RPM)

### Output Targets
- `u_d`: Direct-axis voltage command (V)
- `u_q`: Quadrature-axis voltage command (V)

### Data Files
- **Panel Format:** `data-preparation/data/merged/merged_panel.csv`
  - Preserves run structure with `run_id` column
  - Useful for run-by-run analysis
  
- **Stacked Format:** `data-preparation/data/merged/merged_stacked.csv`
  - Continuous time series across all runs
  - Useful for time-series analysis

## Requirements

### MATLAB
- MATLAB R2020b or later
- Simulink
- Control System Toolbox

### Python
- Python 3.8+
- See `data-preparation/requirements.txt` for dependencies

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourthesis2025,
  title={Neural Network-Based Control for Permanent Magnet Synchronous Motors},
  author={Your Name},
  school={Your University},
  year={2025}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

