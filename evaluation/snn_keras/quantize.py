"""
Export existing Keras model to Akida .fbz format.

Use this if training finished but export failed.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.snn_keras.models import AkidaController
from evaluation.snn_keras.utils.dataset import PMSMKerasDataset

def main():
    # Paths
    model_path = PROJECT_ROOT / "trained_models/akida/final_model"
    data_dir = PROJECT_ROOT / "data/raw/train"
    
    print(f"Loading model from: {model_path}")
    controller = AkidaController.load(str(model_path))
    
    print("Loading calibration data...")
    dataset = PMSMKerasDataset(
        data_dir=str(data_dir),
        window_size=100,
        max_files=5 # Need just a few for calibration
    )
    x_cal, _ = dataset.get_flattened_arrays()
    calibration_data = x_cal[:1000]
    
    print("Quantizing and Exporting...")
    # This will use the FIXED quantization call
    controller.export_akida(str(PROJECT_ROOT / "trained_models/akida/akida_model.fbz"), calibration_data)
    
    print("Done!")

if __name__ == "__main__":
    main()
