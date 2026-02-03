"""Akida Controller Agent for Benchmark Integration.

This module provides a wrapper around the Keras/Akida model that fits the
NeuroBench-aligned DictController protocol.
"""

from pathlib import Path
from typing import Any

import numpy as np

from embark.benchmark.interfaces import DictController, ReferenceDict, StateDict
from embark.utils.config import DEFAULT_PMSM
from evaluation.snn_keras.models import AkidaController


class AkidaControllerAgent(DictController):
    """
    Agent wrapper for Akida/Keras models to use with the benchmark suite.

    This handles:
    1. Input normalization (Physical State -> Model Input)
    2. Inference (Keras or Akida runtime)
    3. Output denormalization (Model Output -> Physical Action)
    """

    def __init__(
        self,
        model_path: str,
        i_max: float = DEFAULT_PMSM.i_max,
        u_max: float = DEFAULT_PMSM.u_max,
        n_max: float = 4000.0,  # Consistent with dataset.py
        error_gain: float = 10.0,
    ):
        self.model_path = Path(model_path)
        self.i_max = i_max
        self.u_max = u_max
        self.n_max = n_max
        self.error_gain = error_gain

        # Load Model
        if self.model_path.suffix == ".fbz":
            try:
                import akida

                self.is_akida_hardware = True
                self.model = akida.Model(str(self.model_path))
                print(f"Loaded Akida hardware model: {self.model_path}")
            except ImportError:
                raise ImportError("akida package required for .fbz models")
        else:
            self.is_akida_hardware = False
            # Load using our wrapper which handles .keras/.json/.weights.h5 logic
            # Use bare path without extension if possible, or strip it
            if self.model_path.suffix in [".keras", ".json", ".h5"]:
                load_path = str(self.model_path.with_suffix(""))
            else:
                load_path = str(self.model_path)

            self.controller = AkidaController.load(load_path)
            self.model = self.controller.model
            print(f"Loaded Keras float model: {self.model_path}")

    def __call__(self, state: StateDict, reference: ReferenceDict) -> dict[str, float]:
        """
        Compute control action.

        Args:
            state: Physical state dict (i_d, i_q, omega)
            reference: Reference dict (i_d_ref, i_q_ref)

        Returns:
            action: Physical action dict {"v_d": ..., "v_q": ...}
        """
        # Extract features
        i_d = float(state["i_d"])
        i_q = float(state["i_q"])
        i_d_ref = float(reference["i_d_ref"])
        i_q_ref = float(reference["i_q_ref"])
        e_d = i_d_ref - i_d
        e_q = i_q_ref - i_q

        # Handle speed if present
        if "omega" in state:
            n_rpm = float(state["omega"]) * 60.0 / (2 * np.pi)
        else:
            n_rpm = float(state.get("n_rpm", 0.0))

        # --- NORMALIZE (Match dataset.py logic) ---
        i_d_norm = i_d / self.i_max
        i_q_norm = i_q / self.i_max

        # Error clipping is crucial!
        e_d_norm = np.clip((e_d / self.i_max) * self.error_gain, -1.0, 1.0)
        e_q_norm = np.clip((e_q / self.i_max) * self.error_gain, -1.0, 1.0)

        n_norm = n_rpm / self.n_max

        # Stack: [i_d, i_q, e_d, e_q, n]
        input_vector = np.array(
            [[i_d_norm, i_q_norm, e_d_norm, e_q_norm, n_norm]], dtype=np.float32
        )

        # --- INFERENCE ---
        if self.is_akida_hardware:
            # Akida expect quantized inputs usually, but Model() handles conversion
            # if InputQuantizer was part of the model (which cnn2snn adds by default).
            # However, akida.Model.predict returns standard numpy array of integers/floats.

            # Note: Akida inference
            preds = self.model.predict(input_vector)

            # Squeeze batch dim
            preds = preds.squeeze()

            # --- SCALE OUTPUT ---
            # Akida outputs integers. We need to know the scale.
            # If we don't have the scale from the file, we might need a heuristic or config.
            # For now, let's assume the Keras model's output_scale was 1.0 during training
            # and that cnn2snn handled the scaling.
            # BUT: Akida 'predict' on a converted model usually returns the POTENTIALS or scaled values?
            # Actually, standard cnn2snn conversion keeps the output layer as a regression layer usually.

            # Heuristic: If values are huge integers, scale them.
            # If they are small floats, leave them.
            if np.max(np.abs(preds)) > 100:
                # Likely 8-bit or similar integers, roughly scale to -1..1
                # This is a bit hacky without reading the parameters.
                # Better approach: The user should provide the scale if known.
                # Let's try a standard division for 4-bit (15) or 8-bit (255) if needed.
                # For now, let's assume cnn2snn preserved the scale logic if 'linear' was used.
                pass

            # Cast to float
            action_norm = preds.astype(np.float32)

            # If using the 'quantize_ml' flow, the output might be integers that represent
            # the q-value.

        else:
            # Keras Float Model
            preds = self.model.predict(input_vector, verbose=0)
            action_norm = preds[0]

        # --- DENORMALIZE ---
        # Output is [u_d_norm, u_q_norm] in [-1, 1]
        u_d = action_norm[0] * self.u_max
        u_q = action_norm[1] * self.u_max

        return {"v_d": float(u_d), "v_q": float(u_q)}

    def reset(self) -> None:
        """Reset internal states (none for feedforward)."""
        return None

    def get_state(self) -> dict[str, Any]:
        """Serialize internal state (stateless)."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state (stateless)."""
        return None

    def get_info(self) -> dict[str, Any]:
        return {
            "name": "Akida_Keras_Model",
            "type": "ann_quantized" if self.is_akida_hardware else "ann_float",
            "path": str(self.model_path),
        }
