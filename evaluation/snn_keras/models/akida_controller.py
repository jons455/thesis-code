"""Akida-compatible Keras model for PMSM control.

This implements a feed-forward neural network that is compatible with
BrainChip's Akida neuromorphic processor (version 1.0).

Key constraints for Akida 1.0:
- Only Dense layers (no Conv2D needed for this use case)
- ReLU activations only (no Tanh, Sigmoid, LeakyReLU)
- No recurrent layers (LSTM, GRU)
- Linear output for regression
- Quantization to 4-bit weights/activations

The model structure mirrors the PyTorch LearnedLinearSNNController
but uses standard Keras layers that Akida can convert.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models

from .config import AkidaConfig


def create_akida_controller(
    config: AkidaConfig | None = None,
    name: str = "akida_controller",
) -> keras.Model:
    """Create an Akida-compatible controller model.

    This function creates a standard Keras Sequential model that follows
    all Akida 1.0 constraints. The model can be:
    1. Trained normally with float32
    2. Quantized using quantizeml
    3. Converted to Akida format using cnn2snn

    Args:
        config: Model configuration. Defaults to AkidaConfig().
        name: Model name.

    Returns:
        Keras Model ready for training.

    Example:
        >>> config = AkidaConfig(hidden_sizes=[64, 64])
        >>> model = create_akida_controller(config)
        >>> model.compile(optimizer='adam', loss='mse')
        >>> model.fit(x_train, y_train, epochs=100)
    """
    if config is None:
        config = AkidaConfig()

    # Build model using Functional API for flexibility
    inputs = layers.Input(shape=(config.input_size,), name="inputs")

    x = inputs

    # Hidden layers: Dense + ReLU (Akida constraint)
    for i, hidden_size in enumerate(config.hidden_sizes):
        x = layers.Dense(
            hidden_size,
            activation="relu",  # MUST be ReLU for Akida 1.0
            name=f"hidden_{i}",
            kernel_regularizer=keras.regularizers.l2(config.weight_decay),
        )(x)

        # Optional batch normalization (before activation in original,
        # but after is more common for Akida compatibility)
        if config.use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)

        # Optional dropout (only during training)
        if config.dropout_rate > 0:
            x = layers.Dropout(config.dropout_rate, name=f"dropout_{i}")(x)

    # Output layer: Linear activation for regression
    # Akida will output quantized integers; scale in post-processing
    outputs = layers.Dense(
        config.output_size,
        activation="linear",  # Linear for regression
        name="output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)

    return model


class AkidaController:
    """Wrapper class for Akida-compatible PMSM controller.

    This class provides a convenient interface matching the PyTorch
    model API while wrapping a Keras model internally.

    Attributes:
        config: Model configuration.
        model: The underlying Keras model.

    Example:
        >>> controller = AkidaController(AkidaConfig(hidden_sizes=[32, 32]))
        >>> controller.compile()
        >>> controller.fit(x_train, y_train, epochs=50)
        >>> controller.export_akida("model.fbz")
    """

    def __init__(
        self,
        config: AkidaConfig | None = None,
        hidden_size: int | None = None,
    ):
        """Initialize the controller.

        Args:
            config: Model configuration.
            hidden_size: Override hidden size (for compatibility with PyTorch API).
        """
        self.config = config or AkidaConfig()

        # Allow hidden_size override for API compatibility
        if hidden_size is not None:
            self.config.hidden_sizes = [hidden_size, hidden_size]

        self.model = create_akida_controller(self.config)
        self._quantized_model = None
        self._akida_model = None

    def compile(
        self,
        optimizer: str | keras.optimizers.Optimizer = "adam",
        loss: str = "mse",
        learning_rate: float | None = None,
        **kwargs,
    ) -> None:
        """Compile the model for training.

        Args:
            optimizer: Optimizer name or instance.
            loss: Loss function.
            learning_rate: Override learning rate from config.
            **kwargs: Additional arguments to model.compile().
        """
        lr = learning_rate or self.config.learning_rate

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                optimizer = keras.optimizers.Adam(learning_rate=lr)
            elif optimizer.lower() == "adamw":
                optimizer = keras.optimizers.AdamW(
                    learning_rate=lr,
                    weight_decay=self.config.weight_decay,
                )
            elif optimizer.lower() == "sgd":
                optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def fit(self, *args, **kwargs):
        """Train the model. See keras.Model.fit()."""
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Run inference. See keras.Model.predict()."""
        return self.model.predict(*args, **kwargs)

    def __call__(self, inputs, training: bool = False):
        """Forward pass."""
        return self.model(inputs, training=training)

    def summary(self):
        """Print model summary."""
        return self.model.summary()

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(
            tf.reduce_prod(var.shape).numpy() for var in self.model.trainable_variables
        )

    def save(self, path: str) -> None:
        """Save model weights and config.

        Args:
            path: Path to save (without extension).
        """
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.model.save_weights(str(path.with_suffix(".weights.h5")))

        # Save config
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Also save full model in Keras format
        self.model.save(str(path.with_suffix(".keras")))

    @classmethod
    def load(cls, path: str) -> "AkidaController":
        """Load model from saved weights and config.

        Args:
            path: Path to saved model (without extension).

        Returns:
            Loaded AkidaController.
        """
        import json
        from pathlib import Path

        path = Path(path)

        # Load config
        with open(path.with_suffix(".json")) as f:
            config_dict = json.load(f)

        config = AkidaConfig.from_dict(config_dict)
        controller = cls(config=config)

        # Load weights
        controller.model.load_weights(str(path.with_suffix(".weights.h5")))

        return controller

    def quantize(self, calibration_data=None):
        """Quantize model for Akida deployment.

        This converts the float32 model to 4-bit quantized format
        suitable for Akida hardware.

        Args:
            calibration_data: Optional data for calibration-based quantization.
                             Shape: (num_samples, input_size)

        Returns:
            Quantized Keras model.

        Raises:
            ImportError: If quantizeml is not installed.
        """
        try:
            import quantizeml
        except ImportError:
            raise ImportError(
                "quantizeml is required for quantization. "
                "Install with: pip install akida-models"
            )

        print("Quantizing model to 4-bit...")

        # Quantize using QuantizeML
        from quantizeml.models import QuantizationParams

        # Explicitly tell QuantizeML that inputs are signed (e.g. current error +/-)
        # using input_weight_bits=8 is correct, but we must handle the sign mismatch.
        # Akida usually expects unsigned inputs (uint8) for the first layer unless configured.
        # We can try per_tensor_activations=True or ensure InputQuantizer handles it.

        qparams = QuantizationParams(
            weight_bits=self.config.weight_bits,
            activation_bits=self.config.activation_bits,
            input_weight_bits=8,
            input_dtype="int8",  # CRITICAL: Our inputs are signed currents/errors
        )

        self._quantized_model = quantizeml.models.quantize(
            self.model, qparams=qparams, samples=calibration_data
        )

        # Optional: Calibrate with data for better accuracy
        if calibration_data is not None:
            print("Calibrating quantized model...")
            # Run inference on calibration data to adjust quantization ranges
            _ = self._quantized_model.predict(calibration_data[:100])

        return self._quantized_model

    def convert_to_akida(self):
        """Convert quantized model to Akida format.

        Must call quantize() first.

        Returns:
            Akida model ready for hardware deployment.

        Raises:
            ImportError: If cnn2snn is not installed.
            RuntimeError: If model has not been quantized.
        """
        if self._quantized_model is None:
            raise RuntimeError("Must call quantize() before convert_to_akida()")

        try:
            from cnn2snn import convert
        except ImportError:
            raise ImportError(
                "cnn2snn is required for Akida conversion. "
                "Install with: pip install akida"
            )

        print("Converting to Akida SNN format...")
        self._akida_model = convert(self._quantized_model)

        return self._akida_model

    def export_akida(self, path: str, calibration_data=None) -> None:
        """Full export pipeline: quantize, convert, and save for Akida.

        This performs the complete export process:
        1. Quantize the float model to 4-bit
        2. Convert to Akida SNN format
        3. Save as .fbz file for Raspberry Pi deployment

        Args:
            path: Output path for .fbz file.
            calibration_data: Optional calibration data for quantization.
        """
        from pathlib import Path

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".fbz")

        # Quantize if not already done
        if self._quantized_model is None:
            self.quantize(calibration_data)

        # Convert to Akida
        if self._akida_model is None:
            self.convert_to_akida()

        # Save
        path.parent.mkdir(parents=True, exist_ok=True)
        self._akida_model.save(str(path))
        print(f"Akida model saved to: {path}")

    def get_network_stats(self) -> dict:
        """Get network architecture statistics (for compatibility)."""
        total_neurons = 0
        total_synapses = 0

        for layer in self.model.layers:
            if hasattr(layer, "units"):
                total_neurons += layer.units
            if hasattr(layer, "kernel"):
                total_synapses += tf.reduce_prod(layer.kernel.shape).numpy()

        return {
            "num_neurons": total_neurons,
            "num_synapses": int(total_synapses),
            "num_layers": len(self.config.hidden_sizes) + 1,
            "hidden_sizes": self.config.hidden_sizes,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
        }
