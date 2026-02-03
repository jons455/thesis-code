"""Training script for Akida-compatible PMSM controller.

This script provides the complete training pipeline:
1. Load and preprocess data
2. Train the float32 Keras model
3. Quantize to 4-bit for Akida
4. Convert to Akida SNN format
5. Export to .fbz for Raspberry Pi deployment

Usage:
    python -m evaluation.snn_keras.utils.train --data_dir data/raw/train --epochs 100

    # With quick test mode:
    python -m evaluation.snn_keras.utils.train --data_dir data/raw/train --max_files 5 --epochs 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tf_keras as keras
# from tensorflow import keras

from evaluation.snn_keras.models import AkidaConfig, AkidaController
from evaluation.snn_keras.utils.dataset import PMSMKerasDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = "data/raw/train"
    window_size: int = 100
    stride: int = 50
    val_split: float = 0.2
    error_gain: float = 10.0
    max_files: int | None = None

    # Model architecture
    hidden_sizes: list[int] | None = None  # Default: [64, 64]
    use_batch_norm: bool = False
    dropout_rate: float = 0.0

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Scheduling
    scheduler: str = "cosine"  # "cosine", "step", "none"

    # Checkpoints
    checkpoint_dir: str = "trained_models/akida"
    run_name: str | None = None
    save_every: int = 10

    # Akida export
    export_akida: bool = True
    quantize_epochs: int = 10  # Fine-tuning epochs after quantization

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]


def create_callbacks(
    config: TrainConfig,
    model_dir: Path,
) -> list[keras.callbacks.Callback]:
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint (save best)
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    # Periodic checkpoint
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "epoch_{epoch:03d}.keras"),
            save_freq=config.save_every * config.batch_size,  # Every N epochs
            save_weights_only=False,
            verbose=0,
        )
    )

    # Learning rate scheduler
    if config.scheduler == "cosine":
        callbacks.append(
            keras.callbacks.LearningRateScheduler(
                lambda epoch: config.learning_rate
                * (0.5 * (1 + np.cos(np.pi * epoch / config.epochs))),
                verbose=0,
            )
        )
    elif config.scheduler == "step":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1,
            )
        )

    # Early stopping (optional, conservative)
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # CSV logger
    callbacks.append(
        keras.callbacks.CSVLogger(
            str(model_dir / "training_log.csv"),
            append=True,
        )
    )

    return callbacks


def train(config: TrainConfig) -> tuple[AkidaController, dict]:
    """Main training function.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (trained_controller, history_dict).
    """
    # Setup directories
    if config.run_name:
        model_dir = Path(config.checkpoint_dir) / config.run_name
    else:
        model_dir = Path(config.checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PMSM Akida Controller Training (Keras/TensorFlow)")
    print("=" * 60)
    print(f"Data: {config.data_dir}")
    print(f"Hidden sizes: {config.hidden_sizes}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Checkpoint dir: {model_dir}")
    print(f"Export Akida: {config.export_akida}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    dataset = PMSMKerasDataset(
        data_dir=config.data_dir,
        window_size=config.window_size,
        stride=config.stride,
        error_gain=config.error_gain,
        max_files=config.max_files,
    )

    train_ds, val_ds = dataset.train_val_split(val_split=config.val_split)

    # Get flattened data for feed-forward training
    x_train, y_train = train_ds.get_flattened_arrays()
    x_val, y_val = val_ds.get_flattened_arrays()

    print(f"Train samples: {len(x_train):,}")
    print(f"Val samples: {len(x_val):,}")

    # Create model
    print("\nCreating model...")
    akida_config = AkidaConfig(
        hidden_sizes=config.hidden_sizes,
        use_batch_norm=config.use_batch_norm,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    controller = AkidaController(config=akida_config)
    controller.compile(optimizer="adamw", loss="mse")
    controller.summary()

    print(f"\nParameters: {controller.count_parameters():,}")

    # Save config
    with open(model_dir / "config.json", "w") as f:
        json.dump(akida_config.to_dict(), f, indent=2)

    with open(model_dir / "train_config.json", "w") as f:
        json.dump(
            {
                "data_dir": config.data_dir,
                "window_size": config.window_size,
                "stride": config.stride,
                "error_gain": config.error_gain,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
            f,
            indent=2,
        )

    # Create callbacks
    callbacks = create_callbacks(config, model_dir)

    # Train
    print("\nTraining...")
    history = controller.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    controller.save(str(model_dir / "final_model"))

    # Save training history
    history_dict = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
    }
    with open(model_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    print("\n" + "=" * 60)
    print("Float32 Training Complete!")
    print(f"Best validation loss: {min(history_dict['val_loss']):.6f}")
    print("=" * 60)

    return controller, history_dict


def evaluate(
    model_path: str,
    data_dir: str,
    window_size: int = 100,
    error_gain: float = 10.0,
) -> dict[str, float]:
    """Evaluate a trained model.

    Args:
        model_path: Path to saved model.
        data_dir: Directory with test data.
        window_size: Window size for data loading.
        error_gain: Error signal amplification.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Load model
    controller = AkidaController.load(model_path)
    controller.compile(loss="mse")

    # Load data
    dataset = PMSMKerasDataset(
        data_dir=data_dir,
        window_size=window_size,
        error_gain=error_gain,
    )

    x, y = dataset.get_flattened_arrays()

    # Evaluate
    loss = controller.model.evaluate(x, y, verbose=0)
    predictions = controller.predict(x)

    # Compute metrics
    errors = np.abs(predictions - y)

    return {
        "mse": float(loss),
        "mae": float(errors.mean()),
        "mae_u_d": float(errors[:, 0].mean()),
        "mae_u_q": float(errors[:, 1].mean()),
        "max_error": float(errors.max()),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Akida-compatible PMSM controller"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/train",
        help="Directory with training CSV files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="trained_models/akida",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Specific name for this training run",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Timesteps per training window",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Stride between windows",
    )
    parser.add_argument(
        "--error_gain",
        type=float,
        default=10.0,
        help="Error signal amplification factor",
    )

    # Model arguments
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden layer sizes (e.g., --hidden_sizes 64 64)",
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Enable batch normalization",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (0.0 to disable)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate scheduler",
    )

    # Akida arguments
    parser.add_argument(
        "--no_akida",
        action="store_true",
        help="Skip Akida export (train float32 model only)",
    )
    parser.add_argument(
        "--quantize_epochs",
        type=int,
        default=10,
        help="Fine-tuning epochs after quantization",
    )

    # Debug arguments
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of files (for quick testing)",
    )

    args = parser.parse_args()

    # Create config
    config = TrainConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        window_size=args.window_size,
        stride=args.stride,
        error_gain=args.error_gain,
        hidden_sizes=args.hidden_sizes,
        use_batch_norm=args.use_batch_norm,
        dropout_rate=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        scheduler=args.scheduler,
        export_akida=not args.no_akida,
        quantize_epochs=args.quantize_epochs,
        max_files=args.max_files,
    )

    # Train
    train(config)


if __name__ == "__main__":
    main()
