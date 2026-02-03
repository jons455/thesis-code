import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embark.utils.paths import DATA_RAW_DIR, MODELS_CHECKPOINTS_DIR  # noqa: E402
from evaluation.snn.utils.dataset import PMSMDataset  # noqa: E402
from evaluation.snn.models import (  # noqa: E402
    DeltaSNNController,
    LearnedLinearSNNController,
    MembraneSNNController,
    PopulationSNNController,
    SNNConfig,
    TTFSSNNController,
    RecurrentSNNController,
)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = str(
        DATA_RAW_DIR / "train"
    )  # Clean training data (100% PI tracking)
    window_size: int = 100
    stride: int = 50
    val_split: float = 0.2
    error_gain: float = 10.0

    # Model
    model_type: str = (
        "membrane"  # "membrane", "population", "learned_linear", "delta", "ttfs", "recurrent"
    )
    hidden_size: int = 64
    num_hidden_layers: int = 2
    beta_hidden: float = 0.9
    beta_output: float = 0.995
    neurons_per_output: int = 50  # For population coding
    delta_scale: float = 0.01  # For delta coding
    delta_beta: float = 0.8  # For delta coding
    output_scale: float = 0.1  # For membrane/learned linear output scaling
    ttfs_time_window: int = 20  # For TTFS coding
    ttfs_beta_output: float = 0.9  # For TTFS output neurons
    ttfs_learn_beta: bool = True  # For TTFS learnable decay

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Scheduling
    scheduler: str = "cosine"  # "cosine", "step", or "none"

    # Checkpoints
    # Will be updated in main() to include model_type subdirectory
    checkpoint_dir: str = str(MODELS_CHECKPOINTS_DIR)
    run_name: str | None = None
    save_every: int = 10  # Save checkpoint every N epochs
    resume_from: str | None = None  # Path to checkpoint to resume from
    start_epoch: int = 1  # Starting epoch (for resume)

    # Debugging
    max_files: int | None = None  # Limit files for quick testing

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # --- FIX: Initialize State with Teacher Forcing ---
        batch_size = inputs.shape[0]

        # Get the default zero-initialized state
        state = model.init_state(batch_size, device)

        # If this is a Delta model (which acts as an integrator), we MUST set
        # the initial voltage accumulator to the actual start value of the window.
        if hasattr(model, "delta_scale"):
            state_list = list(state)

            # The last element in DeltaSNN state is the voltage accumulator
            # targets[:, 0, :] is the true voltage at timestep t=0 of this window
            state_list[-1] = targets[:, 0, :].clone()

            state = tuple(state_list)
        # --------------------------------------------------

        # Pass the correctly initialized state to the model
        outputs, _, _ = model.forward_sequence(inputs, state=state)

        # MSE loss on voltage predictions
        loss = F.mse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Validate model, return metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_errors = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # --- FIX: Initialize State with Teacher Forcing ---
        batch_size = inputs.shape[0]
        state = model.init_state(batch_size, device)

        if hasattr(model, "delta_scale"):
            state_list = list(state)
            state_list[-1] = targets[:, 0, :].clone()
            state = tuple(state_list)

        outputs, _, _ = model.forward_sequence(inputs, state=state)

        loss = F.mse_loss(outputs, targets)
        total_loss += loss.item() * inputs.shape[0]
        total_samples += inputs.shape[0]

        # Track absolute errors
        errors = (outputs - targets).abs()
        all_errors.append(errors.cpu())

    all_errors = torch.cat(all_errors, dim=0)

    return {
        "loss": total_loss / total_samples,
        "mae": all_errors.mean().item(),
        "mae_u_d": all_errors[:, :, 0].mean().item(),
        "mae_u_q": all_errors[:, :, 1].mean().item(),
        "max_error": all_errors.max().item(),
    }


def train(config: TrainConfig) -> nn.Module:
    """
    Main training function.

    Args:
        config: Training configuration.

    Returns:
        Trained SNN model.

    """
    # Create checkpoint directory with model type subdirectory
    # We update it here to ensure it uses the specific model type folder
    base_checkpoint_dir = Path(config.checkpoint_dir)
    # Check if 'trained_models' is already in the path or if we need to add it
    # The config default is MODELS_CHECKPOINTS_DIR, which usually points to a generic location.
    # The user request is specifically "trained_models/{model_type}".
    # Let's construct it cleanly.

    # Assuming standard project structure where we want trained_models at root or similar.
    # But using the passed checkpoint_dir as base is safer.
    # If checkpoint_dir ends with 'checkpoints', we might want to step up or just append.
    # Let's just append model_type to keep it organized.

    # However, user said "safed in the trained_models/ and then the name of the model_type".
    # Let's try to honor that path structure relative to project root if possible,
    # or just use the provided checkpoint dir + model_type.

    # Let's enforce the subdirectory structure:
    if config.run_name:
        model_dir = base_checkpoint_dir / config.run_name
    else:
        model_dir = base_checkpoint_dir / config.model_type

    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PMSM SNN Controller Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Data: {config.data_dir}")
    print(f"Model Type: {config.model_type}")
    print(f"Run Name: {config.run_name}")
    print(f"Checkpoint Dir: {model_dir}")
    print(f"Error Gain: {config.error_gain}")
    print(f"Hidden size: {config.hidden_size}")
    if config.model_type in {"population", "learned_linear"}:
        print(f"Neurons/Output: {config.neurons_per_output}")
    if config.model_type == "delta":
        print(f"Delta scale: {config.delta_scale}")
    if config.model_type == "ttfs":
        print(f"TTFS window: {config.ttfs_time_window}")
    print(f"Epochs: {config.epochs}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")

    if config.max_files:
        # Quick test mode - load limited data
        dataset = PMSMDataset(
            data_dir=config.data_dir,
            window_size=config.window_size,
            stride=config.stride,
            max_files=config.max_files,
            error_gain=config.error_gain,
        )

        # Manual split
        val_size = max(1, int(len(dataset) * config.val_split))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
    else:
        # We need to update create_dataloaders to pass error_gain too,
        # or just instantiate manually here to avoid changing signature of create_dataloaders if it is used elsewhere.
        # Let's instantiate manually to be safe and clear.

        full_dataset = PMSMDataset(
            data_dir=config.data_dir,
            window_size=config.window_size,
            stride=config.stride,
            error_gain=config.error_gain,
        )

        val_size = int(len(full_dataset) * config.val_split)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create or load model
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=config.device)
        # We rely on the config to instantiate the right class
        # Ideally we'd use the checkpoint config, but for resume we often want to
        # keep command line args or mix.
        # Let's instantiate fresh and load state dict.
        pass

    print("\nCreating model...")
    snn_config = SNNConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        beta_hidden=config.beta_hidden,
        beta_output=config.beta_output,
        neurons_per_output=config.neurons_per_output,
        delta_scale=config.delta_scale,
        delta_beta=config.delta_beta,
        output_scale=config.output_scale,
        ttfs_time_window=config.ttfs_time_window,
        ttfs_beta_output=config.ttfs_beta_output,
        ttfs_learn_beta=config.ttfs_learn_beta,
    )

    if config.model_type == "population":
        model = PopulationSNNController(config=snn_config)
    elif config.model_type == "learned_linear":
        model = LearnedLinearSNNController(config=snn_config)
    elif config.model_type == "delta":
        model = DeltaSNNController(config=snn_config)
    elif config.model_type == "ttfs":
        model = TTFSSNNController(config=snn_config)
    elif config.model_type == "recurrent":
        model = RecurrentSNNController(config=snn_config)
    else:
        model = MembraneSNNController(config=snn_config)

    model = model.to(config.device)

    print(
        f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    if config.resume_from:
        checkpoint = torch.load(config.resume_from, map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    # Training loop
    print("\nTraining...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    # Load existing history if resuming
    history_path = model_dir / "history.json"
    if config.resume_from and history_path.exists():
        import json

        with open(history_path) as f:
            history = json.load(f)
        print(f"Loaded history with {len(history['train_loss'])} previous epochs")
        if history["val_loss"]:
            best_val_loss = min(history["val_loss"])

    for epoch in range(config.start_epoch, config.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, config.device, config.grad_clip
        )

        # Validate
        val_metrics = validate(model, val_loader, config.device)

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])

        # Check for best model
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            model.save(model_dir / "best_model.pt")

        # Print progress
        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_loss:.2e} | "  # Scientific notation
            f"Val: {val_metrics['loss']:.2e} | "  # Scientific notation
            f"MAE: {val_metrics['mae']:.4f} | "
            f"LR: {lr:.2e}" + (" *" if is_best else "")
        )
        print(msg, flush=True)

        # Periodic checkpoint
        if epoch % config.save_every == 0:
            model.save(model_dir / f"epoch_{epoch:03d}.pt")

    # Save final model
    model.save(model_dir / "final_model.pt")

    # Save training history
    import json

    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {model_dir}")
    print("=" * 60)

    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train PMSM SNN Controller")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_RAW_DIR / "train"),
        help="Directory with training CSV files (generated by scripts/generate_training_data.py)",
    )
    # Changed default to trained_models
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="trained_models",
        help="Directory to save checkpoints (subfolder per model type or run_name will be created)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Specific name for this training run (overrides model_type folder name)",
    )
    parser.add_argument(
        "--window_size", type=int, default=100, help="Timesteps per training window"
    )
    parser.add_argument("--stride", type=int, default=50, help="Stride between windows")
    parser.add_argument(
        "--error_gain",
        type=float,
        default=10.0,
        help="Amplification factor for error signals",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="membrane",
        choices=[
            "membrane",
            "population",
            "learned_linear",
            "delta",
            "ttfs",
            "recurrent",
        ],
        help=(
            "Model architecture: membrane, population, learned_linear, delta, ttfs, or recurrent"
        ),
    )
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--beta_output", type=float, default=0.995, help="Output layer decay rate"
    )
    parser.add_argument(
        "--neurons_per_output",
        type=int,
        default=50,
        help="Neurons per output dimension (population/learned_linear)",
    )
    parser.add_argument(
        "--delta_scale",
        type=float,
        default=0.01,
        help="Voltage increment per net spike (delta model)",
    )
    parser.add_argument(
        "--delta_beta",
        type=float,
        default=0.8,
        help="Output decay for delta spikes (delta model)",
    )
    parser.add_argument(
        "--output_scale",
        type=float,
        default=0.1,
        help="Output scaling (membrane/learned_linear)",
    )
    parser.add_argument(
        "--ttfs_time_window",
        type=int,
        default=20,
        help="Internal TTFS time window per control cycle",
    )
    parser.add_argument(
        "--ttfs_beta_output",
        type=float,
        default=0.9,
        help="Output decay rate for TTFS",
    )
    parser.add_argument(
        "--ttfs_learn_beta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learnable TTFS output decay",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Debug arguments
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of files (for quick testing)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Resume arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=1, help="Starting epoch number (for resume)"
    )

    args = parser.parse_args()

    # Create config from arguments
    config = TrainConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        window_size=args.window_size,
        stride=args.stride,
        error_gain=args.error_gain,
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        beta_output=args.beta_output,
        neurons_per_output=args.neurons_per_output,
        delta_scale=args.delta_scale,
        delta_beta=args.delta_beta,
        output_scale=args.output_scale,
        ttfs_time_window=args.ttfs_time_window,
        ttfs_beta_output=args.ttfs_beta_output,
        ttfs_learn_beta=args.ttfs_learn_beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_files=args.max_files,
        resume_from=args.resume,
        start_epoch=args.start_epoch,
    )

    if args.device:
        config.device = args.device

    # Train
    train(config)


if __name__ == "__main__":
    main()
