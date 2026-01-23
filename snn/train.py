"""Training script for PMSM SNN controller.

Trains a SimpleSNNController to imitate PI controller behavior using
supervised learning on trajectory data.

Example:
    Basic training::

        python -m snn.train

    With custom parameters::

        python -m snn.train --epochs 100 --batch_size 64 --hidden_size 128

    Quick test run::

        python -m snn.train --epochs 5 --max_files 10

The script loads PI controller trajectories, trains the SNN to predict
voltage commands, saves the best checkpoint, and generates training curves.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from snn.dataset import PMSMDataset, create_dataloaders
from snn.models import SimpleSNNController, SNNConfig


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = "pmsm-pem/export/train_v2"  # Clean training data (100% PI tracking)
    window_size: int = 100
    stride: int = 50
    val_split: float = 0.2

    # Model
    hidden_size: int = 64
    num_hidden_layers: int = 2
    beta_hidden: float = 0.9
    beta_output: float = 0.995

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Scheduling
    scheduler: str = "cosine"  # "cosine", "step", or "none"

    # Checkpoints
    checkpoint_dir: str = "snn/checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs
    resume_from: str | None = None  # Path to checkpoint to resume from
    start_epoch: int = 1  # Starting epoch (for resume)

    # Debugging
    max_files: int | None = None  # Limit files for quick testing

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    model: SimpleSNNController,
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

        # Forward pass through sequence
        outputs, _, _ = model.forward_sequence(inputs)

        # MSE loss on voltage predictions
        loss = F.mse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: SimpleSNNController,
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

        outputs, _, _ = model.forward_sequence(inputs)

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


def train(config: TrainConfig) -> SimpleSNNController:
    """Main training function.

    Args:
        config: Training configuration.

    Returns:
        Trained SimpleSNNController model.
    """
    print("=" * 60)
    print("PMSM SNN Controller Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Data: {config.data_dir}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Epochs: {config.epochs}")
    print("=" * 60)

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")

    if config.max_files:
        # Quick test mode - load limited data
        dataset = PMSMDataset(
            data_dir=config.data_dir,
            window_size=config.window_size,
            stride=config.stride,
            max_files=config.max_files,
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
        train_loader, val_loader = create_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            window_size=config.window_size,
            stride=config.stride,
            val_split=config.val_split,
        )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create or load model
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        model = SimpleSNNController.load(config.resume_from, device=config.device)
        print(f"Loaded model with {model.count_parameters():,} parameters")
    else:
        print("\nCreating model...")
        snn_config = SNNConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            beta_hidden=config.beta_hidden,
            beta_output=config.beta_output,
        )

        model = SimpleSNNController(config=snn_config)
        model = model.to(config.device)

        print(f"Parameters: {model.count_parameters():,}")

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
    history_path = checkpoint_dir / "history.json"
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
            model.save(checkpoint_dir / "best_model.pt")

        # Print progress
        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_metrics['loss']:.6f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"LR: {lr:.2e}" + (" *" if is_best else "")
        )
        print(msg, flush=True)

        # Periodic checkpoint
        if epoch % config.save_every == 0:
            model.save(checkpoint_dir / f"epoch_{epoch:03d}.pt")

    # Save final model
    model.save(checkpoint_dir / "final_model.pt")

    # Save training history
    import json

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)

    # Load best model for return
    model = SimpleSNNController.load(
        checkpoint_dir / "best_model.pt", device=config.device
    )

    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train PMSM SNN Controller")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pmsm-pem/export/train_v2",
        help="Directory with training CSV files (train_v2 = clean data)",
    )
    parser.add_argument(
        "--window_size", type=int, default=100, help="Timesteps per training window"
    )
    parser.add_argument("--stride", type=int, default=50, help="Stride between windows")

    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--beta_output", type=float, default=0.995, help="Output layer decay rate"
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
        window_size=args.window_size,
        stride=args.stride,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        beta_output=args.beta_output,
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
