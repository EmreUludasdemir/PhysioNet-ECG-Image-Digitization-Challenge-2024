"""
Training script for ECG segmentation model.

This script handles model training with cross-validation,
early stopping, and checkpoint saving.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.segmentation_model import (
    ECGSegmentationModel,
    create_model,
    create_loss_function,
    ModelTrainer
)
from src.augmentation import get_training_augmentation, get_validation_augmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ECG Segmentation Model")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Path to save checkpoints"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="efficientnet-b4",
        help="Encoder name (e.g., resnet50, efficientnet-b4)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        help="Model architecture (unet, unetplusplus, deeplabv3plus)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation"
    )

    return parser.parse_args()


def create_data_loaders(data_dir, batch_size, num_workers, fold=0):
    """
    Create training and validation data loaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of workers
        fold: Fold number

    Returns:
        train_loader, val_loader
    """
    # TODO: Implement actual data loading
    # This is a placeholder that should be replaced with actual dataset implementation

    print(f"Loading data from {data_dir} for fold {fold}")
    print("Note: Using placeholder data loaders. Implement actual data loading.")

    # Placeholder - replace with actual implementation
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return dummy data
            image = torch.randn(3, 1024, 1024)
            mask = torch.randint(0, 2, (12, 1024, 1024)).float()
            return {'image': image, 'mask': mask}

    train_dataset = DummyDataset(size=100)
    val_dataset = DummyDataset(size=20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_model(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    # Setup
    config = get_config()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ECG Segmentation Model Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"Architecture: {args.architecture}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Fold: {args.fold}")
    print("=" * 60)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.fold
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = ECGSegmentationModel(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        out_channels=12
    )
    print(f"Model parameters: {model.get_num_params():,}")

    # Create loss function
    criterion = create_loss_function(config)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.model.weight_decay
    )

    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config.model.scheduler_patience,
        factor=config.model.scheduler_factor,
        verbose=True
    )

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}")

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save checkpoint
        checkpoint_path = output_dir / f"fold_{args.fold}_epoch_{epoch+1}.pth"
        trainer.save_checkpoint(checkpoint_path, epoch + 1, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / f"fold_{args.fold}_best.pth"
            trainer.save_checkpoint(best_path, epoch + 1, val_loss)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.model.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {output_dir / f'fold_{args.fold}_best.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
