"""
Segmentation model for ECG lead detection.

This module provides U-Net based segmentation models with various encoders
for detecting individual ECG leads in images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import segmentation_models_pytorch as smp

from .config import get_config


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            pred: Predicted masks (B, C, H, W)
            target: Target masks (B, C, H, W)

        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - Dice as loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.

        Args:
            pred: Predicted logits (B, C, H, W)
            target: Target masks (B, C, H, W)

        Returns:
            Focal loss value
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Calculate binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Calculate focal term
        p_t = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_term = (1 - p_t) ** self.gamma

        # Calculate final loss
        loss = self.alpha * focal_term * bce

        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function (Dice + BCE or Dice + Focal)."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss.

        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE/Focal loss
            use_focal: Use Focal loss instead of BCE
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.use_focal = use_focal

        self.dice_loss = DiceLoss()

        if use_focal:
            self.bce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            pred: Predicted masks (B, C, H, W)
            target: Target masks (B, C, H, W)

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


class ECGSegmentationModel(nn.Module):
    """ECG segmentation model wrapper."""

    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        out_channels: int = 12,
        **kwargs
    ):
        """
        Initialize segmentation model.

        Args:
            architecture: Model architecture ('unet', 'unetplusplus', 'deeplabv3plus', etc.)
            encoder_name: Encoder backbone ('resnet50', 'efficientnet-b4', etc.)
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            out_channels: Number of output channels (number of leads)
            **kwargs: Additional arguments for the model
        """
        super().__init__()

        # Create model based on architecture
        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                **kwargs
            )
        elif architecture == "unetplusplus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                **kwargs
            )
        elif architecture == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                **kwargs
            )
        elif architecture == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.architecture = architecture
        self.encoder_name = encoder_name
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Predicted masks (B, out_channels, H, W)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary masks.

        Args:
            x: Input images (B, C, H, W)
            threshold: Threshold for binarization

        Returns:
            Binary masks (B, out_channels, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()

        return masks

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config=None) -> ECGSegmentationModel:
    """
    Create segmentation model from configuration.

    Args:
        config: Configuration object. If None, uses default config.

    Returns:
        ECGSegmentationModel instance
    """
    cfg = config or get_config()
    model_config = cfg.model

    model = ECGSegmentationModel(
        architecture=model_config.architecture,
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
    )

    return model


def create_loss_function(config=None) -> nn.Module:
    """
    Create loss function from configuration.

    Args:
        config: Configuration object. If None, uses default config.

    Returns:
        Loss function module
    """
    cfg = config or get_config()
    model_config = cfg.model

    loss_type = model_config.loss_function

    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "dice_bce":
        return CombinedLoss(
            dice_weight=model_config.dice_weight,
            bce_weight=model_config.bce_weight,
            use_focal=False
        )
    elif loss_type == "dice_focal":
        return CombinedLoss(
            dice_weight=model_config.dice_weight,
            bce_weight=model_config.bce_weight,
            use_focal=True
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


class ModelTrainer:
    """Trainer class for ECG segmentation model."""

    def __init__(
        self,
        model: ECGSegmentationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Segmentation model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader) -> float:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint.get('epoch', 0)


if __name__ == "__main__":
    # Test model creation
    print("Creating ECG Segmentation Model...")

    model = create_model()
    print(f"Architecture: {model.architecture}")
    print(f"Encoder: {model.encoder_name}")
    print(f"Number of parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 2
    in_channels = 3
    height, width = 1024, 1024

    dummy_input = torch.randn(batch_size, in_channels, height, width)
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test loss functions
    print("\nTesting loss functions...")
    dummy_target = torch.randint(0, 2, (batch_size, 12, height, width)).float()

    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(output, dummy_target).item():.4f}")

    combined_loss = create_loss_function()
    print(f"Combined Loss: {combined_loss(output, dummy_target).item():.4f}")
