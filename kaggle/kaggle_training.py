"""
PhysioNet ECG Image Digitization - Kaggle Training Script
=========================================================

Complete end-to-end training script for ECG image-to-signal regression.
Trains a model to convert scanned ECG images to digital signals.

Author: PhysioNet Challenge Team
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import timm
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration."""

    # Paths
    INPUT_DIR = '/kaggle/input/physionet-ecg-image-digitization'
    OUTPUT_DIR = '/kaggle/working'

    # Image settings
    IMG_SIZE = (512, 512)  # Resize to this size
    NUM_CHANNELS = 3

    # Signal settings
    NUM_LEADS = 12
    SIGNAL_LENGTH = 5000
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Training settings
    BATCH_SIZE = 8  # Adjust based on GPU memory
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Model settings
    ENCODER_NAME = 'efficientnet_b2'  # timm model
    PRETRAINED = True

    # Training strategy
    TRAIN_SPLIT = 0.85
    RANDOM_SEED = 42
    NUM_WORKERS = 2

    # Early stopping
    PATIENCE = 10

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()


# ============================================================================
# DATASET
# ============================================================================

class ECGImageDataset(Dataset):
    """
    Dataset for ECG image-to-signal regression.

    Each sample contains:
    - image: Scanned ECG image
    - signal: Ground truth digital signal (12 leads × 5000 timesteps)
    - metadata: fs, sig_len
    """

    def __init__(self, records, train_df, transform=None):
        """
        Args:
            records: List of record IDs
            train_df: DataFrame with metadata (id, fs, sig_len)
            transform: Optional image transforms
        """
        self.records = records
        self.train_df = train_df
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_id = self.records[idx]
        record_dir = Path(config.INPUT_DIR) / 'train' / str(record_id)

        # Load first image (use -0001.png if exists, else first available)
        images = sorted([f for f in os.listdir(record_dir) if f.endswith('.png')])
        if not images:
            raise ValueError(f"No images found for record {record_id}")

        # Prefer -0001.png
        target_img = f"{record_id}-0001.png"
        if target_img in images:
            img_path = record_dir / target_img
        else:
            img_path = record_dir / images[0]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, config.IMG_SIZE)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert to tensor: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Load ground truth signal
        csv_path = record_dir / f"{record_id}.csv"
        signal_df = pd.read_csv(csv_path)

        # Extract signal values (12 leads × signal_length)
        signal = signal_df[config.LEAD_NAMES].values.T  # (12, signal_length)

        # Handle NaN values (fill with 0)
        signal = np.nan_to_num(signal, nan=0.0)

        # Resize to fixed length (5000)
        if signal.shape[1] != config.SIGNAL_LENGTH:
            # Resample using interpolation
            signal_resampled = np.zeros((config.NUM_LEADS, config.SIGNAL_LENGTH))
            for i in range(config.NUM_LEADS):
                x_old = np.linspace(0, 1, signal.shape[1])
                x_new = np.linspace(0, 1, config.SIGNAL_LENGTH)
                signal_resampled[i] = np.interp(x_new, x_old, signal[i])
            signal = signal_resampled

        signal = torch.from_numpy(signal).float()  # (12, 5000)

        # Get metadata
        meta = self.train_df[self.train_df['id'] == record_id].iloc[0]
        fs = meta['fs']
        sig_len = meta['sig_len']

        return {
            'image': image,
            'signal': signal,
            'record_id': record_id,
            'fs': fs,
            'sig_len': sig_len
        }


# ============================================================================
# MODEL
# ============================================================================

class ECGRegressionModel(nn.Module):
    """
    Image-to-Signal Regression Model.

    Architecture:
    1. Encoder: Pre-trained CNN (EfficientNet)
    2. Global pooling
    3. FC layers to predict 12 × 5000 signal values
    """

    def __init__(self, encoder_name='efficientnet_b2', pretrained=True):
        super().__init__()

        # Encoder: Pre-trained CNN
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling (we'll add custom)
        )

        # Get encoder output channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.encoder(dummy)
            encoder_channels = features.shape[1]

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Decoder: FC layers
        self.decoder = nn.Sequential(
            nn.Linear(encoder_channels, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(4096, config.NUM_LEADS * config.SIGNAL_LENGTH)
        )

        print(f"Model created with encoder: {encoder_name}")
        print(f"Encoder output channels: {encoder_channels}")
        print(f"Output shape: {config.NUM_LEADS} × {config.SIGNAL_LENGTH}")

    def forward(self, x):
        # x: (B, 3, H, W)

        # Encode
        features = self.encoder(x)  # (B, C, H', W')

        # Global pool
        features = self.global_pool(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)

        # Decode
        output = self.decoder(features)  # (B, 12*5000)

        # Reshape to signal format
        output = output.view(-1, config.NUM_LEADS, config.SIGNAL_LENGTH)  # (B, 12, 5000)

        return output


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class SNRLoss(nn.Module):
    """
    Signal-to-Noise Ratio (SNR) based loss.
    Maximizing SNR is equivalent to minimizing this loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: (B, 12, 5000)

        # Signal power
        signal_power = torch.sum(target ** 2, dim=-1)  # (B, 12)

        # Noise (error)
        noise = pred - target
        noise_power = torch.sum(noise ** 2, dim=-1)  # (B, 12)

        # Avoid division by zero
        noise_power = noise_power + 1e-8

        # SNR = 10 * log10(signal_power / noise_power)
        # We want to maximize SNR, so minimize -SNR
        snr = 10 * torch.log10(signal_power / noise_power)

        # Average over leads and batch
        loss = -snr.mean()

        return loss


class CombinedLoss(nn.Module):
    """Combination of MSE and SNR loss."""

    def __init__(self, mse_weight=0.5, snr_weight=0.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.snr_loss = SNRLoss()
        self.mse_weight = mse_weight
        self.snr_weight = snr_weight

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        snr = self.snr_loss(pred, target)
        return self.mse_weight * mse + self.snr_weight * snr


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        signals = batch['signal'].to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, signals)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_snr = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            signals = batch['signal'].to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, signals)

            # Calculate SNR
            signal_power = torch.sum(signals ** 2, dim=-1)
            noise = outputs - signals
            noise_power = torch.sum(noise ** 2, dim=-1)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

            total_loss += loss.item()
            total_snr += snr.mean().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'snr': f'{snr.mean().item():.2f} dB'})

    avg_loss = total_loss / len(dataloader)
    avg_snr = total_snr / len(dataloader)

    return avg_loss, avg_snr


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function."""

    print("=" * 80)
    print("PhysioNet ECG Image Digitization - Training")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 80)

    # Load metadata
    print("\n📊 Loading data...")
    train_csv = Path(config.INPUT_DIR) / 'train.csv'
    train_df = pd.read_csv(train_csv)
    print(f"Total records: {len(train_df)}")

    # Split train/validation
    train_ids, val_ids = train_test_split(
        train_df['id'].values,
        test_size=1-config.TRAIN_SPLIT,
        random_state=config.RANDOM_SEED
    )

    print(f"Train records: {len(train_ids)}")
    print(f"Validation records: {len(val_ids)}")

    # Create datasets
    train_dataset = ECGImageDataset(train_ids, train_df)
    val_dataset = ECGImageDataset(val_ids, train_df)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Create model
    print("\n🔧 Creating model...")
    model = ECGRegressionModel(
        encoder_name=config.ENCODER_NAME,
        pretrained=config.PRETRAINED
    )
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(mse_weight=0.5, snr_weight=0.5)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    best_snr = float('-inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"{'='*80}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_snr = validate(model, val_loader, criterion, config.DEVICE)
        print(f"Val Loss: {val_loss:.4f} | Val SNR: {val_snr:.2f} dB")

        # Scheduler step
        scheduler.step(val_loss)

        # Save best model (based on SNR)
        if val_snr > best_snr:
            best_snr = val_snr
            best_val_loss = val_loss

            save_path = Path(config.OUTPUT_DIR) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_snr': val_snr,
                'config': config.__dict__
            }, save_path)

            print(f"✅ New best model saved! SNR: {val_snr:.2f} dB")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏸️  Early stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"Best Val SNR: {best_snr:.2f} dB")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: {config.OUTPUT_DIR}/best_model.pth")
    print("=" * 80)


if __name__ == "__main__":
    main()
