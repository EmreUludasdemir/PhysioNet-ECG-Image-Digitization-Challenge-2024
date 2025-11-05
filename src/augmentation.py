"""
Data augmentation module for ECG images.

This module provides augmentation techniques specifically designed for
ECG images to improve model robustness.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Optional, Tuple

from .config import get_config


class ECGAugmentation:
    """Augmentation pipeline for ECG images."""

    def __init__(self, config=None, is_training: bool = True):
        """
        Initialize augmentation pipeline.

        Args:
            config: Configuration object. If None, uses default config.
            is_training: Whether this is for training (applies augmentation) or validation
        """
        self.config = config or get_config()
        self.aug_config = self.config.augmentation
        self.is_training = is_training

        # Create augmentation pipeline
        self.transform = self._create_transform()

    def _create_transform(self) -> A.Compose:
        """
        Create albumentations transform pipeline.

        Returns:
            Albumentations Compose object
        """
        if not self.is_training:
            # Validation/test transforms (no augmentation)
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        # Training transforms with augmentation
        transforms = []

        # Geometric augmentations
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=self.aug_config.shift_limit,
                scale_limit=self.aug_config.scale_limit,
                rotate_limit=self.aug_config.rotation_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                mask_value=0,
                p=self.aug_config.augmentation_prob
            )
        )

        # Color augmentations
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_config.brightness_limit,
                contrast_limit=self.aug_config.contrast_limit,
                p=self.aug_config.augmentation_prob
            )
        )

        # Noise augmentation
        if self.aug_config.gaussian_noise_prob > 0:
            transforms.append(
                A.GaussNoise(
                    var_limit=self.aug_config.gaussian_noise_var,
                    p=self.aug_config.gaussian_noise_prob
                )
            )

        # Blur augmentation (simulates image quality variations)
        transforms.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2)
        )

        # Paper texture simulation (optional)
        if self.aug_config.paper_texture_prob > 0:
            transforms.append(
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.3),
                    p=self.aug_config.paper_texture_prob
                )
            )

        # Normalize and convert to tensor
        transforms.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentation to image and mask.

        Args:
            image: Input image (H, W, C) or (H, W)
            mask: Optional mask (H, W, C) or (H, W)

        Returns:
            Dictionary with 'image' and optionally 'mask' keys
        """
        # Ensure image has 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Apply transforms
        if mask is not None:
            # With mask
            transformed = self.transform(image=image, mask=mask)
            return {
                'image': transformed['image'],
                'mask': transformed['mask']
            }
        else:
            # Without mask
            transformed = self.transform(image=image)
            return {'image': transformed['image']}


def get_training_augmentation(config=None) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        config: Configuration object

    Returns:
        Albumentations Compose object
    """
    cfg = config or get_config()
    aug_cfg = cfg.augmentation

    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=aug_cfg.shift_limit,
            scale_limit=aug_cfg.scale_limit,
            rotate_limit=aug_cfg.rotation_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            mask_value=0,
            p=aug_cfg.augmentation_prob
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.brightness_limit,
            contrast_limit=aug_cfg.contrast_limit,
            p=aug_cfg.augmentation_prob
        ),
        A.GaussNoise(
            var_limit=aug_cfg.gaussian_noise_var,
            p=aug_cfg.gaussian_noise_prob
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.3),
            p=aug_cfg.paper_texture_prob
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_validation_augmentation(config=None) -> A.Compose:
    """
    Get validation augmentation pipeline (no augmentation, just normalization).

    Args:
        config: Configuration object

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_test_time_augmentation() -> List[A.Compose]:
    """
    Get test-time augmentation transforms.

    Returns:
        List of transform pipelines for TTA
    """
    tta_transforms = []

    # Original
    tta_transforms.append(
        A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    )

    # Horizontal flip
    tta_transforms.append(
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    )

    # Vertical flip
    tta_transforms.append(
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    )

    # Both flips
    tta_transforms.append(
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    )

    return tta_transforms


class ECGDataset:
    """
    Dataset class for ECG images.

    This is a template/example dataset class.
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths: Optional[list] = None,
        transform: Optional[A.Compose] = None,
        config=None
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            mask_paths: Optional list of mask file paths
            transform: Albumentations transform
            config: Configuration object
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.config = config or get_config()

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Dictionary with image and optionally mask
        """
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask if available
        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        else:
            mask = None

        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                return {
                    'image': transformed['image'],
                    'mask': transformed['mask']
                }
            else:
                transformed = self.transform(image=image)
                return {'image': transformed['image']}
        else:
            sample = {'image': image}
            if mask is not None:
                sample['mask'] = mask
            return sample


if __name__ == "__main__":
    # Test augmentation
    print("Testing ECG Augmentation...")

    # Create dummy image
    image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    # Add some dummy ECG-like pattern
    for i in range(0, 1024, 50):
        cv2.line(image, (0, i), (1024, i), (200, 200, 200), 1)
        cv2.line(image, (i, 0), (i, 1024), (200, 200, 200), 1)

    # Create dummy mask
    mask = np.zeros((1024, 1024, 12), dtype=np.uint8)
    for i in range(12):
        y_pos = 100 + i * 70
        for x in range(1024):
            y = int(y_pos + 20 * np.sin(2 * np.pi * x / 200))
            if 0 <= y < 1024:
                mask[max(0, y-2):min(1024, y+3), x, i] = 1

    # Test training augmentation
    aug = ECGAugmentation(is_training=True)
    augmented = aug(image, mask)

    print(f"Original image shape: {image.shape}")
    print(f"Original mask shape: {mask.shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print(f"Augmented mask shape: {augmented['mask'].shape}")

    # Test validation augmentation
    val_aug = ECGAugmentation(is_training=False)
    val_augmented = val_aug(image, mask)

    print(f"\nValidation image shape: {val_augmented['image'].shape}")
    print(f"Validation mask shape: {val_augmented['mask'].shape}")
