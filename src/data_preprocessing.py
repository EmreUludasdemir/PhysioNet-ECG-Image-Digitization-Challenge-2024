"""
Data preprocessing module for ECG image digitization.

This module handles:
- Image loading and validation
- Grayscale conversion
- Noise reduction (Gaussian blur, median filter)
- Contrast enhancement (CLAHE)
- Image normalization and resizing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image

from .config import get_config


class ECGImagePreprocessor:
    """Preprocessor for ECG images."""

    def __init__(self, config=None):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.data_config = self.config.data
        self.prep_config = self.config.preprocessing

    def load_image(
        self,
        image_path: Union[str, Path],
        color_mode: str = 'rgb'
    ) -> np.ndarray:
        """
        Load an image from file.

        Args:
            image_path: Path to the image file
            color_mode: Color mode ('rgb', 'bgr', 'grayscale')

        Returns:
            Loaded image as numpy array

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image using OpenCV
        if color_mode == 'grayscale':
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_path))
            if color_mode == 'rgb' and image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: Input image (RGB or BGR)

        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            # Already grayscale
            return image

        # Convert to grayscale
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            # RGBA image
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")

    def apply_gaussian_blur(
        self,
        image: np.ndarray,
        kernel_size: Optional[Tuple[int, int]] = None,
        sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.

        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (width, height)
            sigma: Standard deviation of Gaussian kernel

        Returns:
            Blurred image
        """
        kernel_size = kernel_size or self.prep_config.gaussian_kernel_size
        sigma = sigma or self.prep_config.gaussian_sigma

        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_median_filter(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply median filter for noise reduction.

        Args:
            image: Input image
            kernel_size: Size of median filter kernel

        Returns:
            Filtered image
        """
        kernel_size = kernel_size or self.prep_config.median_kernel_size

        return cv2.medianBlur(image, kernel_size)

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        tile_grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            Contrast-enhanced image
        """
        clip_limit = clip_limit or self.prep_config.clahe_clip_limit
        tile_grid_size = tile_grid_size or self.prep_config.clahe_tile_grid_size

        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = self.convert_to_grayscale(image)

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        return clahe.apply(image)

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        interpolation: str = 'bilinear'
    ) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            target_size: Target size (height, width)
            interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')

        Returns:
            Resized image
        """
        target_size = target_size or self.data_config.target_size

        # Map interpolation method to OpenCV constant
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }

        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

        # Resize (OpenCV uses (width, height) order)
        resized = cv2.resize(
            image,
            (target_size[1], target_size[0]),
            interpolation=interp
        )

        return resized

    def normalize_image(
        self,
        image: np.ndarray,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> np.ndarray:
        """
        Normalize image using mean and std.

        Args:
            image: Input image (values in [0, 255])
            mean: Mean values for normalization
            std: Standard deviation values for normalization

        Returns:
            Normalized image (values typically in [-1, 1])
        """
        mean = mean or self.prep_config.normalize_mean
        std = std or self.prep_config.normalize_std

        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize
        if len(image.shape) == 2:
            # Grayscale image - use first mean/std value
            image = (image - mean[0]) / std[0]
        else:
            # Multi-channel image
            for i in range(image.shape[2]):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

        return image

    def denormalize_image(
        self,
        image: np.ndarray,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range.

        Args:
            image: Normalized image
            mean: Mean values used for normalization
            std: Standard deviation values used for normalization

        Returns:
            Denormalized image (values in [0, 255])
        """
        mean = mean or self.prep_config.normalize_mean
        std = std or self.prep_config.normalize_std

        # Denormalize
        if len(image.shape) == 2:
            image = image * std[0] + mean[0]
        else:
            for i in range(image.shape[2]):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]

        # Scale back to [0, 255] and clip
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        return image

    def preprocess(
        self,
        image: Union[str, Path, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        apply_normalization: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for an ECG image.

        Args:
            image: Input image (path or numpy array)
            target_size: Target size for resizing
            apply_normalization: Whether to apply normalization

        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)

        # Convert to grayscale if configured
        if self.prep_config.convert_to_grayscale:
            image = self.convert_to_grayscale(image)

        # Apply Gaussian blur if configured
        if self.prep_config.apply_gaussian_blur:
            image = self.apply_gaussian_blur(image)

        # Apply median filter if configured
        if self.prep_config.apply_median_filter:
            image = self.apply_median_filter(image)

        # Apply CLAHE if configured
        if self.prep_config.apply_clahe:
            image = self.apply_clahe(image)

        # Resize image
        image = self.resize_image(image, target_size)

        # Normalize if requested
        if apply_normalization:
            image = self.normalize_image(image)

        return image

    def preprocess_batch(
        self,
        images: list,
        target_size: Optional[Tuple[int, int]] = None,
        apply_normalization: bool = True
    ) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images: List of images (paths or numpy arrays)
            target_size: Target size for resizing
            apply_normalization: Whether to apply normalization

        Returns:
            Batch of preprocessed images as numpy array
        """
        preprocessed = []

        for image in images:
            preprocessed_image = self.preprocess(
                image,
                target_size=target_size,
                apply_normalization=apply_normalization
            )
            preprocessed.append(preprocessed_image)

        return np.array(preprocessed)


def preprocess_image(
    image_path: Union[str, Path],
    config=None
) -> np.ndarray:
    """
    Convenience function to preprocess a single image.

    Args:
        image_path: Path to the image
        config: Configuration object

    Returns:
        Preprocessed image
    """
    preprocessor = ECGImagePreprocessor(config)
    return preprocessor.preprocess(image_path)


def preprocess_images(
    image_paths: list,
    config=None
) -> np.ndarray:
    """
    Convenience function to preprocess multiple images.

    Args:
        image_paths: List of image paths
        config: Configuration object

    Returns:
        Batch of preprocessed images
    """
    preprocessor = ECGImagePreprocessor(config)
    return preprocessor.preprocess_batch(image_paths)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_preprocessing.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    preprocessor = ECGImagePreprocessor()

    # Load and preprocess
    print(f"Loading image: {image_path}")
    original = preprocessor.load_image(image_path)
    print(f"Original shape: {original.shape}")

    # Apply preprocessing steps
    preprocessed = preprocessor.preprocess(image_path, apply_normalization=False)
    print(f"Preprocessed shape: {preprocessed.shape}")

    # Save preprocessed image for visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_preprocessed.png"
    cv2.imwrite(str(output_path), preprocessed)
    print(f"Saved preprocessed image to: {output_path}")
