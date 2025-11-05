"""
Configuration file for ECG Image Digitization pipeline.

Contains all hyperparameters, paths, and settings for the project.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
SCRIPT_DIR = ROOT_DIR / "scripts"


@dataclass
class DataConfig:
    """Data-related configuration."""

    # Paths
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    synthetic_data_dir: Path = DATA_DIR / "synthetic"

    # Image specifications
    image_size: Tuple[int, int] = (2200, 1700)  # (height, width)
    target_size: Tuple[int, int] = (1024, 1024)  # Resized for training

    # ECG specifications
    num_leads: int = 12
    lead_names: List[str] = field(default_factory=lambda: [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ])

    # Standard ECG format
    sampling_rate: int = 500  # Hz
    duration: float = 10.0  # seconds
    signal_length: int = 5000  # 500 Hz * 10 seconds

    # Grid specifications
    grid_mm_per_mv: float = 10.0  # mm per mV
    grid_mm_per_sec: float = 25.0  # mm per second
    small_grid_mm: float = 1.0  # small grid size in mm
    large_grid_mm: float = 5.0  # large grid size in mm

    # Layout specifications
    rows: int = 3  # Number of rows of leads
    leads_per_row: int = 4  # Leads per row
    segment_duration: float = 2.5  # seconds per segment
    rhythm_strip_lead: str = 'II'  # Bottom rhythm strip lead


@dataclass
class ModelConfig:
    """Model-related configuration."""

    # Architecture
    architecture: str = "unet"  # unet, unetplusplus, deeplabv3plus
    encoder_name: str = "efficientnet-b4"  # resnet50, efficientnet-b4, etc.
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    out_channels: int = 12  # One channel per lead

    # Training
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = "AdamW"
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Loss function
    loss_function: str = "dice_bce"  # dice, bce, dice_bce, focal
    dice_weight: float = 0.5
    bce_weight: float = 0.5

    # Early stopping
    early_stopping_patience: int = 10

    # Validation
    val_split: float = 0.2
    n_folds: int = 5

    # Model paths
    checkpoint_dir: Path = MODEL_DIR / "checkpoints"
    pretrained_dir: Path = MODEL_DIR / "pretrained"
    best_model_name: str = "best_model.pth"


@dataclass
class PreprocessingConfig:
    """Preprocessing-related configuration."""

    # Grayscale conversion
    convert_to_grayscale: bool = True

    # Noise reduction
    apply_gaussian_blur: bool = True
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0

    apply_median_filter: bool = True
    median_kernel_size: int = 3

    # Contrast enhancement
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Normalization
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class RotationConfig:
    """Rotation correction configuration."""

    # Hough Transform parameters
    use_hough_transform: bool = True
    hough_threshold: int = 100
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 10

    # Rotation parameters
    max_rotation_angle: float = 45.0  # degrees
    rotation_interpolation: str = "bilinear"  # nearest, bilinear, bicubic

    # Grid detection
    detect_grid_lines: bool = True
    horizontal_line_threshold: float = 5.0  # degrees from horizontal
    vertical_line_threshold: float = 5.0  # degrees from vertical


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    # Geometric augmentation
    rotation_limit: float = 5.0  # degrees
    scale_limit: float = 0.1
    shift_limit: float = 0.05

    # Color augmentation
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2

    # Noise augmentation
    gaussian_noise_var: Tuple[float, float] = (10.0, 50.0)
    gaussian_noise_prob: float = 0.3

    # Grid augmentation
    grid_color_variation: bool = True
    grid_color_range: Tuple[int, int] = (200, 255)

    # Texture augmentation
    paper_texture_prob: float = 0.3

    # General
    augmentation_prob: float = 0.8


@dataclass
class VectorizationConfig:
    """Vectorization configuration."""

    # Mask processing
    mask_threshold: float = 0.5
    min_signal_length: int = 100  # minimum number of points

    # Signal extraction
    use_median: bool = False  # Use median instead of mean for y-position
    smoothing_window: int = 5  # Savitzky-Golay filter window
    smoothing_order: int = 2  # Savitzky-Golay filter order

    # Interpolation
    interpolation_method: str = "cubic"  # linear, cubic, quintic


@dataclass
class AlignmentConfig:
    """Signal alignment configuration."""

    # Alignment parameters
    max_time_shift: float = 0.5  # seconds
    max_voltage_shift: float = 1.0  # mV

    # Cross-correlation
    correlation_mode: str = "full"  # full, valid, same

    # Optimization
    use_optimization: bool = True
    optimization_method: str = "grid_search"  # grid_search, gradient_descent


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # SNR calculation
    calculate_snr: bool = True

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        'snr', 'mse', 'mae', 'correlation'
    ])

    # Per-lead evaluation
    evaluate_per_lead: bool = True

    # Visualization
    save_plots: bool = True
    plot_dir: Path = ROOT_DIR / "outputs" / "plots"


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Model ensemble
    use_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: [
        "fold_0_best.pth",
        "fold_1_best.pth",
        "fold_2_best.pth",
    ])
    ensemble_method: str = "average"  # average, weighted_average, voting

    # Test-time augmentation
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: [
        "horizontal_flip",
        "vertical_flip",
    ])

    # Post-processing
    apply_post_processing: bool = True
    post_processing_steps: List[str] = field(default_factory=lambda: [
        "smoothing",
        "outlier_removal",
    ])

    # Output
    output_format: str = "csv"  # csv, json, hdf5
    save_visualization: bool = True


# Global configuration
@dataclass
class Config:
    """Master configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    vectorization: VectorizationConfig = field(default_factory=VectorizationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # General settings
    seed: int = 42
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    verbose: bool = True

    # Logging
    log_dir: Path = ROOT_DIR / "logs"
    tensorboard_dir: Path = ROOT_DIR / "runs"

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.data.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.data.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        self.model.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        if self.evaluation.save_plots:
            self.evaluation.plot_dir.mkdir(parents=True, exist_ok=True)


# Create default configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs):
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


if __name__ == "__main__":
    # Print configuration
    cfg = get_config()
    print("PhysioNet ECG Digitization Configuration")
    print("=" * 50)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"\nData Config:")
    print(f"  Image Size: {cfg.data.image_size}")
    print(f"  Num Leads: {cfg.data.num_leads}")
    print(f"  Sampling Rate: {cfg.data.sampling_rate} Hz")
    print(f"\nModel Config:")
    print(f"  Architecture: {cfg.model.architecture}")
    print(f"  Encoder: {cfg.model.encoder_name}")
    print(f"  Batch Size: {cfg.model.batch_size}")
    print(f"  Learning Rate: {cfg.model.learning_rate}")
