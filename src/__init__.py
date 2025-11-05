"""
PhysioNet ECG Image Digitization Challenge 2024
================================================

A comprehensive solution for digitizing ECG images into time-series signals.

Main modules:
- data_preprocessing: Image preprocessing and normalization
- rotation: Hough transform-based rotation correction
- segmentation_model: U-Net model for lead segmentation
- vectorization: Signal extraction from segmentation masks
- signal_alignment: Cross-correlation based signal alignment
- evaluation: SNR calculation and metrics
- inference: End-to-end inference pipeline
"""

__version__ = "0.1.0"
__author__ = "PhysioNet Challenge Team"

from . import config
from . import data_preprocessing
from . import evaluation

__all__ = [
    "config",
    "data_preprocessing",
    "evaluation",
]
