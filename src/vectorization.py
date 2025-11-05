"""
Vectorization module for extracting ECG signals from segmentation masks.

This module converts binary segmentation masks into time-series ECG signals
by extracting signal coordinates and scaling them appropriately.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional, Dict

from .config import get_config


class ECGVectorizer:
    """Vectorizer for extracting ECG signals from segmentation masks."""

    def __init__(self, config=None):
        """
        Initialize the vectorizer.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.data_config = self.config.data
        self.vec_config = self.config.vectorization

    def extract_signal_from_mask(
        self,
        mask: np.ndarray,
        threshold: Optional[float] = None,
        use_median: Optional[bool] = None
    ) -> np.ndarray:
        """
        Extract 1D signal from 2D segmentation mask.

        Args:
            mask: Binary segmentation mask (H, W)
            threshold: Threshold for binarization
            use_median: Use median instead of mean for y-position

        Returns:
            1D signal (num_points,) or None if extraction fails
        """
        threshold = threshold or self.vec_config.mask_threshold
        use_median = use_median if use_median is not None else self.vec_config.use_median

        # Binarize mask
        binary_mask = (mask > threshold).astype(np.uint8)

        # Get dimensions
        height, width = binary_mask.shape

        # Extract signal for each column
        signal = np.zeros(width)
        valid_cols = []

        for col in range(width):
            # Get non-zero pixels in this column
            rows = np.where(binary_mask[:, col] > 0)[0]

            if len(rows) > 0:
                # Use median or mean of y-positions
                if use_median:
                    y_pos = np.median(rows)
                else:
                    y_pos = np.mean(rows)

                signal[col] = y_pos
                valid_cols.append(col)

        if len(valid_cols) < self.vec_config.min_signal_length:
            return None

        # Convert y-position to signal value (invert because image coordinates)
        # Higher y = lower value in image coordinates
        signal = height - signal

        return signal

    def smooth_signal(
        self,
        signal: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None
    ) -> np.ndarray:
        """
        Smooth signal using Savitzky-Golay filter.

        Args:
            signal: Input signal
            window_length: Window length for filter
            polyorder: Polynomial order for filter

        Returns:
            Smoothed signal
        """
        window_length = window_length or self.vec_config.smoothing_window
        polyorder = polyorder or self.vec_config.smoothing_order

        # Ensure window_length is odd and > polyorder
        if window_length % 2 == 0:
            window_length += 1

        if window_length <= polyorder:
            window_length = polyorder + 2

        if len(signal) < window_length:
            return signal

        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(signal, window_length, polyorder)

        return smoothed

    def interpolate_signal(
        self,
        signal: np.ndarray,
        target_length: int,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Interpolate signal to target length.

        Args:
            signal: Input signal
            target_length: Target length
            method: Interpolation method ('linear', 'cubic', 'quintic')

        Returns:
            Interpolated signal
        """
        method = method or self.vec_config.interpolation_method

        # Create interpolation function
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)

        # Use scipy interpolation
        f = interp1d(x_old, signal, kind=method, fill_value='extrapolate')
        interpolated = f(x_new)

        return interpolated

    def pixel_to_voltage(
        self,
        pixel_values: np.ndarray,
        image_height: int,
        voltage_range: Tuple[float, float] = (-2.0, 2.0)
    ) -> np.ndarray:
        """
        Convert pixel values to voltage (mV).

        Args:
            pixel_values: Pixel values (in image coordinates)
            image_height: Image height in pixels
            voltage_range: Voltage range (min_mV, max_mV)

        Returns:
            Voltage values in mV
        """
        # Normalize pixel values to [0, 1]
        normalized = pixel_values / image_height

        # Scale to voltage range
        min_voltage, max_voltage = voltage_range
        voltages = min_voltage + normalized * (max_voltage - min_voltage)

        return voltages

    def extract_lead_signals(
        self,
        masks: np.ndarray,
        lead_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract signals for all leads from multi-channel mask.

        Args:
            masks: Multi-channel segmentation mask (num_leads, H, W) or (H, W, num_leads)
            lead_names: Names of leads

        Returns:
            Dictionary mapping lead names to signals
        """
        lead_names = lead_names or self.data_config.lead_names

        # Ensure masks has shape (num_leads, H, W)
        if masks.ndim == 3 and masks.shape[-1] == len(lead_names):
            # Convert (H, W, C) to (C, H, W)
            masks = np.transpose(masks, (2, 0, 1))

        num_leads = masks.shape[0]
        if num_leads != len(lead_names):
            raise ValueError(
                f"Number of channels ({num_leads}) doesn't match "
                f"number of lead names ({len(lead_names)})"
            )

        # Extract signal for each lead
        signals = {}

        for i, lead_name in enumerate(lead_names):
            signal = self.extract_signal_from_mask(masks[i])

            if signal is not None:
                # Smooth signal
                signal = self.smooth_signal(signal)

                # Interpolate to target length
                target_length = self.data_config.signal_length
                signal = self.interpolate_signal(signal, target_length)

                # Convert to voltage
                signal = self.pixel_to_voltage(signal, masks.shape[1])

                signals[lead_name] = signal
            else:
                # If extraction failed, use zeros
                signals[lead_name] = np.zeros(self.data_config.signal_length)

        return signals

    def signals_to_array(
        self,
        signals: Dict[str, np.ndarray],
        lead_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert signal dictionary to array.

        Args:
            signals: Dictionary mapping lead names to signals
            lead_names: Ordered list of lead names

        Returns:
            Signal array (num_leads, signal_length)
        """
        lead_names = lead_names or self.data_config.lead_names

        signal_array = np.zeros((len(lead_names), self.data_config.signal_length))

        for i, lead_name in enumerate(lead_names):
            if lead_name in signals:
                signal_array[i] = signals[lead_name]

        return signal_array

    def vectorize(
        self,
        masks: np.ndarray,
        lead_names: Optional[List[str]] = None,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete vectorization pipeline.

        Args:
            masks: Multi-channel segmentation mask
            lead_names: Names of leads
            return_dict: If True, return dictionary; else return array

        Returns:
            Extracted signals as array or dictionary
        """
        # Extract signals
        signals = self.extract_lead_signals(masks, lead_names)

        if return_dict:
            return signals
        else:
            return self.signals_to_array(signals, lead_names)


def vectorize_masks(
    masks: np.ndarray,
    config=None,
    return_dict: bool = False
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function to vectorize segmentation masks.

    Args:
        masks: Multi-channel segmentation mask
        config: Configuration object
        return_dict: If True, return dictionary; else return array

    Returns:
        Extracted signals
    """
    vectorizer = ECGVectorizer(config)
    return vectorizer.vectorize(masks, return_dict=return_dict)


def extract_single_lead(
    mask: np.ndarray,
    config=None
) -> np.ndarray:
    """
    Convenience function to extract signal from single lead mask.

    Args:
        mask: Single-channel segmentation mask (H, W)
        config: Configuration object

    Returns:
        Extracted signal
    """
    vectorizer = ECGVectorizer(config)

    # Extract raw signal
    signal = vectorizer.extract_signal_from_mask(mask)

    if signal is None:
        return np.zeros(vectorizer.data_config.signal_length)

    # Smooth
    signal = vectorizer.smooth_signal(signal)

    # Interpolate to target length
    signal = vectorizer.interpolate_signal(
        signal,
        vectorizer.data_config.signal_length
    )

    # Convert to voltage
    signal = vectorizer.pixel_to_voltage(signal, mask.shape[0])

    return signal


if __name__ == "__main__":
    # Test vectorization
    print("Testing ECG Vectorization...")

    # Create dummy mask
    height, width = 1024, 1024
    num_leads = 12

    # Create synthetic mask with diagonal line
    dummy_mask = np.zeros((num_leads, height, width))
    for i in range(num_leads):
        for x in range(width):
            y = int(height / 2 + 100 * np.sin(2 * np.pi * x / width * (i + 1)))
            if 0 <= y < height:
                dummy_mask[i, max(0, y - 2):min(height, y + 3), x] = 1.0

    # Vectorize
    vectorizer = ECGVectorizer()
    signals = vectorizer.vectorize(dummy_mask, return_dict=True)

    print(f"Extracted {len(signals)} lead signals")
    for lead_name, signal in signals.items():
        print(f"  {lead_name}: shape={signal.shape}, "
              f"min={signal.min():.3f}, max={signal.max():.3f}, "
              f"mean={signal.mean():.3f}")

    # Convert to array
    signal_array = vectorizer.signals_to_array(signals)
    print(f"\nSignal array shape: {signal_array.shape}")
