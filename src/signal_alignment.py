"""
Signal alignment module for ECG signals.

This module provides cross-correlation based alignment to optimize
the match between predicted and ground truth signals.
"""

import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize
from typing import Tuple, Optional

from .config import get_config


class SignalAligner:
    """Signal aligner using cross-correlation."""

    def __init__(self, config=None):
        """
        Initialize the signal aligner.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.align_config = self.config.alignment
        self.data_config = self.config.data

    def calculate_time_shift_samples(
        self,
        max_time_shift: Optional[float] = None
    ) -> int:
        """
        Calculate maximum time shift in samples.

        Args:
            max_time_shift: Maximum time shift in seconds

        Returns:
            Maximum time shift in samples
        """
        max_time_shift = max_time_shift or self.align_config.max_time_shift
        sampling_rate = self.data_config.sampling_rate

        max_shift_samples = int(max_time_shift * sampling_rate)

        return max_shift_samples

    def cross_correlate(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        mode: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate cross-correlation between two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            mode: Correlation mode ('full', 'valid', 'same')

        Returns:
            Cross-correlation values
        """
        mode = mode or self.align_config.correlation_mode

        # Normalize signals
        signal1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-8)
        signal2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-8)

        # Calculate cross-correlation
        correlation = correlate(signal1_norm, signal2_norm, mode=mode)

        return correlation

    def find_best_shift(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        max_shift: Optional[int] = None
    ) -> int:
        """
        Find the best time shift using cross-correlation.

        Args:
            predicted: Predicted signal
            target: Target signal
            max_shift: Maximum shift in samples

        Returns:
            Best shift in samples (positive = predicted is ahead)
        """
        if max_shift is None:
            max_shift = self.calculate_time_shift_samples()

        # Calculate full cross-correlation
        correlation = self.cross_correlate(target, predicted, mode='full')

        # Find the center (zero shift position)
        center = len(correlation) // 2

        # Restrict search to ±max_shift
        start = max(0, center - max_shift)
        end = min(len(correlation), center + max_shift + 1)

        # Find the peak within the allowed range
        search_region = correlation[start:end]
        best_idx = np.argmax(search_region)

        # Convert to shift value
        shift = best_idx + start - center

        return shift

    def apply_shift(
        self,
        signal: np.ndarray,
        shift: int,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Apply time shift to signal.

        Args:
            signal: Input signal
            shift: Shift in samples (positive = shift right)
            fill_value: Value for padded regions

        Returns:
            Shifted signal
        """
        shifted = np.full_like(signal, fill_value)

        if shift > 0:
            # Shift right
            shifted[shift:] = signal[:-shift]
        elif shift < 0:
            # Shift left
            shifted[:shift] = signal[-shift:]
        else:
            # No shift
            shifted = signal.copy()

        return shifted

    def align_voltage(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        max_voltage_shift: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Align voltage levels between predicted and target signals.

        Args:
            predicted: Predicted signal
            target: Target signal
            max_voltage_shift: Maximum voltage shift in mV

        Returns:
            Voltage-aligned predicted signal and the shift applied
        """
        max_voltage_shift = max_voltage_shift or self.align_config.max_voltage_shift

        # Calculate mean difference
        voltage_shift = np.mean(target) - np.mean(predicted)

        # Clip to maximum allowed shift
        voltage_shift = np.clip(voltage_shift, -max_voltage_shift, max_voltage_shift)

        # Apply shift
        aligned = predicted + voltage_shift

        return aligned, voltage_shift

    def align_signals(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        align_time: bool = True,
        align_voltage: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Align predicted signal to target signal.

        Args:
            predicted: Predicted signal
            target: Target signal
            align_time: Whether to align time (horizontal shift)
            align_voltage: Whether to align voltage (vertical shift)

        Returns:
            Aligned predicted signal and alignment parameters
        """
        aligned = predicted.copy()
        params = {}

        # Time alignment
        if align_time:
            shift = self.find_best_shift(aligned, target)
            aligned = self.apply_shift(aligned, shift)
            params['time_shift_samples'] = shift
            params['time_shift_seconds'] = shift / self.data_config.sampling_rate
        else:
            params['time_shift_samples'] = 0
            params['time_shift_seconds'] = 0.0

        # Voltage alignment
        if align_voltage:
            aligned, voltage_shift = self.align_voltage(aligned, target)
            params['voltage_shift_mv'] = voltage_shift
        else:
            params['voltage_shift_mv'] = 0.0

        return aligned, params

    def align_multi_lead(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        lead_names: Optional[list] = None
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Align multiple leads.

        Args:
            predicted: Predicted signals (num_leads, signal_length)
            target: Target signals (num_leads, signal_length)
            lead_names: Names of leads

        Returns:
            Aligned predicted signals and parameters for each lead
        """
        lead_names = lead_names or self.data_config.lead_names

        num_leads = predicted.shape[0]
        aligned_signals = np.zeros_like(predicted)
        all_params = {}

        for i in range(num_leads):
            aligned_signals[i], params = self.align_signals(
                predicted[i],
                target[i]
            )

            lead_name = lead_names[i] if i < len(lead_names) else f"Lead_{i}"
            all_params[lead_name] = params

        return aligned_signals, all_params

    def optimize_alignment(
        self,
        predicted: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize alignment using gradient-based optimization.

        Args:
            predicted: Predicted signal
            target: Target signal

        Returns:
            Aligned predicted signal and optimal parameters
        """
        if not self.align_config.use_optimization:
            return self.align_signals(predicted, target)

        # Get constraints
        max_time_shift_samples = self.calculate_time_shift_samples()
        max_voltage_shift = self.align_config.max_voltage_shift

        # Define objective function (negative correlation)
        def objective(params):
            time_shift, voltage_shift = params

            # Apply shifts
            shifted = self.apply_shift(predicted, int(time_shift))
            shifted = shifted + voltage_shift

            # Calculate negative correlation (to minimize)
            corr = np.corrcoef(shifted, target)[0, 1]

            return -corr

        # Initial guess
        x0 = [0, 0]

        # Bounds
        bounds = [
            (-max_time_shift_samples, max_time_shift_samples),
            (-max_voltage_shift, max_voltage_shift)
        ]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        # Apply optimal shifts
        optimal_time_shift = int(result.x[0])
        optimal_voltage_shift = result.x[1]

        aligned = self.apply_shift(predicted, optimal_time_shift)
        aligned = aligned + optimal_voltage_shift

        params = {
            'time_shift_samples': optimal_time_shift,
            'time_shift_seconds': optimal_time_shift / self.data_config.sampling_rate,
            'voltage_shift_mv': optimal_voltage_shift,
            'correlation': -result.fun
        }

        return aligned, params


def align_signal(
    predicted: np.ndarray,
    target: np.ndarray,
    config=None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience function to align a single signal.

    Args:
        predicted: Predicted signal
        target: Target signal
        config: Configuration object

    Returns:
        Aligned predicted signal and alignment parameters
    """
    aligner = SignalAligner(config)
    return aligner.align_signals(predicted, target)


def align_multi_lead_signals(
    predicted: np.ndarray,
    target: np.ndarray,
    config=None
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Convenience function to align multiple leads.

    Args:
        predicted: Predicted signals (num_leads, signal_length)
        target: Target signals (num_leads, signal_length)
        config: Configuration object

    Returns:
        Aligned predicted signals and parameters for each lead
    """
    aligner = SignalAligner(config)
    return aligner.align_multi_lead(predicted, target)


if __name__ == "__main__":
    # Test signal alignment
    print("Testing Signal Alignment...")

    # Create synthetic signals
    t = np.linspace(0, 10, 5000)
    target = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

    # Create shifted and scaled version
    shift_samples = 50
    voltage_offset = 0.3
    predicted = np.roll(target, shift_samples) + voltage_offset + 0.1 * np.random.randn(len(target))

    # Align
    aligner = SignalAligner()
    aligned, params = aligner.align_signals(predicted, target)

    print(f"Detected time shift: {params['time_shift_samples']} samples "
          f"({params['time_shift_seconds']:.3f} seconds)")
    print(f"Detected voltage shift: {params['voltage_shift_mv']:.3f} mV")
    print(f"Expected time shift: {-shift_samples} samples")
    print(f"Expected voltage shift: {-voltage_offset:.3f} mV")

    # Calculate correlation
    corr_before = np.corrcoef(predicted, target)[0, 1]
    corr_after = np.corrcoef(aligned, target)[0, 1]

    print(f"\nCorrelation before alignment: {corr_before:.4f}")
    print(f"Correlation after alignment: {corr_after:.4f}")
